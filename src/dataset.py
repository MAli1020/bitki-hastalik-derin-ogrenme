from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Callable

import yaml
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_ALIASES = {
    "train": ("train",),
    "val": ("val", "valid", "validation"),
    "test": ("test",),
}
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15


def _list_images(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def _is_class_directory(path: Path) -> bool:
    return path.is_dir() and bool(_list_images(path))


def _has_explicit_splits(root: Path) -> bool:
    return any(_find_split_dir(root, split) is not None for split in ("train", "val", "test"))


def _has_flat_imagefolder(root: Path) -> bool:
    return len(_find_class_directories(root)) > 0


def _find_class_directories(root: Path) -> list[Path]:
    """Kökte veya tek bir alt klasörde (ör. PlantVillage) sınıf dizinlerini bulur."""
    class_dirs = sorted(
        path for path in root.iterdir() if path.is_dir() and _is_class_directory(path)
    )
    if class_dirs:
        return class_dirs

    containers: list[tuple[Path, list[Path]]] = []
    for path in root.iterdir():
        if not path.is_dir() or _is_class_directory(path):
            continue
        nested = sorted(
            child for child in path.iterdir() if child.is_dir() and _is_class_directory(child)
        )
        if nested:
            containers.append((path, nested))

    if not containers:
        return []

    _, best_nested = max(containers, key=lambda item: len(item[1]))
    return best_nested


def _stratified_split_class_samples(
    class_dirs: list[Path],
    seed: int = 42,
) -> tuple[dict[str, list[tuple[Path, int]]], list[str]]:
    """Her sınıftan %70 train, %15 val, %15 test (stratified)."""
    split_samples: dict[str, list[tuple[Path, int]]] = {
        "train": [],
        "val": [],
        "test": [],
    }
    class_names = [class_dir.name for class_dir in class_dirs]

    for label, class_dir in enumerate(class_dirs):
        images = _list_images(class_dir)
        if len(images) < 3:
            raise RuntimeError(
                f"Stratified bölme için sınıfta en az 3 görüntü gerekli: "
                f"{class_dir.name} ({len(images)} görüntü)"
            )

        train_val, test_images = train_test_split(
            images,
            test_size=TEST_RATIO,
            random_state=seed,
            shuffle=True,
        )
        val_ratio_within_train_val = VAL_RATIO / (TRAIN_RATIO + VAL_RATIO)
        train_images, val_images = train_test_split(
            train_val,
            test_size=val_ratio_within_train_val,
            random_state=seed,
            shuffle=True,
        )

        split_samples["train"].extend((path, label) for path in train_images)
        split_samples["val"].extend((path, label) for path in val_images)
        split_samples["test"].extend((path, label) for path in test_images)

    return split_samples, class_names


def _resolve_dataset_root(data_dir: str | Path) -> Path:
    root = Path(data_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"Veri klasörü bulunamadı: {root}")

    archive_dir = root / "archive"
    if (archive_dir / "data.yaml").exists():
        return archive_dir

    if _has_explicit_splits(root) or (root / "data.yaml").exists():
        return root

    if _has_flat_imagefolder(root):
        return root

    raise FileNotFoundError(
        f"Desteklenen veri yapısı bulunamadı: {root}. "
        "train/val/test bölümleri, archive/data.yaml veya düz ImageFolder yapısı bekleniyor."
    )


def _find_split_dir(root: Path, split: str) -> Path | None:
    for name in SPLIT_ALIASES[split]:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return None


def _load_class_names(root: Path) -> list[str] | None:
    yaml_path = root / "data.yaml"
    if not yaml_path.exists():
        return None

    with yaml_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}

    names = config.get("names")
    if isinstance(names, dict):
        return [names[key] for key in sorted(names, key=lambda item: int(item))]
    if isinstance(names, list):
        return names

    return None


def _label_path_for_image(image_path: Path, labels_dir: Path) -> Path:
    return labels_dir / f"{image_path.stem}.txt"


def _class_id_from_yolo_label(label_path: Path) -> int | None:
    with label_path.open("r", encoding="utf-8") as file:
        class_ids = [int(line.split()[0]) for line in file if line.strip()]

    if not class_ids:
        return None

    return Counter(class_ids).most_common(1)[0][0]


def _collect_yolo_samples(split_dir: Path) -> list[tuple[Path, int]]:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"YOLO formatı için images/ ve labels/ klasörleri gerekli: {split_dir}"
        )

    samples: list[tuple[Path, int]] = []
    for image_path in _list_images(images_dir):
        label_path = _label_path_for_image(image_path, labels_dir)
        if not label_path.exists():
            continue
        class_id = _class_id_from_yolo_label(label_path)
        if class_id is None:
            continue
        samples.append((image_path, class_id))

    if not samples:
        raise RuntimeError(f"YOLO split içinde kullanılabilir örnek bulunamadı: {split_dir}")

    return samples


def _collect_imagefolder_samples(split_dir: Path) -> tuple[list[tuple[Path, int]], list[str]]:
    class_dirs = sorted(
        path for path in split_dir.iterdir() if path.is_dir() and _list_images(path)
    )
    if not class_dirs:
        raise RuntimeError(
            f"ImageFolder formatı için sınıf alt klasörleri gerekli: {split_dir}"
        )

    class_names = [path.name for path in class_dirs]
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    samples: list[tuple[Path, int]] = []
    for class_dir in class_dirs:
        label = class_to_idx[class_dir.name]
        for image_path in _list_images(class_dir):
            samples.append((image_path, label))

    return samples, class_names


def _detect_format(split_dir: Path) -> str:
    if (split_dir / "images").is_dir() and (split_dir / "labels").is_dir():
        return "yolo"
    return "imagefolder"


def _collect_split_samples(
    split_dir: Path,
    class_names: list[str] | None,
) -> tuple[list[tuple[Path, int]], list[str]]:
    dataset_format = _detect_format(split_dir)

    if dataset_format == "yolo":
        samples = _collect_yolo_samples(split_dir)
        if not class_names:
            max_label = max(label for _, label in samples)
            class_names = [str(index) for index in range(max_label + 1)]
        return samples, class_names

    samples, split_class_names = _collect_imagefolder_samples(split_dir)
    if class_names is None:
        class_names = split_class_names
    elif class_names != split_class_names:
        name_to_idx = {name: idx for idx, name in enumerate(class_names)}
        missing = [name for name in split_class_names if name not in name_to_idx]
        if missing:
            raise ValueError(
                "ImageFolder sınıf isimleri split'ler arasında tutarsız. "
                f"Eksik sınıflar: {missing}"
            )
        samples = [(path, name_to_idx[class_names[label]]) for path, label in samples]

    return samples, class_names


def get_train_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_eval_transforms(image_size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class PlantDataset(Dataset):
    """Bitki veya hastalık sınıflandırması için PyTorch Dataset."""

    def __init__(
        self,
        samples: list[tuple[Path, int]],
        class_names: list[str],
        transform: Callable | None = None,
    ) -> None:
        self.samples = samples
        self.class_names = class_names
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_dataloaders(
    data_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 0,
    pin_memory: bool = True,
    seed: int = 42,
) -> dict:
    """
    Veri klasöründen train/val/test DataLoader'ları oluşturur.

    Args:
        data_dir: `data/plant_doc/` veya `data/plant_disease/` gibi kök yol.
        batch_size: Batch boyutu.
        image_size: Görüntülerin yeniden boyutlandırılacağı kare kenar uzunluğu.
        num_workers: DataLoader worker sayısı.
        pin_memory: CUDA kullanımında bellek sabitleme.
        seed: Düz ImageFolder yapısında stratified bölme için rastgele tohum.

    Returns:
        train_loader, val_loader, test_loader, class_names ve num_classes içeren sözlük.
    """
    root = _resolve_dataset_root(data_dir)
    class_names = _load_class_names(root)
    split_samples: dict[str, list[tuple[Path, int]]] = {}

    if _has_explicit_splits(root):
        for split in ("train", "val", "test"):
            split_dir = _find_split_dir(root, split)
            if split_dir is None:
                continue
            samples, class_names = _collect_split_samples(split_dir, class_names)
            split_samples[split] = samples
    else:
        class_dirs = _find_class_directories(root)
        if not class_dirs:
            raise RuntimeError(f"Sınıf klasörleri bulunamadı: {root}")
        split_samples, discovered_names = _stratified_split_class_samples(class_dirs, seed=seed)
        if class_names is None:
            class_names = discovered_names

    if "train" not in split_samples:
        raise RuntimeError(f"Eğitim split'i bulunamadı: {root}")
    if "val" not in split_samples:
        raise RuntimeError(f"Doğrulama split'i bulunamadı: {root}")

    assert class_names is not None

    train_dataset = PlantDataset(
        split_samples["train"],
        class_names,
        transform=get_train_transforms(image_size),
    )
    val_dataset = PlantDataset(
        split_samples["val"],
        class_names,
        transform=get_eval_transforms(image_size),
    )

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "class_names": class_names,
        "num_classes": len(class_names),
    }

    if "test" in split_samples:
        test_dataset = PlantDataset(
            split_samples["test"],
            class_names,
            transform=get_eval_transforms(image_size),
        )
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
    else:
        loaders["test"] = None

    return loaders
