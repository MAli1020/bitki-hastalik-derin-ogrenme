from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders
from src.train import MODE_CONFIG, build_model, get_device, load_config


def load_checkpoint(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names, checkpoint


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    all_labels: list[int] = []
    all_predictions: list[int] = []

    for images, labels in data_loader:
        images = images.to(device)
        outputs = model(images)
        predictions = outputs.argmax(dim=1).cpu().tolist()

        all_predictions.extend(predictions)
        all_labels.extend(labels.tolist())

    return np.array(all_labels), np.array(all_predictions)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    num_classes: int,
    figure_path: Path,
    title: str,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    labels = list(range(num_classes))
    matrix = confusion_matrix(y_true, y_pred, labels=labels)

    figure_size = max(8, num_classes * 0.45)
    figure, axis = plt.subplots(figsize=(figure_size, figure_size))

    display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=class_names)
    display.plot(
        ax=axis,
        cmap="Blues",
        values_format="d",
        xticks_rotation=90,
        colorbar=False,
    )
    axis.set_title(title)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def evaluate_model(mode: str, config_path: Path) -> None:
    if mode not in MODE_CONFIG:
        raise ValueError(f"Geçersiz mode: {mode}. 'plant' veya 'disease' kullanın.")

    config = load_config(config_path)
    mode_config = MODE_CONFIG[mode]
    device = get_device()

    data_dir = PROJECT_ROOT / config["paths"][mode_config["data_key"]]
    model_path = (
        PROJECT_ROOT / config["paths"]["output_dir"] / mode_config["model_dir"] / "best_model.pt"
    )
    figure_path = (
        PROJECT_ROOT
        / config["paths"]["output_dir"]
        / "figures"
        / f"{mode_config['figure_prefix']}_confusion_matrix.png"
    )

    print(f"Cihaz: {device}")
    print(f"Mod: {mode}")
    print(f"Model: {model_path}")
    print(f"Veri klasörü: {data_dir}")

    loaders = get_dataloaders(
        data_dir=data_dir,
        batch_size=config["data"]["batch_size"],
        image_size=config["data"]["image_size"],
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    test_loader = loaders["test"]
    if test_loader is None:
        raise RuntimeError(f"Test split'i bulunamadı: {data_dir}")

    model, class_names, checkpoint = load_checkpoint(model_path, device)
    num_classes = checkpoint["num_classes"]
    labels = list(range(num_classes))

    y_true, y_pred = collect_predictions(model, test_loader, device)

    test_accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)

    print("\n--- Test Metrikleri ---")
    print(f"Test Accuracy : {test_accuracy:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")

    if "epoch" in checkpoint:
        print(
            f"\nKayıtlı en iyi model -> Epoch: {checkpoint['epoch']}, "
            f"Val Loss: {checkpoint.get('val_loss', 'N/A'):.4f}, "
            f"Val Acc: {checkpoint.get('val_accuracy', 'N/A'):.4f}"
        )

    print("\n--- Sınıf Bazlı Rapor ---")
    print(
        classification_report(
            y_true,
            y_pred,
            labels=labels,
            target_names=class_names,
            zero_division=0,
        )
    )

    plot_confusion_matrix(
        y_true,
        y_pred,
        class_names,
        num_classes,
        figure_path,
        title=f"{mode.capitalize()} Model - Confusion Matrix",
    )
    print(f"\nConfusion matrix kaydedildi: {figure_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eğitilmiş modeli test verisi üzerinde değerlendir")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["plant", "disease"],
        help="Değerlendirilecek model: plant veya disease",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(PROJECT_ROOT / "config.yaml"),
        help="Yapılandırma dosyası yolu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evaluate_model(mode=args.mode, config_path=Path(args.config))


if __name__ == "__main__":
    main()
