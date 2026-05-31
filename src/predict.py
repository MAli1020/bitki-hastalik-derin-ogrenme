from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_eval_transforms
from src.train import build_model, get_device, load_config


def load_model_from_checkpoint(model_path: Path, device: torch.device):
    if not model_path.exists():
        raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")

    try:
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint["class_names"]
    num_classes = checkpoint["num_classes"]

    model = build_model(num_classes, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    return model, class_names


def preprocess_image(image_path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(image_path).convert("RGB")
    transform = get_eval_transforms(image_size)
    return transform(image).unsqueeze(0)


@torch.no_grad()
def predict_with_names(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    class_names: list[str],
    device: torch.device,
    class_indices: list[int] | None = None,
    top_k: int = 3,
) -> tuple[str, float, list[tuple[str, float]]]:
    image_tensor = image_tensor.to(device)
    logits = model(image_tensor)

    if class_indices is not None:
        masked_logits = torch.full_like(logits, float("-inf"))
        masked_logits[:, class_indices] = logits[:, class_indices]
        logits = masked_logits

    probabilities = F.softmax(logits, dim=1)[0]
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)

    if class_indices is not None:
        top_k = min(top_k, len(class_indices))

    top_predictions: list[tuple[str, float]] = []
    for probability, index in zip(sorted_probs[:top_k], sorted_indices[:top_k]):
        if probability.item() <= 0 or torch.isinf(probability):
            continue
        class_idx = index.item()
        top_predictions.append((class_names[class_idx], probability.item()))

    best_name, best_confidence = top_predictions[0]
    return best_name, best_confidence, top_predictions


def detect_crop_type(plant_class_name: str) -> str | None:
    name = plant_class_name.lower().replace(" ", "_")

    if "tomato" in name:
        return "tomato"
    if "potato" in name:
        return "potato"
    if "pepper" in name or "bell_pepper" in name:
        return "pepper"
    return None


def disease_class_indices_for_crop(class_names: list[str], crop: str) -> list[int]:
    crop = crop.lower()
    indices: list[int] = []

    for index, class_name in enumerate(class_names):
        name = class_name.lower()
        if crop == "tomato" and "tomato" in name:
            indices.append(index)
        elif crop == "potato" and "potato" in name:
            indices.append(index)
        elif crop == "pepper" and "pepper" in name:
            indices.append(index)

    return indices


def format_class_label(label: str) -> str:
    return label.replace("___", " — ").replace("__", " ").replace("_", " ").strip()


def is_healthy_disease_label(label: str) -> bool:
    return "healthy" in label.lower()


def print_header(title: str) -> None:
    line = "=" * 56
    print(f"\n{line}")
    print(title.center(56))
    print(line)


def print_prediction_block(
    stage: str,
    label: str,
    confidence: float,
    top_predictions: list[tuple[str, float]] | None = None,
    extra_lines: list[str] | None = None,
) -> None:
    print(f"\n[{stage}]")
    print(f"  Tahmin    : {format_class_label(label)}")
    print(f"  Güven     : {confidence * 100:.2f}%")

    if extra_lines:
        for line in extra_lines:
            print(f"  {line}")

    if top_predictions and len(top_predictions) > 1:
        print("  Alternatifler:")
        for name, prob in top_predictions[1:]:
            print(f"    - {format_class_label(name)} ({prob * 100:.2f}%)")


def run_prediction(image_path: Path, config_path: Path) -> None:
    config = load_config(config_path)
    device = get_device()
    image_size = config["data"]["image_size"]
    output_dir = PROJECT_ROOT / config["paths"]["output_dir"]

    plant_model_path = output_dir / "plant_model" / "best_model.pt"
    disease_model_path = output_dir / "disease_model" / "best_model.pt"

    if not image_path.exists():
        raise FileNotFoundError(f"Görüntü bulunamadı: {image_path}")

    print_header("Yaprak Analizi — İki Aşamalı Tahmin")
    print(f"\nGörüntü : {image_path.resolve()}")
    print(f"Cihaz   : {device}")

    image_tensor = preprocess_image(image_path, image_size)

    plant_model, plant_class_names = load_model_from_checkpoint(plant_model_path, device)
    plant_label, plant_confidence, plant_top = predict_with_names(
        plant_model,
        image_tensor,
        plant_class_names,
        device,
        top_k=3,
    )

    crop_type = detect_crop_type(plant_label)
    plant_extra = []
    if crop_type:
        crop_tr = {"tomato": "Domates", "potato": "Patates", "pepper": "Biber"}[crop_type]
        plant_extra.append(f"Kültür    : {crop_tr} (hastalık analizi uygulanacak)")
    else:
        plant_extra.append("Kültür    : Domates / patates / biber dışı (hastalık analizi atlandı)")

    print_prediction_block(
        "Aşama 1 — Bitki / yaprak türü",
        plant_label,
        plant_confidence,
        plant_top,
        plant_extra,
    )

    if crop_type is None:
        print("\nSonuç: Bu yaprak için hastalık modeli devreye alınmadı.")
        return

    if not disease_model_path.exists():
        print(
            f"\nUyarı: Hastalık modeli bulunamadı ({disease_model_path}). "
            "Önce `python -m src.train --mode disease` çalıştırın."
        )
        return

    disease_model, disease_class_names = load_model_from_checkpoint(disease_model_path, device)
    crop_indices = disease_class_indices_for_crop(disease_class_names, crop_type)

    if not crop_indices:
        print(f"\nUyarı: Hastalık modelinde '{crop_type}' sınıfları bulunamadı.")
        return

    disease_label, disease_confidence, disease_top = predict_with_names(
        disease_model,
        image_tensor,
        disease_class_names,
        device,
        class_indices=crop_indices,
        top_k=3,
    )

    health_status = "Sağlıklı görünüyor" if is_healthy_disease_label(disease_label) else "Hastalık belirtisi tespit edildi"
    disease_extra = [f"Durum     : {health_status}"]

    print_prediction_block(
        "Aşama 2 — Hastalık / sağlık durumu",
        disease_label,
        disease_confidence,
        disease_top,
        disease_extra,
    )

    print("\n" + "-" * 56)
    crop_tr = {"tomato": "domates", "potato": "patates", "pepper": "biber"}[crop_type]
    print(
        f"Özet: {format_class_label(plant_label)} ({plant_confidence * 100:.1f}%) → "
        f"{format_class_label(disease_label)} ({disease_confidence * 100:.1f}%)"
    )
    print(f"      ({crop_tr} için hastalık analizi tamamlandı)")
    print("-" * 56 + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Yaprak görüntüsü üzerinde iki aşamalı bitki ve hastalık tahmini"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Analiz edilecek yaprak görüntüsünün dosya yolu",
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
    run_prediction(
        image_path=Path(args.image).expanduser().resolve(),
        config_path=Path(args.config),
    )


if __name__ == "__main__":
    main()
