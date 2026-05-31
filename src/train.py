from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import models

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import get_dataloaders

FROZEN_EPOCHS = 5
MODE_CONFIG = {
    "plant": {
        "data_key": "plant_doc_dir",
        "model_dir": "plant_model",
        "figure_prefix": "plant",
    },
    "disease": {
        "data_key": "plant_disease_dir",
        "model_dir": "disease_model",
        "figure_prefix": "disease",
    },
}


def load_config(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(num_classes: int, pretrained: bool = True) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = models.resnet18(weights=weights)
    except AttributeError:
        model = models.resnet18(pretrained=pretrained)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    for name, parameter in model.named_parameters():
        if not name.startswith("fc."):
            parameter.requires_grad = trainable


def get_trainable_parameters(model: nn.Module):
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def create_optimizer(model: nn.Module, learning_rate: float) -> torch.optim.Optimizer:
    return torch.optim.Adam(get_trainable_parameters(model), lr=learning_rate)


def run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
) -> tuple[float, float]:
    if train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in data_loader:
        images = images.to(device)
        labels = labels.to(device)

        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(images)
            loss = criterion(outputs, labels)

            if train:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        predictions = outputs.argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += batch_size

    epoch_loss = total_loss / total
    epoch_accuracy = correct / total
    return epoch_loss, epoch_accuracy


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False


def save_checkpoint(
    path: Path,
    model: nn.Module,
    class_names: list[str],
    epoch: int,
    val_loss: float,
    val_accuracy: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": len(class_names),
            "epoch": epoch,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
        },
        path,
    )


def plot_training_history(
    history: dict[str, list[float]],
    figure_path: Path,
    title: str,
) -> None:
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(epochs, history["train_loss"], label="Train Loss")
    axes[0].plot(epochs, history["val_loss"], label="Val Loss")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, history["train_accuracy"], label="Train Accuracy")
    axes[1].plot(epochs, history["val_accuracy"], label="Val Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(figure_path, dpi=150, bbox_inches="tight")
    plt.close(figure)


def train_model(mode: str, config_path: Path) -> None:
    if mode not in MODE_CONFIG:
        raise ValueError(f"Geçersiz mode: {mode}. 'plant' veya 'disease' kullanın.")

    config = load_config(config_path)
    mode_config = MODE_CONFIG[mode]
    set_seed(config["project"]["seed"])

    data_dir = PROJECT_ROOT / config["paths"][mode_config["data_key"]]
    model_save_path = (
        PROJECT_ROOT / config["paths"]["output_dir"] / mode_config["model_dir"] / "best_model.pt"
    )
    figures_dir = PROJECT_ROOT / config["paths"]["output_dir"] / "figures"
    figure_path = figures_dir / f"{mode_config['figure_prefix']}_training_history.png"

    device = get_device()
    print(f"Cihaz: {device}")
    print(f"Mod: {mode}")
    print(f"Veri klasörü: {data_dir}")

    loaders = get_dataloaders(
        data_dir=data_dir,
        batch_size=config["data"]["batch_size"],
        image_size=config["data"]["image_size"],
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    num_classes = loaders["num_classes"]
    class_names = loaders["class_names"]
    print(f"Sınıf sayısı: {num_classes}")

    model = build_model(num_classes, pretrained=config["model"]["pretrained"])
    set_backbone_trainable(model, trainable=False)
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = create_optimizer(model, learning_rate=config["training"]["frozen_lr"])
    early_stopping = EarlyStopping(patience=config["training"]["early_stopping_patience"])

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_accuracy": [],
        "val_accuracy": [],
    }

    best_val_loss = float("inf")
    total_epochs = config["training"]["epochs"]
    finetune_started = False

    for epoch in range(1, total_epochs + 1):
        if epoch == FROZEN_EPOCHS + 1 and not finetune_started:
            print("\nFine-tuning başlıyor: tüm ağ çözülüyor, LR = "
                  f"{config['training']['finetune_lr']}")
            set_backbone_trainable(model, trainable=True)
            optimizer = create_optimizer(model, learning_rate=config["training"]["finetune_lr"])
            finetune_started = True

        phase = "Frozen FC" if epoch <= FROZEN_EPOCHS else "Full Fine-tune"
        train_loss, train_accuracy = run_epoch(
            model,
            loaders["train"],
            criterion,
            optimizer,
            device,
            train=True,
        )
        val_loss, val_accuracy = run_epoch(
            model,
            loaders["val"],
            criterion,
            optimizer=None,
            device=device,
            train=False,
        )

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_accuracy"].append(val_accuracy)

        print(
            f"Epoch [{epoch}/{total_epochs}] ({phase}) | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.4f}"
        )

        improved = early_stopping.step(val_loss)
        if improved:
            best_val_loss = val_loss
            save_checkpoint(
                model_save_path,
                model,
                class_names,
                epoch,
                val_loss,
                val_accuracy,
            )
            print(f"  -> En iyi model kaydedildi: {model_save_path}")

        if early_stopping.should_stop:
            print(
                f"\nEarly stopping: validation loss {config['training']['early_stopping_patience']} "
                "epoch boyunca iyileşmedi."
            )
            break

    plot_training_history(
        history,
        figure_path,
        title=f"{mode.capitalize()} Model Training History",
    )
    print(f"\nEğitim grafikleri kaydedildi: {figure_path}")
    print(f"En iyi validation loss: {best_val_loss:.4f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bitki / hastalık sınıflandırma modeli eğitimi")
    parser.add_argument(
        "--mode",
        type=str,
        required=True,
        choices=["plant", "disease"],
        help="Eğitilecek model: plant (bitki) veya disease (hastalık)",
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
    train_model(mode=args.mode, config_path=Path(args.config))


if __name__ == "__main__":
    main()
