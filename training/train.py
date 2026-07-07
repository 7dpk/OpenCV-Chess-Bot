import argparse
from pathlib import Path

import cv2
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from chessbot.vision.position import CLASSES


class SquaresDataset(Dataset):
    def __init__(self, root: Path):
        self.samples = []
        for idx, cls in enumerate(CLASSES):
            for path in sorted((Path(root) / cls).glob("*.png")):
                self.samples.append((path, idx))
        if not self.samples:
            raise ValueError(f"no samples under {root}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, label = self.samples[i]
        img = cv2.imread(str(path))
        tensor = torch.from_numpy(img.transpose(2, 0, 1)).float() / 255.0
        return tensor, label


class PieceNet(nn.Module):
    def __init__(self, n_classes: int = 13):
        super().__init__()

        def block(cin, cout):
            return [nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(), nn.MaxPool2d(2)]

        self.features = nn.Sequential(*block(3, 24), *block(24, 48), *block(48, 96), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(96, n_classes)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def _accuracy(model, loader, device) -> float:
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            pred = model(x.to(device)).argmax(dim=1).cpu()
            correct += int((pred == y).sum())
            total += len(y)
    return correct / max(1, total)


def _pick_device(name: str) -> str:
    if name != "auto":
        return name
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def train_model(data_root: Path, epochs: int = 12, batch_size: int = 256, lr: float = 1e-3,
                 device: str = "auto"):
    device = _pick_device(device)
    train_loader = DataLoader(SquaresDataset(Path(data_root) / "train"), batch_size=batch_size,
                               shuffle=True, num_workers=2)
    val_loader = DataLoader(SquaresDataset(Path(data_root) / "val"), batch_size=batch_size, num_workers=2)
    model = PieceNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state = 0.0, None
    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        acc = _accuracy(model, val_loader, device)
        print(f"epoch {epoch + 1}/{epochs}: val acc {acc:.4%}")
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model.cpu(), best_acc


def export_onnx(model, out_path: Path) -> None:
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        torch.zeros(1, 3, 48, 48),
        str(out_path),
        input_names=["squares"],
        output_names=["logits"],
        dynamic_axes={"squares": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=12,
        dynamo=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the piece classifier and export ONNX")
    parser.add_argument("--data", type=Path, default=Path("training/dataset"))
    parser.add_argument("--out", type=Path, default=Path("models/piece_classifier.onnx"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    model, acc = train_model(args.data, args.epochs, args.batch, args.lr, args.device)
    export_onnx(model, args.out)
    print(f"exported {args.out} (best val acc {acc:.4%})")


if __name__ == "__main__":
    main()
