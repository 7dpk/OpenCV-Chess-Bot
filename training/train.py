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

        self.features = nn.Sequential(*block(3, 32), *block(32, 64), *block(64, 128), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(128, n_classes)

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
                 device: str = "auto", export_path: Path | None = None,
                 ckpt_path: Path | None = None):
    device = _pick_device(device)
    train_loader = DataLoader(SquaresDataset(Path(data_root) / "train"), batch_size=batch_size,
                               shuffle=True, num_workers=2)
    val_loader = DataLoader(SquaresDataset(Path(data_root) / "val"), batch_size=batch_size, num_workers=2)
    model = PieceNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[int(epochs * 0.6), int(epochs * 0.85)], gamma=0.3
    )
    loss_fn = nn.CrossEntropyLoss()
    best_acc, best_state, start_epoch = 0.0, None, 0
    if ckpt_path is not None and Path(ckpt_path).exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model"])
        model.to(device)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch, best_acc, best_state = ckpt["epoch"], ckpt["best_acc"], ckpt["best_state"]
        print(f"resuming at epoch {start_epoch + 1}/{epochs} (best so far {best_acc:.4%})", flush=True)
    for epoch in range(start_epoch, epochs):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(x.to(device)), y.to(device))
            loss.backward()
            optimizer.step()
        scheduler.step()
        acc = _accuracy(model, val_loader, device)
        print(f"epoch {epoch + 1}/{epochs}: val acc {acc:.4%}", flush=True)
        if acc >= best_acc:
            best_acc = acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            if export_path is not None:
                # export on every improvement so an interrupted run still leaves
                # the best model on disk
                snapshot = PieceNet()
                snapshot.load_state_dict(best_state)
                export_onnx(snapshot, export_path)
        if ckpt_path is not None:
            torch.save({
                "model": {k: v.cpu() for k, v in model.state_dict().items()},
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "epoch": epoch + 1,
                "best_acc": best_acc,
                "best_state": best_state,
            }, ckpt_path)
    if best_state is not None:
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
    parser.add_argument("--ckpt", type=Path, help="checkpoint file for resumable training")
    args = parser.parse_args()
    model, acc = train_model(args.data, args.epochs, args.batch, args.lr, args.device,
                             export_path=args.out, ckpt_path=args.ckpt)
    export_onnx(model, args.out)
    print(f"exported {args.out} (best val acc {acc:.4%})")


if __name__ == "__main__":
    main()
