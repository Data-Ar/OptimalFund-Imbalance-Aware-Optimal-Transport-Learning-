from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


IMG_EXTS = {".jpg",".jpeg",".png",".bmp",".tif",".tiff",".webp"}


class DRFolderDataset(Dataset):
    def __init__(self, root: str, split: str, transform=None, num_classes: int = 5):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        base = self.root / split
        if not base.exists():
            raise FileNotFoundError(f"Missing split folder: {base}")

        samples = []
        for y in range(num_classes):
            cls_dir = base / str(y)
            if not cls_dir.exists():
                raise FileNotFoundError(f"Missing class folder: {cls_dir}")
            for p in cls_dir.rglob("*"):
                if p.suffix.lower() in IMG_EXTS:
                    samples.append((str(p), y))

        if len(samples) == 0:
            raise RuntimeError(f"No images found under {base}")

        self.samples = samples
        self.labels = np.array([y for _, y in samples], dtype=np.int64)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, int(y), path
