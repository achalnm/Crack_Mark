import os
import sys
import glob
from pathlib import Path

import numpy as np
from PIL import Image

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).parent))
from model import build_model

DATA_ROOT   = os.environ.get('DATA_ROOT', './DeepCrack')
OUT_PATH    = os.environ.get('OUT_PATH', 'deepcrack_cracknet.pt')
EPOCHS      = int(os.environ.get('EPOCHS', 40))
BATCH_SIZE  = int(os.environ.get('BATCH_SIZE', 4))
LR          = float(os.environ.get('LR', 3e-4))
NUM_WORKERS = int(os.environ.get('NUM_WORKERS', 2))
IMAGE_SIZE  = 512
VAL_FRAC    = 0.1
SEED        = 42

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')


def list_images(folder):
    files = []
    for ext in IMG_EXT:
        files += glob.glob(str(folder / f'*{ext}'))
        files += glob.glob(str(folder / f'*{ext.upper()}'))
    return sorted(set(files))


def pair_by_stem(imgs, masks):
    by_stem = {Path(m).stem: m for m in masks}
    pairs, skipped = [], 0
    for img in imgs:
        stem = Path(img).stem
        m = by_stem.get(stem)
        if m is None:
            for s, c in by_stem.items():
                if s.startswith(stem):
                    m = c
                    break
        if m is None:
            skipped += 1
            continue
        pairs.append((img, m))
    if not pairs and len(imgs) == len(masks):
        pairs = list(zip(imgs, masks))
    if skipped:
        print(f'  {skipped} images had no matching mask and were skipped')
    return pairs


def split(pairs):
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(pairs))
    n_val = max(1, int(len(pairs) * VAL_FRAC))
    val_idx = set(idx[:n_val].tolist())
    return (
        [p for i, p in enumerate(pairs) if i not in val_idx],
        [p for i, p in enumerate(pairs) if i in val_idx],
    )


def discover_pairs(root):
    if (root / 'train_img').is_dir() and (root / 'train_lab').is_dir():
        train = pair_by_stem(list_images(root / 'train_img'), list_images(root / 'train_lab'))
        if (root / 'test_img').is_dir() and (root / 'test_lab').is_dir():
            val = pair_by_stem(list_images(root / 'test_img'), list_images(root / 'test_lab'))
        else:
            train, val = split(train)
        return train, val

    if (root / 'images').is_dir() and (root / 'masks').is_dir():
        return split(pair_by_stem(list_images(root / 'images'), list_images(root / 'masks')))

    raise FileNotFoundError(
        f'No DeepCrack layout found under {root}. '
        f'Rename your folders to images/ and masks/ and try again.'
    )


class CrackDataset(Dataset):
    def __init__(self, pairs, train):
        self.pairs = pairs
        self.train = train
        self.img_tf = transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        img_path, mask_path = self.pairs[i]
        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L').resize((IMAGE_SIZE, IMAGE_SIZE), Image.NEAREST)

        if self.train and np.random.rand() < 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        if self.train and np.random.rand() < 0.5:
            img  = img.transpose(Image.FLIP_TOP_BOTTOM)
            mask = mask.transpose(Image.FLIP_TOP_BOTTOM)

        x = self.img_tf(img)
        y = torch.from_numpy((np.array(mask) > 127).astype(np.int64))
        return x, y


def loss_fn(logits, targets, device):
    w    = torch.tensor([1.0, 10.0], device=device)
    ce   = F.cross_entropy(logits, targets, weight=w)
    p    = torch.softmax(logits, dim=1)[:, 1]
    t    = (targets == 1).float()
    dice = 1.0 - (2.0 * (p * t).sum() + 1e-6) / (p.sum() + t.sum() + 1e-6)
    return 0.5 * ce + 0.5 * dice


@torch.no_grad()
def crack_iou(logits, targets):
    pred  = logits.argmax(dim=1)
    inter = ((pred == 1) & (targets == 1)).sum().item()
    union = ((pred == 1) | (targets == 1)).sum().item()
    return inter / union if union > 0 else 1.0


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
        print('\nNo GPU detected - training on CPU will take many hours. Use Colab or Kaggle.\n')

    root = Path(DATA_ROOT).expanduser().resolve()
    print(f'Dataset: {root}')
    train_pairs, val_pairs = discover_pairs(root)
    print(f'  train: {len(train_pairs)}   val: {len(val_pairs)}')

    if not train_pairs:
        raise SystemExit('No training pairs found - check DATA_ROOT.')

    pin = device.type == 'cuda'
    train_dl = DataLoader(CrackDataset(train_pairs, train=True),
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=pin)
    val_dl   = DataLoader(CrackDataset(val_pairs, train=False),
                          batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=pin)

    model = build_model('default').to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)

    best_iou = -1.0
    print(f'\n{EPOCHS} epochs on {device}  batch={BATCH_SIZE}  lr={LR}\n')

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y, device)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            running += loss.item() * x.size(0)
        sched.step()

        model.eval()
        ious, vl = [], 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                vl    += loss_fn(logits, y, device).item() * x.size(0)
                ious.append(crack_iou(logits, y))

        val_iou = float(np.mean(ious)) if ious else 0.0
        flag = ''
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), OUT_PATH)
            flag = '  *'

        print(f'epoch {epoch:3d}/{EPOCHS}  '
              f'train={running/len(train_pairs):.4f}  '
              f'val={vl/max(1,len(val_pairs)):.4f}  '
              f'iou={val_iou:.4f}{flag}')

    print(f'\nBest IoU: {best_iou:.4f}')
    print(f'Checkpoint: {Path(OUT_PATH).resolve()}')
    print(f'\nCRACKMARK_CKPT={OUT_PATH} python server.py')


if __name__ == '__main__':
    main()
