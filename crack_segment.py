import cv2
import numpy as np
from pathlib import Path
import argparse
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes, closing, disk
from skimage.filters import apply_hysteresis_threshold


def load_grayscale(path: Path) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        from PIL import Image
        img = np.array(Image.open(path).convert("L"))
    return img


def remove_illumination(gray: np.ndarray, sigma: float = 100) -> np.ndarray:
    g  = gray.astype(np.float32) + 1.0
    bg = cv2.GaussianBlur(g, (0, 0), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101)
    norm = np.clip((g / bg) * 120.0, 0, 255)
    return norm.astype(np.uint8)


def gaussian_tophat(gray: np.ndarray, sigma: float) -> np.ndarray:
    ksize   = max(3, int(sigma * 6 + 1) | 1)
    blurred = cv2.GaussianBlur(gray, (ksize, ksize), sigma)
    return np.clip(blurred.astype(np.int32) - gray.astype(np.int32), 0, 255).astype(np.uint8)


def shape_filter(binary: np.ndarray, gray: np.ndarray) -> np.ndarray:
    labeled = label(binary)
    result  = np.zeros_like(binary, dtype=bool)

    for region in regionprops(labeled):
        pixels    = gray[labeled == region.label]
        non_black = pixels[pixels >= 15]

        if len(non_black) < 50:
            continue
        mean_gray = float(non_black.mean())
        if mean_gray < 30:
            continue

        area = region.area
        ecc  = region.eccentricity
        sol  = region.solidity

        if area >= 20_000 and sol < 0.62:
            result[labeled == region.label] = True
        elif area >= 2_000 and ecc >= 0.92:
            result[labeled == region.label] = True
        elif area >= 400 and ecc >= 0.96:
            result[labeled == region.label] = True

    result &= (gray >= 15)
    return result


def segment_crack(input_path: Path, output_path: Path) -> None:
    gray = load_grayscale(input_path)
    print(f"  Processing: {input_path.name}  ({gray.shape[1]}x{gray.shape[0]}px)")

    denoised = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7, searchWindowSize=21)
    norm     = remove_illumination(denoised, sigma=100)

    saliency = np.maximum(
        gaussian_tophat(norm, sigma=8),
        gaussian_tophat(norm, sigma=16),
    )

    bright     = (norm > 175).astype(np.uint8)
    bright_dil = cv2.dilate(bright, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 31)))
    saliency[bright_dil > 0] = 0

    peak        = int(saliency.max())
    thresh_high = max(8, int(peak * 0.30))
    thresh_low  = max(3, int(peak * 0.10))
    binary = apply_hysteresis_threshold(saliency.astype(np.float64), thresh_low, thresh_high)

    closed = closing(binary, disk(5))
    filled = remove_small_holes(closed, max_size=500)
    result = (shape_filter(filled, gray) * 255).astype(np.uint8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), result)
    print(f"  [OK] Saved: {output_path}")


def process_folder(input_folder: Path, output_folder: Path) -> None:
    extensions = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    images = sorted(
        f for f in input_folder.iterdir()
        if f.is_file() and f.suffix.lower() in extensions
    )
    if not images:
        print(f"No images found in: {input_folder}")
        return

    print(f"\nFound {len(images)} image(s) in '{input_folder}'")
    print(f"Output: '{output_folder}'\n" + "-" * 60)

    ok, err = 0, 0
    for img_path in images:
        out = output_folder / f"{img_path.stem}_crack_mask.png"
        try:
            segment_crack(img_path, out)
            ok += 1
        except Exception as e:
            print(f"  [ERR] {img_path.name}: {e}")
            err += 1

    print("-" * 60)
    print(f"\nDone. {ok} processed" + (f", {err} failed." if err else "."))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  "-i", default="INPUT RAW")
    parser.add_argument("--output", "-o", default="OUTPUT")
    args = parser.parse_args()

    base          = Path(__file__).parent
    input_folder  = (base / args.input).resolve()
    output_folder = (base / args.output).resolve()

    if not input_folder.exists():
        print(f"Error: Folder not found: {input_folder}")
        return

    process_folder(input_folder, output_folder)


if __name__ == "__main__":
    main()
