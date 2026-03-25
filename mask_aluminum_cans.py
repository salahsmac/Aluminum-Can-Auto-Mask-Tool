from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class OutputRecord:
    source: Path
    mask: Path
    cutout: Path
    overlay: Path
    mask_pixels: int
    mask_ratio: float


def parse_args() -> argparse.Namespace:
    default_input = Path(__file__).resolve().parent / "aluminum_can"
    parser = argparse.ArgumentParser(
        description="Automatically segment aluminum cans and save masks, cutouts, overlays, and sample previews."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help=f"Folder with source images. Default: {default_input}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output root. Default: <input-dir>_auto_mask",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=8,
        help="Number of examples to include in the sample sheet.",
    )
    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Optional cap for quick testing.",
    )
    return parser.parse_args()


def iter_images(input_dir: Path) -> list[Path]:
    return [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]


def ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def background_distance(lab_image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h, w = lab_image.shape[:2]
    border_width = max(8, min(h, w) // 18)
    border_pixels = np.concatenate(
        [
            lab_image[:border_width, :, :].reshape(-1, 3),
            lab_image[-border_width:, :, :].reshape(-1, 3),
            lab_image[:, :border_width, :].reshape(-1, 3),
            lab_image[:, -border_width:, :].reshape(-1, 3),
        ],
        axis=0,
    ).astype(np.float32)

    cluster_count = min(6, max(2, len(border_pixels) // 5000 + 2))
    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.5,
    )
    _compactness, _labels, centers = cv2.kmeans(
        border_pixels,
        cluster_count,
        None,
        criteria,
        2,
        cv2.KMEANS_PP_CENTERS,
    )

    flat = lab_image.reshape(-1, 3).astype(np.float32)
    distances = [
        np.sum((flat - center.astype(np.float32)) ** 2, axis=1) for center in centers
    ]
    min_distance = np.sqrt(np.min(np.stack(distances, axis=1), axis=1)).reshape(h, w)
    return min_distance, border_pixels


def detect_clean_background(border_pixels_lab: np.ndarray) -> bool:
    border_bgr = cv2.cvtColor(
        border_pixels_lab.reshape(-1, 1, 3).astype(np.uint8),
        cv2.COLOR_LAB2BGR,
    )
    border_hsv = cv2.cvtColor(border_bgr, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    bright_ratio = float((border_hsv[:, 2] > 225).mean())
    low_sat_ratio = float((border_hsv[:, 1] < 35).mean())
    return bright_ratio > 0.75 and low_sat_ratio > 0.7


def keep_components(
    mask: np.ndarray,
    bg_distance_norm: np.ndarray,
    saliency_map: np.ndarray,
    clean_background: bool,
) -> np.ndarray:
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    h, w = mask.shape
    min_area_ratio = 0.0008 if clean_background else 0.0012
    min_area = max(32, int(h * w * min_area_ratio))
    kept = np.zeros_like(mask)
    kept_areas: list[int] = []

    for label in range(1, num_labels):
        x, y, box_w, box_h, area = stats[label]
        if area < min_area:
            continue

        component = labels == label
        border_touch = int(x == 0 or y == 0 or x + box_w >= w or y + box_h >= h)
        mean_bg_distance = float(bg_distance_norm[component].mean())
        mean_saliency = float(saliency_map[component].mean())
        score = area / (h * w) + 0.9 * mean_bg_distance + 0.7 * mean_saliency - 0.25 * border_touch

        if clean_background:
            keep = (not border_touch) and (
                mean_bg_distance > 0.08 or mean_saliency > 0.18 or area > h * w * 0.005
            )
        else:
            keep = (not border_touch and (mean_bg_distance > 0.12 or mean_saliency > 0.22)) or score > 0.34

        if keep:
            kept[component] = 255
            kept_areas.append(area)

    if kept.sum() == 0 and num_labels > 1:
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        kept[labels == largest] = 255
        return kept

    if len(kept_areas) > 1:
        largest_kept = max(kept_areas)
        total_kept = sum(kept_areas)
        if largest_kept / max(total_kept, 1) >= 0.8:
            num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(kept, 8)
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            cleaned = np.zeros_like(kept)
            cleaned[labels == largest_label] = 255
            return cleaned

        small_component_floor = max(min_area, int(largest_kept * 0.05))
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(kept, 8)
        cleaned = np.zeros_like(kept)
        for label in range(1, num_labels):
            area = stats[label, cv2.CC_STAT_AREA]
            if area >= small_component_floor:
                cleaned[labels == label] = 255
        if cleaned.sum() > 0:
            kept = cleaned

    return kept


def fill_holes(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    flood = mask.copy()
    flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, flood_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return mask | holes


def build_mask(image_bgr: np.ndarray) -> np.ndarray:
    original_h, original_w = image_bgr.shape[:2]
    scale = min(1.0, 900.0 / max(original_h, original_w))
    working = image_bgr
    if scale < 1.0:
        working = cv2.resize(
            image_bgr,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA,
        )

    hsv = cv2.cvtColor(working, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(working, cv2.COLOR_BGR2LAB)
    saturation = hsv[:, :, 1].astype(np.float32) / 255.0
    value = hsv[:, :, 2].astype(np.float32) / 255.0

    bg_distance, border_pixels = background_distance(lab)
    bg_distance_norm = cv2.normalize(bg_distance, None, 0.0, 1.0, cv2.NORM_MINMAX)
    clean_background = detect_clean_background(border_pixels)

    saliency_detector = cv2.saliency.StaticSaliencyFineGrained_create()
    _success, saliency_map = saliency_detector.computeSaliency(working)
    saliency_map = cv2.GaussianBlur(saliency_map.astype(np.float32), (0, 0), 3)

    bg_hi = np.percentile(bg_distance_norm, 84 if clean_background else 92)
    bg_lo = np.percentile(bg_distance_norm, 35 if clean_background else 52)
    sal_hi = np.percentile(saliency_map, 68 if clean_background else 82)
    sal_lo = np.percentile(saliency_map, 38 if clean_background else 58)

    raw_foreground = ((bg_distance_norm > bg_hi) & (saliency_map > sal_lo)) | (
        bg_distance_norm > np.percentile(bg_distance_norm, 96)
    )
    if clean_background:
        raw_foreground |= (saturation > max(0.08, np.percentile(saturation, 55))) | (value < 0.93)

    raw_mask = raw_foreground.astype(np.uint8) * 255
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_OPEN, kernel)
    raw_mask = cv2.morphologyEx(raw_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    raw_mask = keep_components(raw_mask, bg_distance_norm, saliency_map, clean_background)

    gc_mask = np.full(raw_mask.shape, cv2.GC_PR_BGD, dtype=np.uint8)
    h, w = raw_mask.shape
    margin = max(4, min(h, w) // 30)
    sure_background = np.zeros_like(raw_mask, dtype=bool)
    sure_background[:margin, :] = True
    sure_background[-margin:, :] = True
    sure_background[:, :margin] = True
    sure_background[:, -margin:] = True
    sure_background |= (bg_distance_norm < bg_lo) & (saliency_map < sal_hi * 0.55)

    probable_foreground = cv2.dilate(raw_mask, kernel, iterations=2) > 0
    sure_foreground = cv2.erode(raw_mask, kernel, iterations=1) > 0

    gc_mask[sure_background] = cv2.GC_BGD
    gc_mask[probable_foreground] = cv2.GC_PR_FGD
    gc_mask[sure_foreground] = cv2.GC_FGD

    if sure_foreground.any():
        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                working,
                gc_mask,
                None,
                bg_model,
                fg_model,
                3,
                cv2.GC_INIT_WITH_MASK,
            )
            refined = np.where(
                (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
                255,
                0,
            ).astype(np.uint8)
        except cv2.error:
            refined = raw_mask
    else:
        refined = raw_mask

    refined = cv2.morphologyEx(refined, cv2.MORPH_CLOSE, kernel, iterations=2)
    refined = fill_holes(refined)
    refined = keep_components(refined, bg_distance_norm, saliency_map, clean_background)

    if scale < 1.0:
        refined = cv2.resize(
            refined,
            (original_w, original_h),
            interpolation=cv2.INTER_NEAREST,
        )

    return refined


def build_overlay(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    masked = mask > 0
    overlay[masked] = (
        0.35 * overlay[masked] + 0.65 * np.array([0, 220, 90])
    ).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (255, 60, 60), 2)
    return overlay


def build_cutout(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    rgba = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGBA)
    rgba[:, :, 3] = mask
    return rgba


def save_record(
    source_path: Path,
    input_dir: Path,
    output_root: Path,
    image_bgr: np.ndarray,
    mask: np.ndarray,
) -> OutputRecord:
    relative = source_path.relative_to(input_dir)
    mask_path = output_root / "masks" / relative.with_suffix(".png")
    cutout_path = output_root / "cutouts" / relative.with_suffix(".png")
    overlay_path = output_root / "overlays" / relative.with_suffix(".png")

    ensure_parent(mask_path)
    ensure_parent(cutout_path)
    ensure_parent(overlay_path)

    cv2.imwrite(str(mask_path), mask)
    Image.fromarray(build_overlay(image_bgr, mask)).save(overlay_path)
    Image.fromarray(build_cutout(image_bgr, mask)).save(cutout_path)

    mask_pixels = int(mask.sum() // 255)
    mask_ratio = mask_pixels / float(mask.shape[0] * mask.shape[1])
    return OutputRecord(
        source=source_path,
        mask=mask_path,
        cutout=cutout_path,
        overlay=overlay_path,
        mask_pixels=mask_pixels,
        mask_ratio=mask_ratio,
    )


def choose_samples(records: list[OutputRecord], sample_count: int) -> list[OutputRecord]:
    if not records or sample_count <= 0:
        return []
    if sample_count >= len(records):
        return records

    positions = np.linspace(0, len(records) - 1, sample_count, dtype=int)
    unique_positions = []
    seen = set()
    for position in positions.tolist():
        if position not in seen:
            unique_positions.append(position)
            seen.add(position)
    return [records[index] for index in unique_positions]


def fit_panel(image: Image.Image, panel_size: tuple[int, int]) -> Image.Image:
    panel_w, panel_h = panel_size
    canvas = Image.new("RGBA", panel_size, (245, 245, 245, 255))
    copy = image.copy()
    copy.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
    x = (panel_w - copy.width) // 2
    y = (panel_h - copy.height) // 2
    canvas.paste(copy, (x, y), copy if copy.mode == "RGBA" else None)
    return canvas


def build_sample_sheet(records: list[OutputRecord], output_path: Path) -> None:
    if not records:
        return

    font = ImageFont.load_default()
    panel_w, panel_h = 210, 210
    left_margin = 24
    top_margin = 52
    row_gap = 28
    col_gap = 16
    label_height = 28
    row_height = label_height + panel_h + row_gap
    width = left_margin * 2 + 3 * panel_w + 2 * col_gap
    height = top_margin + len(records) * row_height + 20

    sheet = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.text((left_margin, 18), "Original", fill=(0, 0, 0), font=font)
    draw.text((left_margin + panel_w + col_gap, 18), "Overlay", fill=(0, 0, 0), font=font)
    draw.text((left_margin + 2 * (panel_w + col_gap), 18), "Cutout", fill=(0, 0, 0), font=font)

    for row, record in enumerate(records):
        y = top_margin + row * row_height
        draw.text(
            (left_margin, y),
            f"{record.source.name} | mask={record.mask_ratio:.2%}",
            fill=(0, 0, 0),
            font=font,
        )

        original = fit_panel(Image.open(record.source).convert("RGBA"), (panel_w, panel_h))
        overlay = fit_panel(Image.open(record.overlay).convert("RGBA"), (panel_w, panel_h))
        cutout = fit_panel(Image.open(record.cutout).convert("RGBA"), (panel_w, panel_h))

        image_y = y + label_height
        sheet.paste(original, (left_margin, image_y), original)
        sheet.paste(overlay, (left_margin + panel_w + col_gap, image_y), overlay)
        sheet.paste(cutout, (left_margin + 2 * (panel_w + col_gap), image_y), cutout)

    ensure_parent(output_path)
    sheet.save(output_path)


def write_manifest(records: Iterable[OutputRecord], output_path: Path) -> None:
    ensure_parent(output_path)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["source", "mask", "cutout", "overlay", "mask_pixels", "mask_ratio"]
        )
        for record in records:
            writer.writerow(
                [
                    str(record.source),
                    str(record.mask),
                    str(record.cutout),
                    str(record.overlay),
                    record.mask_pixels,
                    f"{record.mask_ratio:.8f}",
                ]
            )


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_root = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else input_dir.parent / f"{input_dir.name}_auto_mask"
    )

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    images = iter_images(input_dir)
    if args.max_images is not None:
        images = images[: args.max_images]
    if not images:
        raise FileNotFoundError(f"No supported images found in: {input_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    records: list[OutputRecord] = []

    print(f"Input: {input_dir}")
    print(f"Output: {output_root}")
    print(f"Images: {len(images)}")

    for index, image_path in enumerate(images, start=1):
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[skip] unreadable image: {image_path}")
            continue

        mask = build_mask(image)
        record = save_record(image_path, input_dir, output_root, image, mask)
        records.append(record)

        if index == len(images) or index % 25 == 0:
            print(f"[{index}/{len(images)}] processed {image_path.name}")

    manifest_path = output_root / "manifest.csv"
    write_manifest(records, manifest_path)

    sample_records = choose_samples(records, args.sample_count)
    sample_sheet_path = output_root / "samples" / "sample_sheet.png"
    build_sample_sheet(sample_records, sample_sheet_path)

    print(f"Saved manifest: {manifest_path}")
    print(f"Saved sample sheet: {sample_sheet_path}")


if __name__ == "__main__":
    main()
