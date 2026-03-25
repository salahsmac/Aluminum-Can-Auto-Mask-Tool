# Aluminum Can Auto Mask Tool


![sample_sheet_selected](https://github.com/user-attachments/assets/ffea0a1d-3453-4803-9c24-e6c7275ca372)

This project contains a local Python tool for automatically masking aluminum cans from images.

The script scans an input folder, creates a binary mask for each image, and also saves:

- transparent cutouts
- overlay previews
- a CSV manifest
- sample sheets for quick visual checking

The main script is:

`mask_aluminum_cans.py`

## What This Tool Does

The tool is designed for batch processing images of aluminum cans and similar metal containers.

It works well for:

- centered can photos on simple backgrounds
- tabletop photos
- wood, fabric, or plain-floor backgrounds
- white-background product images
- images containing more than one can

It does not require downloading a segmentation model. It uses classical computer vision with OpenCV, saliency detection, and GrabCut refinement.

## Features

- Batch processes an entire folder recursively
- Supports `.jpg`, `.jpeg`, `.png`, `.bmp`, and `.webp`
- Generates one mask per source image
- Creates RGBA cutouts with transparent background
- Creates overlays so you can inspect mask quality quickly
- Writes a `manifest.csv` with paths and mask statistics
- Builds a sample sheet automatically
- Preserves the input folder structure inside the output folders

## How It Works

At a high level, the script:

1. Estimates the background from border pixels.
2. Detects visually distinct foreground regions.
3. Uses saliency to help find likely objects.
4. Refines the result with GrabCut.
5. Cleans small fragments and fills mask holes.

This makes it lightweight and easy to run locally, but it is still heuristic-based. Some reflective, low-contrast, shadow-heavy, or watermarked images may need manual cleanup.

## Requirements

- Python 3.9+
- `opencv-contrib-python`
- `numpy`
- `Pillow`

Install dependencies with:

```powershell
pip install opencv-contrib-python numpy pillow
```

Note:

`opencv-contrib-python` is important because the script uses OpenCV saliency modules that are not included in the base `opencv-python` package.

## Project Layout

Current key files and folders:

```text
first dataset/
├── aluminum_can/
├── aluminum_can_auto_mask/
├── mask_aluminum_cans.py
└── README.md
```

## Basic Usage

Run the script from PowerShell:

```powershell
python .\mask_aluminum_cans.py
```

By default, it reads from:

`.\aluminum_can`

and writes to:

`.\aluminum_can_auto_mask`

## Command Line Options

### Use the default folder

```powershell
python .\mask_aluminum_cans.py
```

### Process a different input folder

```powershell
python .\mask_aluminum_cans.py --input-dir "C:\path\to\images"
```

### Write outputs to a custom location

```powershell
python .\mask_aluminum_cans.py --input-dir "C:\path\to\images" --output-dir "C:\path\to\output"
```

### Run a quick test on only a few images

```powershell
python .\mask_aluminum_cans.py --max-images 25
```

### Change the number of preview samples

```powershell
python .\mask_aluminum_cans.py --sample-count 12
```

## Output Structure

After running, the output folder will look like this:

```text
aluminum_can_auto_mask/
├── cutouts/
├── masks/
├── overlays/
├── samples/
│   └── sample_sheet.png
└── manifest.csv
```

### Output Types

`masks/`

- Binary mask images
- White or non-zero pixels represent the detected object
- Saved as `.png`

`cutouts/`

- RGBA images
- Background is transparent
- Useful for dataset cleanup, annotation review, and asset extraction

`overlays/`

- Original image with the mask drawn on top
- Green fill shows the masked region
- Red outline shows the detected contour

`manifest.csv`

- Lists source file path
- Lists generated mask, cutout, and overlay paths
- Includes `mask_pixels`
- Includes `mask_ratio`

`samples/`

- Visual summary sheet for fast quality review
- The script generates `sample_sheet.png` automatically
- You can create extra custom sample sheets separately if you want

## Example Workflow

1. Put source images in `aluminum_can/`.
2. Run:

```powershell
python .\mask_aluminum_cans.py
```

3. Open:

- `aluminum_can_auto_mask\overlays`
- `aluminum_can_auto_mask\samples\sample_sheet.png`

4. Review any bad masks and manually fix only those cases if needed.

## When This Tool Works Best

- Object is clearly visible
- Background is different from the can
- Can is near the center or isolated
- Lighting is not too harsh
- There are not too many strong reflections around the object

## Known Limitations

- Reflective metal can blend into bright backgrounds
- Strong shadows may be included in the mask
- Watermarks, borders, or stock-photo graphics can confuse segmentation
- Very cluttered scenes can produce false positives
- Extremely small objects may be under-segmented

If you need pixel-perfect masks for difficult images, this tool is best used as a fast first pass rather than a final manual-quality segmentation system.

## Troubleshooting

### `ModuleNotFoundError: No module named 'cv2'`

Install dependencies:

```powershell
pip install opencv-contrib-python numpy pillow
```

### `AttributeError` related to `cv2.saliency`

You likely installed `opencv-python` instead of `opencv-contrib-python`.

Fix:

```powershell
pip uninstall opencv-python
pip install opencv-contrib-python
```

### No files are generated

Check that:

- the input folder exists
- it contains supported image files
- the script has permission to write to the output folder

### Masks are too loose or too tight

This tool is heuristic-based. For hard images, review the overlays and manually correct the small number of bad cases.

## Notes

- The script processes images recursively.
- Relative folder structure is preserved inside the output folders.
- Re-running the script will overwrite existing generated files with updated outputs.

## File Reference

- Script: [mask_aluminum_cans.py](c:/Users/STON/Desktop/projects/G%20PROJECT/DATASET%20V1/first%20dataset/mask_aluminum_cans.py)
- Default output folder: [aluminum_can_auto_mask](c:/Users/STON/Desktop/projects/G%20PROJECT/DATASET%20V1/first%20dataset/aluminum_can_auto_mask)
"# Aluminum-Can-Auto-Mask-Tool" 
"# Aluminum-Can-Auto-Mask-Tool" 
