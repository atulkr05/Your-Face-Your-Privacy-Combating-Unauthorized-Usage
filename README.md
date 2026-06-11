# Your-Face-Your-Privacy-Combating-Unauthorized-Usage
Your Face Your Privacy: Combating Unauthorized Usage


1. **Face patch generation** using dlib facial landmarks.
2. **Face recognition / verification** using DeepFace embeddings.

---

## 1. Face Patch Generation

The patch script applies different occlusion or replacement patches on detected faces. It uses the 68-point dlib landmark model to locate the face region and place the patch consistently.

### Supported patch types

- `black_rect`
- `black_oval`
- `cheek_rect`
- `cheek_oval`
- `gray_rect`
- `gray_oval`
- `grayscale_rect`
- `grayscale_oval`
- `face_pixel_rect`
- `face_pixel_oval`
- `eyes_black`
- `object`
- `object_mask`

### Basic usage

```bash
python face_patches.py \
  --input "/path/Probe Set" \
  --output "/path/output" \
  --predictor "/path/shape_predictor_68_face_landmarks.dat" \
  --patch black_rect
```

### Run all patches

```bash
python face_patches.py \
  --input "/path/Probe Set" \
  --output "/path/all_patches_output" \
  --predictor "/path/shape_predictor_68_face_landmarks.dat" \
  --patch all_complete \
  --reference-image "/path/reference.jpg" \
  --overlay "/path/hand.png"
```

### Extra arguments

- `--reference-image`: required for `face_pixel_rect` and `face_pixel_oval`.
- `--overlay`: required for `object` and `object_mask`.
- `--images`: list of image names to process. Default is `2.jpg 3.jpg 4.jpg`.

### Output behavior

The script preserves the folder structure of the input dataset and writes patched images into the output directory. Output filenames include a suffix that identifies the patch type.

---

## 2. Face Recognition / Verification

The face recognition script extracts embeddings using DeepFace and compares gallery and probe images using a distance metric. It supports multiple models and writes results to log files and a CSV summary.

### Supported models

- `ArcFace`
- `Dlib`
- `VGG-Face`
- `SFace`
- `Facenet`
- `Facenet512`
- `DeepFace`

### Distance metric

The refined script currently uses:

- `cosine`

### Usage

```bash
python deep_face_recognition_csv_only.py \
  /path/to/gallery \
  /path/to/probe \
  /path/to/logs \
  --output_csv verification_results.csv
```

### Folder structure

Both gallery and probe directories should contain one subfolder per subject identity.

Example:

```text
gallery/
  subject1/
    img1.jpg
  subject2/
    img1.jpg

probe/
  subject1/
    img1.jpg
    img2.jpg
  subject2/
    img1.jpg
```

### Outputs

- Per-model matched logs.
- Per-model non-matched logs.
- One CSV summary file.

### Why CSV only

CSV is easier to automate, simpler to inspect, and enough for most result analysis. If you later need formatted reports, you can generate them from the same CSV.

---

## Requirements

Suggested Python packages:

```bash
pip install opencv-python dlib deepface numpy pandas scipy
```

### Notes

- `dlib` may require system dependencies on some platforms.
- DeepFace can download model weights on first use.
- For best results, ensure the face images are reasonably aligned and clear.

---

## Recommended files

- `face_patches.py`
- `deep_face_recognition_csv_only.py`
- `shape_predictor_68_face_landmarks.dat`
- `README.md`

---

## Typical workflow

1. Generate patched face images.
2. Organize patched images into gallery and probe folders.
3. Run the recognition script on the prepared folders.
4. Review logs and the CSV summary file.

