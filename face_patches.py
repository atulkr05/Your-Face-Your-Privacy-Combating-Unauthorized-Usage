
"""
Usage Examples:

1. Simple black rectangle patch on all images:
   python face_patches.py --patch black_rect --image_dir ./images

2. Patch with custom output directory:
   python face_patches.py --patch cheek_rect --image_dir ./images --output_dir ./results

3. Applying an object overlay (phone.png):
   python face_patches.py --patch object --image_dir ./images --overlay ./phone.png

4. Creating a grayscale patch with custom suffix:
   python face_patches.py --patch grayscale_oval --image_dir ./images --suffix _custom.jpg

5. Preserving output filename while changing patch type:
   python face_patches.py --patch black_oval --image_dir ./images/person1.jpg --output ./images/person1.jpg

6. Processing multiple face images:
   python face_patches.py --patch gray_rect --image_dir ./ Dataset_faces

7. Using different landmark predictor:
   python face_patches.py --patch cheek_oval --image_dir ./images --predictor_path /path/to/predictor.dat


Common Options:
  -p, --patch        Type of patch to apply (default: grayscale_oval)
  -i, --image_dir    Directory containing images (default: Dataset_faces)
  -o, --output       Output image path or directory (default: Dataset_faces_modified)
  -s, --suffix       Suffix to append to output filenames (default: changes based on patch type)
  -l, --landmarks    Path to dlib landmark predictor (default: shape_predictor_68_face_landmarks.dat)
  -r, --reference    Reference image for face pixel color (default: None)
  -ov, --overlay     Overlay image for object patch (default: None)

Patch Types:
  - black_rect: Black rectangle over mouth
  - black_oval: Black oval over mouth
  - cheek_rect: Cheek color rectangle
  - cheek_oval: Cheek color oval
  - gray_rect: Grayscale rectangle
  - gray_oval: Grayscale oval (default)
  - grayscale_rect: Grayscale rectangle (alternative)
  - grayscale_oval: Grayscale oval (alternative)
  - face_pixel_rect: Face pixel color rectangle
  - face_pixel_oval: Face pixel color oval
  - eyes_black: Black patches over eyes
  - object: Overlay object on face
  - object_mask: Generate mask for object patch
"""




import os
import argparse
from pathlib import Path

import cv2
import dlib
import numpy as np


def detect_landmarks(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        return predictor(gray, face)
    return None


def face_width_from_landmarks(landmarks):
    return landmarks.part(45).x - landmarks.part(36).x


def mouth_region_rect(landmarks, width_scale=0.25):
    face_width = face_width_from_landmarks(landmarks)
    rect_width = int(face_width * width_scale)
    center_x = (landmarks.part(27).x + landmarks.part(30).x) // 2
    top_left = (center_x - rect_width // 2, landmarks.part(27).y)
    bottom_right = (center_x + rect_width // 2, landmarks.part(57).y)
    return top_left, bottom_right


def mouth_region_ellipse(landmarks, width_scale=0.40, height_scale=1.0):
    face_width = face_width_from_landmarks(landmarks)
    ellipse_width = int(face_width * width_scale)
    ellipse_height = int(face_width * height_scale)
    center_x = (landmarks.part(27).x + landmarks.part(30).x) // 2
    center_y = (landmarks.part(27).y + landmarks.part(57).y) // 2
    return (center_x, center_y), (ellipse_width // 2, ellipse_height // 2)


def both_eye_rect(landmarks):
    points = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 48)]
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return (min(xs), min(ys)), (max(xs), max(ys))


def sample_cheek_color(image, landmarks):
    return tuple(image[landmarks.part(29).y, landmarks.part(29).x].tolist())


def get_reference_face_pixel(reference_image, detector, predictor):
    landmarks = detect_landmarks(reference_image, detector, predictor)
    if landmarks is None:
        return None
    pos = (landmarks.part(2).x, landmarks.part(2).y)
    return tuple(reference_image[pos[1], pos[0]].tolist())


def apply_black_rect(image, landmarks):
    top_left, bottom_right = mouth_region_rect(landmarks, 0.25)
    cv2.rectangle(image, top_left, bottom_right, (0, 0, 0), -1)
    return image


def apply_black_oval(image, landmarks):
    center, axes = mouth_region_ellipse(landmarks, 0.40, 1.0)
    cv2.ellipse(image, center, axes, 0, 0, 360, (0, 0, 0), -1)
    return image


def apply_cheek_rect(image, landmarks):
    color = sample_cheek_color(image, landmarks)
    top_left, bottom_right = mouth_region_rect(landmarks, 0.25)
    cv2.rectangle(image, top_left, bottom_right, color, -1)
    return image


def apply_cheek_oval(image, landmarks):
    color = sample_cheek_color(image, landmarks)
    center, axes = mouth_region_ellipse(landmarks, 0.40, 1.0)
    cv2.ellipse(image, center, axes, 0, 0, 360, color, -1)
    return image


def apply_gray_rect(image, landmarks, gray_color=(127, 127, 127)):
    top_left, bottom_right = mouth_region_rect(landmarks, 0.25)
    cv2.rectangle(image, top_left, bottom_right, gray_color, -1)
    return image


def apply_gray_oval(image, landmarks, gray_color=(127, 127, 127)):
    center, axes = mouth_region_ellipse(landmarks, 0.40, 1.0)
    cv2.ellipse(image, center, axes, 0, 0, 360, gray_color, -1)
    return image


def apply_grayscale_rect(image, landmarks):
    top_left, bottom_right = mouth_region_rect(landmarks, 0.25)
    x1, y1 = top_left
    x2, y2 = bottom_right
    patch = image[y1:y2 + 1, x1:x2 + 1]
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    image[y1:y2 + 1, x1:x2 + 1] = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return image


def apply_grayscale_oval(image, landmarks):
    center, axes = mouth_region_ellipse(landmarks, 0.40, 1.0)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.ellipse(mask, center, axes, 0, 0, 360, 255, -1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    image[mask == 255] = gray_bgr[mask == 255]
    return image


def apply_face_pixel_rect(image, landmarks, reference_color):
    top_left, bottom_right = mouth_region_rect(landmarks, 0.25)
    cv2.rectangle(image, top_left, bottom_right, reference_color, -1)
    return image


def apply_face_pixel_oval(image, landmarks, reference_color):
    center, axes = mouth_region_ellipse(landmarks, 0.40, 1.0)
    cv2.ellipse(image, center, axes, 0, 0, 360, reference_color, -1)
    return image


def overlay_rgba(base_image, overlay_rgba, x, y):
    h, w = overlay_rgba.shape[:2]
    for oy in range(h):
        for ox in range(w):
            iy, ix = y + oy, x + ox
            if 0 <= iy < base_image.shape[0] and 0 <= ix < base_image.shape[1]:
                pixel = overlay_rgba[oy, ox]
                if overlay_rgba.shape[2] == 4:
                    alpha = pixel[3] / 255.0
                    if alpha > 0:
                        base_image[iy, ix] = (alpha * pixel[:3] + (1 - alpha) * base_image[iy, ix]).astype(np.uint8)
                else:
                    if np.any(pixel != 0):
                        base_image[iy, ix] = pixel[:3]
    return base_image


def apply_object_patch(image, landmarks, overlay_image, increase_size_percent=40):
    face_width = landmarks.part(54).x - landmarks.part(48).x
    top_y = landmarks.part(29).y
    bottom_y = landmarks.part(57).y
    rect_height = bottom_y - top_y
    factor = 1 + (increase_size_percent / 100.0)
    face_width = int(face_width * factor)
    rect_height = int(rect_height * factor)
    resized = cv2.resize(overlay_image, (face_width, rect_height))
    x = landmarks.part(48).x - int((face_width - (landmarks.part(54).x - landmarks.part(48).x)) / 2)
    y = top_y - int((rect_height - (bottom_y - top_y)) / 2)
    return overlay_rgba(image, resized, x, y)


def build_object_mask(image, landmarks, overlay_image, increase_size_percent=40):
    mask = np.ones_like(image) * 255
    face_width = landmarks.part(54).x - landmarks.part(48).x
    top_y = landmarks.part(29).y
    bottom_y = landmarks.part(57).y
    rect_height = bottom_y - top_y
    factor = 1 + (increase_size_percent / 100.0)
    face_width = int(face_width * factor)
    rect_height = int(rect_height * factor)
    resized = cv2.resize(overlay_image, (face_width, rect_height))
    x = landmarks.part(48).x - int((face_width - (landmarks.part(54).x - landmarks.part(48).x)) / 2)
    y = top_y - int((rect_height - (bottom_y - top_y)) / 2)
    for oy in range(resized.shape[0]):
        for ox in range(resized.shape[1]):
            iy, ix = y + oy, x + ox
            if 0 <= iy < mask.shape[0] and 0 <= ix < mask.shape[1]:
                pixel = resized[oy, ox]
                if resized.shape[2] == 4:
                    if pixel[3] != 0:
                        mask[iy, ix] = 0
                else:
                    if np.any(pixel[:3] != 0):
                        mask[iy, ix] = 0
    return mask


PATCH_BUILDERS = {
    "black_rect": lambda img, lm, args, det, pred: apply_black_rect(img, lm),
    "black_oval": lambda img, lm, args, det, pred: apply_black_oval(img, lm),
    "cheek_rect": lambda img, lm, args, det, pred: apply_cheek_rect(img, lm),
    "cheek_oval": lambda img, lm, args, det, pred: apply_cheek_oval(img, lm),
    "gray_rect": lambda img, lm, args, det, pred: apply_gray_rect(img, lm),
    "gray_oval": lambda img, lm, args, det, pred: apply_gray_oval(img, lm),
    "grayscale_rect": lambda img, lm, args, det, pred: apply_grayscale_rect(img, lm),
    "grayscale_oval": lambda img, lm, args, det, pred: apply_grayscale_oval(img, lm),
    "face_pixel_rect": lambda img, lm, args, det, pred: apply_face_pixel_rect(img, lm, args.reference_color),
    "face_pixel_oval": lambda img, lm, args, det, pred: apply_face_pixel_oval(img, lm, args.reference_color),
    "object": lambda img, lm, args, det, pred: apply_object_patch(img, lm, args.overlay_image, args.increase_size_percent),
    "hand": lambda img, lm, args, det, pred: apply_object_patch(img, lm, args.hand_image, args.increase_size_percent),
    "phone": lambda img, lm, args, det, pred: apply_object_patch(img, lm, args.phone_image, args.increase_size_percent),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Apply face patches to images using dlib landmarks.")
    parser.add_argument("--input", required=True, help="Input folder containing images or subject subfolders")
    parser.add_argument("--output", required=True, help="Output folder")
    parser.add_argument("--predictor", required=True, help="Path to shape_predictor_68_face_landmarks.dat")
    parser.add_argument("--patch", required=True, choices=list(PATCH_BUILDERS.keys()) + ["hand_mask", "phone_mask"], help="Patch type")
    parser.add_argument("--images", nargs="*", default=None, help="Image names to process (default: all images)")
    parser.add_argument("--reference-image",default="path/to/reference.jpeg", help="Reference image for face_pixel patches")
    parser.add_argument("--overlay", default="path/to/phone.png", help="PNG/object image for object or object_mask patch")
    parser.add_argument("--hand-overlay", default="path/to/hand.png", help="PNG image for hand patch")
    parser.add_argument("--phone-overlay", default="path/to/phone.png", help="PNG image for phone patch")
    parser.add_argument("--increase-size-percent", type=float, default=40.0, help="Object patch size increase percent")
    parser.add_argument("--suffix", default=None, help="Optional output filename suffix override")
    return parser.parse_args()


def iter_images(base_path):
    base = Path(base_path)
    for image_path in sorted(base.rglob("*")):
        if image_path.is_file() and image_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            try:
                rel_dir = image_path.parent.relative_to(base)
                yield str(rel_dir) if str(rel_dir) != "." else "", image_path
            except ValueError:
                continue


def output_path(base_output, patch_name, subject_name, image_name):
    out_dir = Path(base_output) / patch_name / subject_name if subject_name else Path(base_output) / patch_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / image_name


def main():
    args = parse_args()
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(args.predictor)

    args.reference_color = None
    if args.patch in {"face_pixel_rect", "face_pixel_oval"}:
        if not args.reference_image:
            raise ValueError("--reference-image is required for face_pixel patches")
        ref = cv2.imread(args.reference_image)
        if ref is None:
            raise ValueError(f"Could not read reference image: {args.reference_image}")
        args.reference_color = get_reference_face_pixel(ref, detector, predictor)
        if args.reference_color is None:
            raise ValueError("No face found in reference image")

    args.overlay_image = None
    if args.patch in {"object", "object_mask"}:
        if not args.overlay:
            raise ValueError("--overlay is required for object or object_mask patches")
        args.overlay_image = cv2.imread(args.overlay, cv2.IMREAD_UNCHANGED)
        if args.overlay_image is None:
            raise ValueError(f"Could not read overlay image: {args.overlay}")

    args.hand_image = None
    if args.patch in {"hand", "hand_mask"}:
        args.hand_image = cv2.imread(args.hand_overlay, cv2.IMREAD_UNCHANGED)
        if args.hand_image is None:
            raise ValueError(f"Could not read hand image: {args.hand_overlay}")

    args.phone_image = None
    if args.patch in {"phone", "phone_mask"}:
        args.phone_image = cv2.imread(args.phone_overlay, cv2.IMREAD_UNCHANGED)
        if args.phone_image is None:
            raise ValueError(f"Could not read phone image: {args.phone_overlay}")

    suffix_map = {
        "black_rect": "_black_rect.jpg",
        "black_oval": "_black_oval.jpg",
        "cheek_rect": "_cheek_rect.jpg",
        "cheek_oval": "_cheek_oval.jpg",
        "gray_rect": "_gray_rect.jpg",
        "gray_oval": "_gray_oval.jpg",
        "grayscale_rect": "_grayscale_rect.jpg",
        "grayscale_oval": "_grayscale_oval.jpg",
        "face_pixel_rect": "_face_pixel_rect.jpg",
        "face_pixel_oval": "_face_pixel_oval.jpg",
        "object": "_obj_patch.jpg",
        "object_mask": "_obj_mask.jpg",
        "hand": "_hand.jpg",
        "hand_mask": "_hand_mask.jpg",
        "phone": "_phone.jpg",
        "phone_mask": "_phone_mask.jpg",
    }
    suffix = args.suffix or suffix_map[args.patch]

    processed = 0
    skipped = 0

    for subject_name, image_path in iter_images(args.input):
        if args.images and image_path.name not in args.images:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            skipped += 1
            continue

        landmarks = detect_landmarks(image, detector, predictor)
        if landmarks is None:
            print(f"No landmarks detected in image: {image_path}")
            skipped += 1
            continue

        stem = image_path.stem
        out_name = f"{stem}{suffix}"
        out_path = output_path(args.output, args.patch, subject_name, out_name)

        if args.patch == "object_mask":
            result = build_object_mask(image, landmarks, args.overlay_image, args.increase_size_percent)
        elif args.patch == "hand_mask":
            result = build_object_mask(image, landmarks, args.hand_image, args.increase_size_percent)
        elif args.patch == "phone_mask":
            result = build_object_mask(image, landmarks, args.phone_image, args.increase_size_percent)
        else:
            result = PATCH_BUILDERS[args.patch](image.copy(), landmarks, args, detector, predictor)

        cv2.imwrite(str(out_path), result)
        processed += 1

    print(f"Processed: {processed}")
    print(f"Skipped: {skipped}")


if __name__ == "__main__":
    main()
