# You can pass the input and output directories as arguments
# Example: bash run_all_patches.sh /path/to/your/input_images /path/to/your/output_images

INPUT_DIR="${1:-./input_dir}"
OUTPUT_DIR="${2:-./output_dir}"
PREDICTOR="/DATA2/Atul/2027/IJCV/Code/shape_predictor_68_face_landmarks.dat"

# Check if input directory exists
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory '$INPUT_DIR' does not exist."
    echo "Please provide a valid input directory."
    echo "Usage: bash run_all_patches.sh <input_dir> <output_dir>"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# List of all patches to apply
patches=(
    "black_rect" 
    "black_oval" 
    "cheek_rect" 
    "cheek_oval" 
    "gray_rect" 
    "gray_oval" 
    "grayscale_rect" 
    "grayscale_oval" 
    "face_pixel_rect" 
    "face_pixel_oval" 
    "hand" 
    "phone"
    "hand_mask"
    "phone_mask"
)

echo "Starting batch patch generation..."
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "========================================"

for patch in "${patches[@]}"; do
    echo "--> Applying patch: $patch"
    python face_patches.py \
        --input "$INPUT_DIR" \
        --output "$OUTPUT_DIR" \
        --predictor "$PREDICTOR" \
        --patch "$patch"
done

echo "========================================"
echo "Finished! All patches have been generated in $OUTPUT_DIR"
