import os
import glob
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from iam_segmentor.segmentor import split_form_into_sections
from iam_segmentor.utils import save_image

def process_all_forms(data_dir="data/IAM_Dataset", output_dir="outputs"):
    # Set up output folders
    comp_out = os.path.join(output_dir, "computer_written")
    hand_out = os.path.join(output_dir, "handwritten")
    bottom_out = os.path.join(output_dir, "bottom")
    os.makedirs(comp_out, exist_ok=True)
    os.makedirs(hand_out, exist_ok=True)
    os.makedirs(bottom_out, exist_ok=True)

    # Iterate through all PNG files in the dataset
    files = glob.glob(os.path.join(data_dir, "*.png"))
    print(f"[INFO] Found {len(files)} form images in {data_dir}")

    for i, file_path in enumerate(files):
        form_id = os.path.splitext(os.path.basename(file_path))[0]

        try:
            result = split_form_into_sections(file_path)
        except Exception as e:
            print(f"[ERROR] Failed to process {form_id}: {e}")
            continue

        # Save the three crops
        save_image(result["computer_written"], os.path.join(comp_out, f"{form_id}.png"))
        save_image(result["hand_written"], os.path.join(hand_out, f"{form_id}.png"))

        if result["bottom"] is not None:
            save_image(result["bottom"], os.path.join(bottom_out, f"{form_id}.png"))

        print(f"[{i+1}/{len(files)}] Processed {form_id}")

if __name__ == "__main__":
    process_all_forms()
