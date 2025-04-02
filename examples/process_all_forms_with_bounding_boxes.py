import os
import glob
import sys
from collections import defaultdict
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from iam_segmentor.segmentor import split_form_with_handwritten_bounding_box
from iam_segmentor.utils import save_image


def load_word_annotations(path="data/annotations/words.txt"):
    """Load words.txt and group lines by form ID (e.g., a01-000u)"""
    with open(path, "r") as f:
        lines = f.readlines()

    form_map = defaultdict(list)
    for line in lines:
        if line.startswith("#") or line.strip() == "":
            continue
        word_id = line.split()[0]
        form_id = "-".join(word_id.split("-")[:2])
        form_map[form_id].append(line)

    return form_map


def process_all_forms(data_dir="data/IAM_Dataset", output_dir="outputs", annotation_path="data/IAM_Dataset/annotations/words.txt"):
    word_groups = load_word_annotations(annotation_path)

    comp_out = os.path.join(output_dir, "computer_written")
    hand_out = os.path.join(output_dir, "handwritten")
    os.makedirs(comp_out, exist_ok=True)
    os.makedirs(hand_out, exist_ok=True)

    files = glob.glob(os.path.join(data_dir, "*.png"))
    print(f"[INFO] Found {len(files)} form images in {data_dir}")

    for i, file_path in enumerate(files):
        form_id = os.path.splitext(os.path.basename(file_path))[0]

        if form_id not in word_groups:
            print(f"[SKIP] No word annotations for {form_id}")
            continue

        try:
            result = split_form_with_handwritten_bounding_box(file_path, word_groups[form_id])
        except Exception as e:
            print(f"[ERROR] Failed to process {form_id}: {e}")
            continue

        if result["computer_written"] is not None and result["computer_written"].size > 0:
            save_image(result["computer_written"], os.path.join(comp_out, f"{form_id}.png"))

        if result["hand_written"] is not None and result["hand_written"].size > 0:
            save_image(result["hand_written"], os.path.join(hand_out, f"{form_id}.png"))


        print(f"[{i+1}/{len(files)}] Processed {form_id}")

if __name__ == "__main__":
    process_all_forms()
