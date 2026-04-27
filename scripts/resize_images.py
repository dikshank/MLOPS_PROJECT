import os
import argparse
from PIL import Image

# argument parser (so you can pass folders via cmd)
parser = argparse.ArgumentParser(description="Resize images to 64x64")
parser.add_argument("--input", required=True, help="Input folder path")
parser.add_argument("--output", required=True, help="Output folder path")
args = parser.parse_args()

input_folder = args.input
output_folder = args.output

os.makedirs(output_folder, exist_ok=True)

valid_ext = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

for filename in os.listdir(input_folder):
    if filename.lower().endswith(valid_ext):
        input_path = os.path.join(input_folder, filename)
        output_path = os.path.join(output_folder, filename)

        try:
            with Image.open(input_path) as img:
                img = img.convert("RGB")
                img_resized = img.resize((64, 64), Image.Resampling.LANCZOS)
                img_resized.save(output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

print("Done resizing images ✅")