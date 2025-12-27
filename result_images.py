import os
import torch
import torch.nn as nn
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
from model_architecture import UNetGenerator

# --- CONFIGURATION ---
# Path to your BEST checkpoint (Check your checkpoints folder)
MODEL_PATH = r"C:\lakshya\Sagar sahayak\checkpoints\best_model.pth" 
# Input: The Murky images you want to fix
INPUT_DIR = r"C:\lakshya\yol\synthetic_underwater"
# Output: Where the enhanced images will go
OUTPUT_DIR = r"C:\lakshya\yol\enhanced_underwater"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def contrast_booster(image_rgb):
    """
    Balanced enhancement without color shifts.
    """
    # 1. Apply CLAHE in LAB space to avoid color distortion
    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    # Moderate CLAHE on lightness only
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    # Merge back - this preserves color balance
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
      # 2. Very subtle saturation adjustment - avoid color shift
    hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.multiply(s, 1.05)  # type: ignore[arg-type]  # Only 5% boost to avoid red/orange tint
    s = np.clip(s, 0, 255).astype(np.uint8)
    final = cv2.cvtColor(cv2.merge((h, s, v)), cv2.COLOR_HSV2RGB)
    
    return final

def generate_enhanced_dataset():
    # 1. Setup Folders
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 2. Load the Trained U-Net
    print(f"Loading model from {MODEL_PATH}...")
    model = UNetGenerator().to(DEVICE)
    # Load weights
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval() # Freeze layers for inference
    
    # 3. Define Transform (Must match what you used in training!)
    # NO RESIZE - preserve original dimensions
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    # 4. Process Every Image (recursively through subdirectories)
    image_files = []
    for root, dirs, files in os.walk(INPUT_DIR):
        for f in files:
            if f.endswith(('.jpg', '.png', '.jpeg', '.jfif', '.ashx')):
                rel_path = os.path.relpath(root, INPUT_DIR)
                image_files.append((os.path.join(root, f), rel_path, f))
    
    print(f"Found {len(image_files)} images to enhance...")
    
    count = 0
    with torch.no_grad(): # Speed up by disabling gradients
        for img_path, rel_dir, img_name in image_files:
            try:                # Load image and preserve original size
                img = Image.open(img_path).convert("RGB")
                orig_w, orig_h = img.size
                
                # Resize to 256x256 for model processing
                img_resized = img.resize((256, 256), Image.Resampling.LANCZOS)  # type: ignore[attr-defined]
                img_tensor = transform(img_resized).unsqueeze(0).to(DEVICE)  # type: ignore[attr-defined]
                
                # Inference (The Magic)
                enhanced_tensor = model(img_tensor)
                
                # Post-process the AI output
                enhanced_img = enhanced_tensor.squeeze().cpu().permute(1, 2, 0).numpy()

                # 1. Shift from [-1, 1] to [0, 1]
                enhanced_img = (enhanced_img + 1) / 2.0

                # 2. Clip to ensure we stay in bounds
                enhanced_img = np.clip(enhanced_img, 0, 1)
                
                # 3. Convert to 0-255 (no additional processing)
                enhanced_img = (enhanced_img * 255).astype(np.uint8)
                
                # 4. Resize back to original dimensions (no color adjustment)
                enhanced_img = cv2.resize(enhanced_img, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)
                
                # Convert RGB -> BGR for OpenCV saving
                enhanced_img = cv2.cvtColor(enhanced_img, cv2.COLOR_RGB2BGR)
                
                # Preserve folder structure in output
                output_subdir = os.path.join(OUTPUT_DIR, rel_dir)
                os.makedirs(output_subdir, exist_ok=True)
                save_path = os.path.join(output_subdir, img_name)
                cv2.imwrite(save_path, enhanced_img)
                
                count += 1
                if count % 100 == 0:
                    print(f"Enhanced {count}/{len(image_files)} images...")
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

    print(f"Done! {count} images saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    generate_enhanced_dataset()