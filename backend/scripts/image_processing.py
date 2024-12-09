import sys
import os
from PIL import Image
from skimage import img_as_ubyte
from skimage.transform import resize
from skimage.util import random_noise
import numpy as np
import random

def crop_image(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    # Simple cropping: remove borders by trimming 10% from each side
    width, height = img.size
    left = width * 0.1
    top = height * 0.1
    right = width * 0.9
    bottom = height * 0.9
    img_cropped = img.crop((left, top, right, bottom))
    img_cropped.save(output_path, format='JPEG', quality=95)

def augment_image(image_path, output_path):
    img = Image.open(image_path).convert('RGB')
    img_np = np.array(img)

    # Resize to 224x224
    img_resized = resize(img_np, (224, 224), anti_aliasing=True)
    img_resized = img_as_ubyte(img_resized)

    # Random horizontal flip
    if random.random() < 0.5:
        img_resized = np.fliplr(img_resized)

    # Add random noise
    img_noisy = random_noise(img_resized, mode='gaussian', var=0.01)
    img_noisy = img_as_ubyte(img_noisy)

    img_aug = Image.fromarray(img_noisy)
    img_aug.save(output_path, format='JPEG', quality=95)

def process_images(raw_dir, cropped_dir, augmented_dir):
    for root, dirs, files in os.walk(raw_dir):
        rel_path = os.path.relpath(root, raw_dir)
        if rel_path == ".":
            rel_path = ""
        cropped_out_dir = os.path.join(cropped_dir, rel_path)
        augmented_out_dir = os.path.join(augmented_dir, rel_path)
        os.makedirs(cropped_out_dir, exist_ok=True)
        os.makedirs(augmented_out_dir, exist_ok=True)

        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')):
                in_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                cropped_path = os.path.join(cropped_out_dir, base_name + ".jpg")
                augmented_path = os.path.join(augmented_out_dir, base_name + "_aug.jpg")
                crop_image(in_path, cropped_path)
                augment_image(cropped_path, augmented_path)

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print("Usage: python image_processing.py <raw_dir> <cropped_dir> <augmented_dir>")
        sys.exit(1)

    raw_dir = sys.argv[1]
    cropped_dir = sys.argv[2]
    augmented_dir = sys.argv[3]

    process_images(raw_dir, cropped_dir, augmented_dir)
    print("Image processing complete.")
