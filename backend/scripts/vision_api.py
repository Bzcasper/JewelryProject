import os
import sys
import json
from google.cloud import vision
from dotenv import load_dotenv
from rich.console import Console

# Initialize Rich console
console = Console()

# Load environment variables
load_dotenv()

def analyze_image(image_path):
    client = vision.ImageAnnotatorClient()

    with open(image_path, 'rb') as image_file:
        content = image_file.read()

    image = vision.Image(content=content)

    try:
        response = client.label_detection(image=image)
        labels = response.label_annotations

        if response.error.message:
            raise Exception(response.error.message)

        label_descriptions = [label.description for label in labels]
        return label_descriptions
    except Exception as e:
        console.log(f"[red]Google Vision API error for {image_path}: {e}[/red]")
        raise e

def analyze_images_in_directory(augmented_dir):
    labels_dict = {}
    for root, dirs, files in os.walk(augmented_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_path = os.path.join(root, file)
                try:
                    labels = analyze_image(image_path)
                    labels_dict[image_path] = labels
                    console.log(f"[blue]{image_path}[/blue] labels: {labels}")
                except Exception as e:
                    console.log(f"[red]Failed to analyze {image_path} with Google Vision API.[/red]")
    return labels_dict

if __name__ == '__main__':
    if len(sys.argv) < 2:
        console.log("[red]Usage: python vision_api.py <augmented_dir>[/red]")
        sys.exit(1)

    augmented_dir = sys.argv[1]
    analyze_images_in_directory(augmented_dir)
    console.log("[green]Vision analysis complete using Google Vision API.[/green]")
