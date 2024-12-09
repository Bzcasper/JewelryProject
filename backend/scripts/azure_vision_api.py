import os
import sys
import json
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from dotenv import load_dotenv
from rich.console import Console

# Initialize Rich console
console = Console()

# Load environment variables
load_dotenv()

def analyze_image_azure(cv_client, image_path):
    try:
        with open(image_path, 'rb') as image_stream:
            description_results = cv_client.describe_image_in_stream(image_stream)
        
        if not description_results.captions:
            return []
        
        captions = [caption.text for caption in description_results.captions]
        return captions
    except Exception as e:
        console.log(f"[red]Azure Vision API error for {image_path}: {e}[/red]")
        raise e

def analyze_images_in_directory_azure(augmented_dir, cv_client):
    labels_dict = {}
    for root, dirs, files in os.walk(augmented_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_path = os.path.join(root, file)
                try:
                    captions = analyze_image_azure(cv_client, image_path)
                    labels_dict[image_path] = captions
                    console.log(f"[blue]{image_path}[/blue] captions: {captions}")
                except Exception as e:
                    console.log(f"[red]Failed to analyze {image_path} with Azure Vision API.[/red]")
    return labels_dict

if __name__ == '__main__':
    if len(sys.argv) < 2:
        console.log("[red]Usage: python azure_vision_api.py <augmented_dir>[/red]")
        sys.exit(1)

    augmented_dir = sys.argv[1]

    # Initialize Azure Vision Client
    AZURE_VISION_API_KEY = os.getenv('AZURE_VISION_API_KEY')
    AZURE_VISION_ENDPOINT = os.getenv('AZURE_VISION_ENDPOINT')

    if not AZURE_VISION_API_KEY or not AZURE_VISION_ENDPOINT:
        console.log("[red]Azure Vision API credentials are not set in .env file.[/red]")
        sys.exit(1)

    cv_client = ComputerVisionClient(AZURE_VISION_ENDPOINT, CognitiveServicesCredentials(AZURE_VISION_API_KEY))

    analyze_images_in_directory_azure(augmented_dir, cv_client)
    console.log("[green]Vision analysis complete using Azure Vision API.[/green]")
