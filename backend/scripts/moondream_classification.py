import os
import sys
import json
from sklearn.preprocessing import LabelEncoder
import torch
import torchvision.transforms as transforms
from PIL import Image
from dotenv import load_dotenv
from rich.console import Console

# Initialize Rich console
console = Console()

# Load environment variables
load_dotenv()

def load_model():
    # Placeholder for Moondream model loading
    # Replace with actual Moondream model loading code
    from torchvision import models
    model = models.resnet50(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 6)  # 6 categories
    # Load custom trained weights if available
    model_path = os.path.join(os.getcwd(), 'models', 'moondream_resnet50.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        console.log(f"[green]Loaded Moondream model weights from {model_path}[/green]")
    else:
        console.log(f"[yellow]Moondream model weights not found at {model_path}. Using pre-trained weights.[/yellow]")
    model.eval()
    return model

def classify_image(model, image_path, transform, le):
    try:
        image = Image.open(image_path).convert('RGB')
        image = transform(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(image)
            _, preds = torch.max(outputs, 1)
            predicted_class = le.inverse_transform(preds.cpu().numpy())[0]
            return predicted_class
    except Exception as e:
        console.log(f"[red]Error classifying {image_path}: {e}[/red]")
        return "Unknown"

def classify_images(augmented_dir):
    model = load_model()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    # Define the 6 categories
    categories = ['necklace', 'ring', 'bracelet', 'pendant', 'earring', 'wristwatch']
    le = LabelEncoder()
    le.fit(categories)

    classifications = {}
    for root, dirs, files in os.walk(augmented_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_path = os.path.join(root, file)
                predicted_class = classify_image(model, image_path, transform, le)
                classifications[image_path] = predicted_class
                console.log(f"[blue]{image_path}[/blue] classified as [green]{predicted_class}[/green]")

    return classifications

if __name__ == '__main__':
    if len(sys.argv) < 2:
        console.log("[red]Usage: python moondream_classification.py <augmented_dir>[/red]")
        sys.exit(1)

    augmented_dir = sys.argv[1]
    classify_images(augmented_dir)
    console.log("[green]Moondream classification complete.[/green]")
