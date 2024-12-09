import os
import sys
import openai
import csv
from dotenv import load_dotenv
from rich.console import Console

# Initialize Rich console
console = Console()

# Load environment variables
load_dotenv()

# OpenAI API Key
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

def generate_caption(image_path):
    try:
        # Since GPT-2 does not process images, we use image metadata or labels to generate captions
        # This is a placeholder; integrate with actual image description if available
        prompt = f"Provide a detailed caption for the jewelry product image located at {image_path}."
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=prompt,
            max_tokens=50
        )
        caption = response.choices[0].text.strip()
        return caption
    except Exception as e:
        console.log(f"[red]Error generating caption for {image_path}: {e}[/red]")
        return "No caption available."

def generate_captions(augmented_dir, captions_csv):
    fieldnames = ['image_path', 'caption']
    rows = []

    for root, dirs, files in os.walk(augmented_dir):
        for file in files:
            if file.lower().endswith('.jpg'):
                image_path = os.path.join(root, file)
                caption = generate_caption(image_path)
                rows.append({'image_path': image_path, 'caption': caption})
                console.log(f"[blue]{image_path}[/blue] caption generated.")

    # Write to CSV
    try:
        os.makedirs(os.path.dirname(captions_csv), exist_ok=True)
        with open(captions_csv, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        console.log(f"[green]Captions CSV saved at {captions_csv}[/green]")
    except Exception as e:
        console.log(f"[red]Failed to write captions CSV. Error: {e}[/red]")
        raise e

if __name__ == '__main__':
    if len(sys.argv) < 3:
        console.log("[red]Usage: python captions_generator.py <augmented_dir> <captions_csv>[/red]")
        sys.exit(1)

    augmented_dir = sys.argv[1]
    captions_csv = sys.argv[2]

    generate_captions(augmented_dir, captions_csv)
    console.log("[green]Captions generation complete.[/green]")
