import os
import sys
import csv
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch
from dotenv import load_dotenv
from rich.console import Console

# Initialize Rich console
console = Console()

# Load environment variables
load_dotenv()

def generate_text(prompt, model, tokenizer, max_length=50):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(inputs, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

def generate_texts(listings_csv, generated_text_csv):
    # Load GPT-2 model and tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    model.eval()

    rows = []
    with open(listings_csv, 'r', newline='', encoding='utf-8') as infile:
        reader = csv.DictReader(infile)
        for row in reader:
            prompt = f"Generate a compelling title and description for a jewelry product. Product ID: {row['product_id']}, Type: {row['item_specifics'].split(':')[1]}, Description: {row['description']}"
            generated = generate_text(prompt, model, tokenizer)
            # Split the generated text into title and description
            parts = generated.split('\n', 1)
            title = parts[0].strip() if len(parts) > 0 else "No Title"
            description = parts[1].strip() if len(parts) > 1 else "No Description"
            rows.append({
                'product_id': row['product_id'],
                'generated_title': title,
                'generated_description': description
            })
            console.log(f"[blue]Generated text for {row['product_id']}[/blue]")

    # Write to CSV
    os.makedirs(os.path.dirname(generated_text_csv), exist_ok=True)
    with open(generated_text_csv, 'w', newline='', encoding='utf-8') as outfile:
        fieldnames = ['product_id', 'generated_title', 'generated_description']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        console.log("[red]Usage: python gpt2_text_generation.py <listings_csv> <generated_text_csv>[/red]")
        sys.exit(1)

    listings_csv = sys.argv[1]
    generated_text_csv = sys.argv[2]

    generate_texts(listings_csv, generated_text_csv)
    console.log("[green]GPT-2 text generation complete.[/green]")
