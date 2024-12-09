import os
import sys
import pandas as pd
import shutil
import zipfile
from rich.console import Console

# Initialize Rich console
console = Console()

def prepare_resnet_dataset(augmented_dir, resnet_dataset_dir):
    if os.path.exists(resnet_dataset_dir):
        shutil.rmtree(resnet_dataset_dir)
    shutil.copytree(augmented_dir, resnet_dataset_dir)
    console.log(f"[green]ResNet-50 dataset prepared at {resnet_dataset_dir}[/green]")

def prepare_llava_dataset(augmented_dir, captions_csv, llava_dataset_zip):
    captions_df = pd.read_csv(captions_csv)
    with open('llava_dataset.txt', 'w', encoding='utf-8') as f:
        for index, row in captions_df.iterrows():
            f.write(f"Image: {row['image_path']}\nCaption: {row['caption']}\n\n")

    # Zip the dataset
    with zipfile.ZipFile(llava_dataset_zip, 'w') as zipf:
        zipf.write('llava_dataset.txt', arcname='llava_dataset.txt')
    os.remove('llava_dataset.txt')
    console.log(f"[green]LLaVA dataset zipped at {llava_dataset_zip}[/green]")

if __name__ == '__main__':
    if len(sys.argv) < 5:
        console.log("[red]Usage: python dataset_preparation.py <augmented_dir> <captions_csv> <resnet_dataset_dir> <llava_dataset_zip>[/red]")
        sys.exit(1)

    augmented_dir = sys.argv[1]
    captions_csv = sys.argv[2]
    resnet_dataset_dir = sys.argv[3]
    llava_dataset_zip = sys.argv[4]

    prepare_resnet_dataset(augmented_dir, resnet_dataset_dir)
    prepare_llava_dataset(augmented_dir, captions_csv, llava_dataset_zip)
    console.log("[green]Dataset preparation complete.[/green]")
