import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets, models
from torch.utils.data import DataLoader
from tqdm import tqdm
from rich.console import Console
import deepspeed
from transformers import AdamW, get_linear_schedule_with_warmup
from peft import get_peft_model, LoraConfig, TaskType

# Initialize Rich console
console = Console()

def train_resnet50(resnet_dataset_dir, model_save_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    console.log(f"[blue]Using device: {device}[/blue]")

    # Data transformations
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    dataset = datasets.ImageFolder(resnet_dataset_dir, transform=transform)
    class_count = len(dataset.classes)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    console.log(f"[green]Number of classes: {class_count}[/green]")
    console.log(f"[green]Number of samples: {len(dataset)}[/green]")

    model = models.resnet50(pretrained=True)
    # Replace the final layer
    model.fc = nn.Linear(model.fc.in_features, class_count)

    # Apply LoRA
    lora_config = LoraConfig(
        task_type=TaskType.CLASSIFICATION,
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1
    )
    model = get_peft_model(model, lora_config)
    console.log("[green]LoRA applied to the model.[/green]")

    model = model.to(device)

    # DeepSpeed configuration
    ds_config = {
        "train_batch_size": 32,
        "gradient_accumulation_steps": 1,
        "fp16": {
            "enabled": True
        },
        "zero_optimization": {
            "stage": 2
        }
    }

    try:
        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=model.parameters(),
            config=ds_config
        )
        console.log("[green]DeepSpeed initialized successfully.[/green]")
    except Exception as e:
        console.log(f"[red]Failed to initialize DeepSpeed: {e}[/red]")
        sys.exit(1)

    criterion = nn.CrossEntropyLoss()

    epochs = 10
    model_engine.train()

    for epoch in range(epochs):
        running_loss = 0.0
        running_corrects = 0
        total = 0

        for inputs, labels in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model_engine(inputs)
            loss = criterion(outputs, labels)
            model_engine.backward(loss)
            model_engine.step()
            _, preds = torch.max(outputs, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels).item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = running_corrects / total
        console.log(f"[bold]Epoch {epoch+1}/{epochs}, Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f}[/bold]")

    # Save model
    try:
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model_engine.save_checkpoint(os.path.dirname(model_save_path), os.path.basename(model_save_path))
        console.log(f"[green]Model saved to {model_save_path}[/green]")
    except Exception as e:
        console.log(f"[red]Failed to save model: {e}[/red]")
        sys.exit(1)

if __name__ == '__main__':
    if len(sys.argv) < 3:
        console.log("[red]Usage: python train_resnet50.py <resnet_dataset_dir> <model_save_path>[/red]")
        sys.exit(1)

    resnet_dataset_dir = sys.argv[1]
    model_save_path = sys.argv[2]

    train_resnet50(resnet_dataset_dir, model_save_path)
    console.log("[green]ResNet-50 training complete.[/green]")
