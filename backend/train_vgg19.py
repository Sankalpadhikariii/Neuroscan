import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from datetime import datetime
from model_vgg19 import BrainTumorVGG19

# Configuration
CONFIG = {
    'data_dir': 'brain',  # Your dataset folder
    'batch_size': 8,  # Optimized for GTX 1650 (4GB VRAM)
    'num_epochs': 60,  # ← INCREASED FROM 20 TO 60
    'learning_rate': 0.001,
    'num_classes': 4,
    'image_size': 224,
    'num_workers': 2,  # For Windows
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'save_dir': 'models',
    'checkpoint_interval': 10,  # ← NEW: Save checkpoint every 10 epochs
    'early_stopping_patience': 15,  # ← NEW: Stop if no improvement for 15 epochs
}

CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']


def get_transforms():
    """Data augmentation and preprocessing"""
    train_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((CONFIG['image_size'], CONFIG['image_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def load_data():
    """Load dataset"""
    print("\n" + "=" * 60)
    print("Loading Dataset...")
    print("=" * 60)

    train_transform, val_transform = get_transforms()

    # Load training data
    train_dataset = datasets.ImageFolder(
        root=os.path.join(CONFIG['data_dir'], 'train'),
        transform=train_transform
    )

    # Load validation data
    val_dataset = datasets.ImageFolder(
        root=os.path.join(CONFIG['data_dir'], 'val'),
        transform=val_transform
    )

    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(val_dataset)}")
    print(f"Classes: {train_dataset.classes}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True if CONFIG['device'] == 'cuda' else False
    )

    return train_loader, val_loader


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc='Training', leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{running_loss / (pbar.n + 1):.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def validate(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation', leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            pbar.set_postfix({
                'loss': f'{running_loss / (pbar.n + 1):.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100. * correct / total

    return epoch_loss, epoch_acc


def plot_history(history, save_path):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history['train_loss'], label='Train Loss', marker='o')
    ax1.plot(history['val_loss'], label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(history['train_acc'], label='Train Acc', marker='o')
    ax2.plot(history['val_acc'], label='Val Acc', marker='s')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    print(f"\n✓ Training curves saved to {save_path}")


def save_checkpoint(model, optimizer, epoch, val_acc, history, checkpoint_path):
    """Save training checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'history': history,
        'class_names': CLASS_NAMES,
        'config': CONFIG
    }, checkpoint_path)
    print(f"💾 Checkpoint saved: {checkpoint_path}")


def train_model():
    """Main training function"""
    print("\n" + "=" * 60)
    print("VGG19 BRAIN TUMOR DETECTION - TRAINING")
    print("=" * 60)
    print(f"Device: {CONFIG['device'].upper()}")
    print(f"Batch size: {CONFIG['batch_size']}")
    print(f"Epochs: {CONFIG['num_epochs']}")
    print(f"Learning rate: {CONFIG['learning_rate']}")
    print(f"Early stopping patience: {CONFIG['early_stopping_patience']}")
    print(f"Checkpoint interval: Every {CONFIG['checkpoint_interval']} epochs")
    print("=" * 60)

    # Create save directory
    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Load data
    train_loader, val_loader = load_data()

    # Initialize model
    print("\nInitializing VGG19 model...")
    model = BrainTumorVGG19(num_classes=4, freeze_features=True)
    model = model.to(CONFIG['device'])
    print(f"✓ Model loaded on {CONFIG['device']}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=CONFIG['learning_rate']
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5,
    )

    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0
    epochs_without_improvement = 0

    print("\n" + "=" * 60)
    print("STARTING TRAINING...")
    print("=" * 60 + "\n")

    for epoch in range(1, CONFIG['num_epochs'] + 1):
        print(f"Epoch {epoch}/{CONFIG['num_epochs']}")
        print("-" * 60)

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, CONFIG['device']
        )

        # Validate
        val_loss, val_acc = validate(
            model, val_loader, criterion, CONFIG['device']
        )

        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # Scheduler step
        scheduler.step(val_loss)

        # Print summary
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Learning Rate: {current_lr:.6f}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            best_path = os.path.join(CONFIG['save_dir'], 'vgg19_brain_tumor_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': CLASS_NAMES,
                'config': CONFIG
            }, best_path)
            print(f"⭐ BEST MODEL SAVED (Val Acc: {val_acc:.2f}%)")
        else:
            epochs_without_improvement += 1

        # Save periodic checkpoint
        if epoch % CONFIG['checkpoint_interval'] == 0:
            checkpoint_path = os.path.join(
                CONFIG['save_dir'],
                f'checkpoint_epoch_{epoch}_{timestamp}.pth'
            )
            save_checkpoint(model, optimizer, epoch, val_acc, history, checkpoint_path)

        # Early stopping check
        if epochs_without_improvement >= CONFIG['early_stopping_patience']:
            print(f"\n⚠️  Early stopping triggered! No improvement for {CONFIG['early_stopping_patience']} epochs.")
            print(f"Best validation accuracy: {best_val_acc:.2f}%")
            break

        print()

    # Save final model
    final_path = os.path.join(CONFIG['save_dir'], f'vgg19_final_{timestamp}.pth')
    torch.save({
        'model_state_dict': model.state_dict(),
        'history': history,
        'class_names': CLASS_NAMES,
        'config': CONFIG
    }, final_path)

    # Plot curves
    plot_path = os.path.join(CONFIG['save_dir'], f'training_curves_{timestamp}.png')
    plot_history(history, plot_path)

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best model: {best_path}")
    print(f"Final model: {final_path}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    train_model()