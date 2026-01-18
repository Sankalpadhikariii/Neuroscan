"""
NeuroScan Model Evaluation Script - VGG19 Architecture
Correctly handles your actual VGG19-based model
"""

import torch
import torch.nn as nn
from torchvision import transforms, datasets, models
from torch.utils.data import DataLoader
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
import json
import warnings

warnings.filterwarnings('ignore')


# ============================================
# Model Architecture (VGG19-based)
# ============================================
class BrainTumorVGG19(nn.Module):
    def __init__(self, num_classes=4):
        super(BrainTumorVGG19, self).__init__()

        # Load pretrained VGG19
        self.vgg19 = models.vgg19(pretrained=False)

        # YOUR EXACT classifier architecture from the checkpoint
        num_features = self.vgg19.classifier[0].in_features  # 25088
        self.vgg19.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),  # Layer 0
            nn.ReLU(True),  # Layer 1
            nn.Dropout(0.5),  # Layer 2
            nn.Linear(4096, 1024),  # Layer 3 - YOUR MODEL
            nn.ReLU(True),  # Layer 4
            nn.Dropout(0.5),  # Layer 5
            nn.Linear(1024, 256),  # Layer 6 - YOUR MODEL
            nn.ReLU(True),  # Layer 7
            nn.Dropout(0.5),  # Layer 8
            nn.Linear(256, num_classes)  # Layer 9 - YOUR MODEL (256 -> 4)
        )

    def forward(self, x):
        return self.vgg19(x)


# ============================================
# Configuration
# ============================================
# UPDATE THESE PATHS!
MODEL_PATH = r'C:\Users\Acer\OneDrive\Desktop\final year project\backend\models\vgg19_final_20260110_154609.pth'
TEST_DATA_PATH = r"C:\Users\Acer\OneDrive\Desktop\final year project\backend\brain_retrain\val"
OUTPUT_DIR = 'evaluation_results'

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 4
CLASS_NAMES = ['glioma', 'meningioma', 'notumor', 'pituitary']
CLASS_DISPLAY = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {DEVICE}")


# ============================================
# Manual Confusion Matrix (no sklearn)
# ============================================
def compute_confusion_matrix(y_true, y_pred, num_classes):
    """Compute confusion matrix without sklearn"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def compute_metrics(y_true, y_pred, num_classes):
    """Compute metrics without sklearn"""
    cm = compute_confusion_matrix(y_true, y_pred, num_classes)

    # Per-class metrics
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1 = np.zeros(num_classes)
    support = np.zeros(num_classes, dtype=int)

    for i in range(num_classes):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp

        support[i] = cm[i, :].sum()

        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0

        if (precision[i] + recall[i]) > 0:
            f1[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
        else:
            f1[i] = 0

    # Overall accuracy
    accuracy = np.trace(cm) / cm.sum() * 100

    # Weighted F1
    f1_weighted = np.average(f1, weights=support)

    # Macro F1
    f1_macro = np.mean(f1)

    return {
        'accuracy': accuracy,
        'f1_weighted': f1_weighted,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'support': support,
        'confusion_matrix': cm
    }


# ============================================
# Load Model
# ============================================
def load_model(model_path, device):
    """Load VGG19-based model"""
    print(f"\nLoading model from: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    # Initialize model
    model = BrainTumorVGG19(num_classes=NUM_CLASSES)

    # Load weights
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            print("✓ Found wrapped model_state_dict")
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()
    print("✓ Model loaded successfully\n")

    return model


# ============================================
# Data Loading
# ============================================
def get_data_loader(data_path):
    """Load test dataset"""
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    try:
        test_dataset = datasets.ImageFolder(data_path, transform=transform)
        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0  # Windows compatibility
        )
        return test_loader, test_dataset
    except Exception as e:
        print(f"❌ Error loading test data: {e}")
        print(f"   Make sure {data_path} exists with subfolders:")
        print(f"   - glioma/")
        print(f"   - meningioma/")
        print(f"   - notumor/")
        print(f"   - pituitary/")
        raise


# ============================================
# Evaluation
# ============================================
def evaluate_model(model, test_loader):
    """Run evaluation"""
    model.eval()
    all_predictions = []
    all_labels = []

    print("Evaluating model on test set...")
    total_batches = len(test_loader)

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{total_batches} batches...")

    return np.array(all_predictions), np.array(all_labels)


# ============================================
# Visualization
# ============================================
def plot_confusion_matrix(cm, save_path):
    """Plot confusion matrix"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=CLASS_DISPLAY,
           yticklabels=CLASS_DISPLAY,
           title='Confusion Matrix - Brain Tumor Classification',
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14, fontweight='bold')

    fig.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved: {save_path}")


def plot_f1_scores(f1_scores, save_path):
    """Plot F1-scores"""
    colors = ['#FF9999', '#66B2FF', '#99FF99', '#FFCC99']

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(CLASS_DISPLAY, f1_scores, color=colors,
                  edgecolor='black', linewidth=1.2)

    # Add value labels
    for bar, score in zip(bars, f1_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height + 0.01,
                f'{score:.2f}',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_title('Final F1-Scores by Disease Category',
                 fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('F1-Score', fontsize=12, fontweight='bold')
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ F1-scores chart saved: {save_path}")


# ============================================
# Reports
# ============================================
def generate_text_report(metrics, save_path):
    """Generate text report"""
    with open(save_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("NEUROSCAN MODEL EVALUATION REPORT\n")
        f.write("=" * 70 + "\n\n")

        f.write(f"Overall Accuracy: {metrics['accuracy']:.2f}%\n")
        f.write(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}\n")
        f.write(f"Macro F1-Score: {metrics['f1_macro']:.4f}\n\n")

        f.write("-" * 70 + "\n")
        f.write("PER-CLASS METRICS\n")
        f.write("-" * 70 + "\n\n")

        for i, class_name in enumerate(CLASS_NAMES):
            display_name = CLASS_DISPLAY[i]

            f.write(f"{display_name}:\n")
            f.write(f"  Precision: {metrics['precision'][i]:.4f}\n")
            f.write(f"  Recall:    {metrics['recall'][i]:.4f}\n")
            f.write(f"  F1-Score:  {metrics['f1'][i]:.4f}\n")
            f.write(f"  Support:   {metrics['support'][i]}\n\n")

        f.write("=" * 70 + "\n")
        f.write("\nCONFUSION MATRIX\n")
        f.write("-" * 70 + "\n")

        # Header
        f.write("True\\Pred  ")
        for name in CLASS_DISPLAY:
            f.write(f"{name:12s} ")
        f.write("\n" + "-" * 70 + "\n")

        # Rows
        cm = metrics['confusion_matrix']
        for i, true_name in enumerate(CLASS_DISPLAY):
            f.write(f"{true_name:10s} ")
            for j in range(len(CLASS_DISPLAY)):
                f.write(f"{cm[i, j]:12d} ")
            f.write("\n")

        f.write("=" * 70 + "\n")

    print(f"✓ Text report saved: {save_path}")


# ============================================
# Main
# ============================================
def main():
    output_dir = Path(OUTPUT_DIR)
    output_dir.mkdir(exist_ok=True)

    print("\n" + "=" * 70)
    print("NEUROSCAN MODEL EVALUATION (VGG19)")
    print("=" * 70)

    try:
        # Load model
        model = load_model(MODEL_PATH, DEVICE)

        # Load data
        print(f"Loading test data from: {TEST_DATA_PATH}")
        test_loader, test_dataset = get_data_loader(TEST_DATA_PATH)
        print(f"✓ Loaded {len(test_dataset)} test images\n")

        # Evaluate
        y_pred, y_true = evaluate_model(model, test_loader)
        print(f"✓ Evaluation complete\n")

        # Calculate metrics
        print("Calculating metrics...")
        metrics = compute_metrics(y_true, y_pred, NUM_CLASSES)
        print("✓ Metrics calculated\n")

        # Generate visualizations
        print("Generating visualizations...")
        plot_confusion_matrix(metrics['confusion_matrix'],
                              output_dir / 'confusion_matrix.png')
        plot_f1_scores(metrics['f1'],
                       output_dir / 'f1_scores.png')

        # Generate reports
        print("\nGenerating reports...")
        generate_text_report(metrics, output_dir / 'evaluation_report.txt')

        # Save JSON
        json_metrics = {
            'accuracy': float(metrics['accuracy']),
            'f1_weighted': float(metrics['f1_weighted']),
            'f1_macro': float(metrics['f1_macro']),
            'per_class': {
                CLASS_NAMES[i]: {
                    'precision': float(metrics['precision'][i]),
                    'recall': float(metrics['recall'][i]),
                    'f1_score': float(metrics['f1'][i]),
                    'support': int(metrics['support'][i])
                } for i in range(NUM_CLASSES)
            }
        }

        with open(output_dir / 'metrics.json', 'w') as f:
            json.dump(json_metrics, f, indent=2)
        print(f"✓ Metrics JSON saved: {output_dir / 'metrics.json'}")

        # Print summary
        print("\n" + "=" * 70)
        print("EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Overall Accuracy: {metrics['accuracy']:.2f}%")
        print(f"Weighted F1-Score: {metrics['f1_weighted']:.4f}")
        print(f"Macro F1-Score: {metrics['f1_macro']:.4f}")
        print("\nPer-Class F1-Scores:")
        for i, class_name in enumerate(CLASS_NAMES):
            display_name = CLASS_DISPLAY[i]
            f1 = metrics['f1'][i]
            print(f"  {display_name:15s}: {f1:.4f}")
        print("=" * 70 + "\n")

        print(f"✓ All results saved to: {output_dir.absolute()}\n")

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()