import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import VGG19_Weights

class BrainTumorVGG19(nn.Module):
    def __init__(self, num_classes=4, freeze_features=True):
        super(BrainTumorVGG19, self).__init__()
        
        # Load pre-trained VGG19
        print("Loading pre-trained VGG19...")
        self.vgg19 = models.vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        
        # Freeze feature extraction layers
        if freeze_features:
            for param in self.vgg19.features.parameters():
                param.requires_grad = False
        
        # Replace classifier
        num_features = self.vgg19.classifier[0].in_features
        
        self.vgg19.classifier = nn.Sequential(
            nn.Linear(num_features, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            
            nn.Linear(1024, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.vgg19(x)


if __name__ == "__main__":
    print("\n" + "="*60)
    print("Testing VGG19 Model Architecture")
    print("="*60)
    
    model = BrainTumorVGG19(num_classes=4)
    print("\n✓ Model created successfully!")
    
    # Test with dummy input
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"\nInput shape: {dummy_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("\n" + "="*60)
    print("✓ MODEL TEST PASSED!")
    print("="*60 + "\n")