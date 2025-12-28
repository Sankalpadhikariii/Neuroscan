

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io
import base64


class GradCAM:
    """
    Grad-CAM implementation for CNN models
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: PyTorch CNN model
            target_layer: The convolutional layer to visualize (e.g., model.conv4)
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Save forward pass activations"""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients"""
        self.gradients = grad_output[0].detach()

    def generate_heatmap(self, input_tensor, class_idx=None):
        """
        Generate Grad-CAM heatmap

        Args:
            input_tensor: Input image tensor (1, C, H, W)
            class_idx: Target class index (if None, uses predicted class)

        Returns:
            heatmap: Numpy array (H, W) with values in [0, 1]
        """
        self.model.eval()

        # Forward pass
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        # Zero gradients
        self.model.zero_grad()

        # Backward pass for target class
        target = output[0, class_idx]
        target.backward()

        # Get gradients and activations
        gradients = self.gradients[0]  # (C, H, W)
        activations = self.activations[0]  # (C, H, W)

        # Global average pooling of gradients
        weights = gradients.mean(dim=(1, 2), keepdim=True)  # (C, 1, 1)

        # Weighted combination of activations
        cam = (weights * activations).sum(dim=0)  # (H, W)

        # Apply ReLU (only positive contributions)
        cam = F.relu(cam)

        # Normalize to [0, 1]
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

        return cam, class_idx


def create_heatmap_overlay(original_image, heatmap, alpha=0.4, colormap=cv2.COLORMAP_JET):
    """
    Create overlay of heatmap on original image

    Args:
        original_image: PIL Image or numpy array (H, W, 3)
        heatmap: Numpy array (H, W) with values in [0, 1]
        alpha: Transparency of heatmap overlay (0-1)
        colormap: OpenCV colormap

    Returns:
        overlay_image: PIL Image with heatmap overlay
    """
    # Convert PIL to numpy if needed
    if isinstance(original_image, Image.Image):
        original_image = np.array(original_image)

    # Ensure original image is RGB
    if len(original_image.shape) == 2:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_GRAY2RGB)
    elif original_image.shape[2] == 4:
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGBA2RGB)

    # Resize heatmap to match image size
    H, W = original_image.shape[:2]
    heatmap_resized = cv2.resize(heatmap, (W, H))

    # Convert heatmap to color using colormap
    heatmap_colored = cv2.applyColorMap(
        np.uint8(255 * heatmap_resized),
        colormap
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend original image with heatmap
    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap_colored, alpha, 0)

    return Image.fromarray(overlay)


def generate_gradcam_visualization(model, image_tensor, original_image, target_layer, class_names):
    """
    Generate complete Grad-CAM visualization with multiple views

    Args:
        model: PyTorch CNN model
        image_tensor: Preprocessed image tensor (1, C, H, W)
        original_image: Original PIL Image
        target_layer: Target conv layer for Grad-CAM
        class_names: List of class names

    Returns:
        dict with:
            - 'heatmap_base64': Base64 encoded heatmap overlay
            - 'predicted_class': Predicted class name
            - 'confidence': Confidence score
            - 'all_probs': Dictionary of all class probabilities
    """
    device = next(model.parameters()).device
    image_tensor = image_tensor.to(device)

    # Initialize Grad-CAM
    gradcam = GradCAM(model, target_layer)

    # Generate heatmap
    heatmap, class_idx = gradcam.generate_heatmap(image_tensor)

    # Get predictions
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.exp(output)[0]
        confidence = probs[class_idx].item()

    # Create overlay
    overlay_image = create_heatmap_overlay(original_image, heatmap, alpha=0.5)

    # Convert to base64
    buffer = io.BytesIO()
    overlay_image.save(buffer, format='PNG')
    overlay_base64 = base64.b64encode(buffer.getvalue()).decode()

    # Prepare probability dictionary
    all_probs = {
        class_names[i]: round(float(probs[i].item()) * 100, 2)
        for i in range(len(class_names))
    }

    return {
        'heatmap_base64': overlay_base64,
        'predicted_class': class_names[class_idx],
        'confidence': round(confidence * 100, 2),
        'all_probs': all_probs,
        'class_idx': class_idx
    }


def create_side_by_side_comparison(original_image, heatmap_overlay):
    """
    Create side-by-side comparison of original and heatmap

    Args:
        original_image: PIL Image
        heatmap_overlay: PIL Image with heatmap overlay

    Returns:
        comparison_image: PIL Image with side-by-side comparison
    """
    # Ensure same size
    width, height = original_image.size
    heatmap_overlay = heatmap_overlay.resize((width, height))

    # Create new image with double width
    comparison = Image.new('RGB', (width * 2, height))
    comparison.paste(original_image, (0, 0))
    comparison.paste(heatmap_overlay, (width, 0))

    # Convert to base64
    buffer = io.BytesIO()
    comparison.save(buffer, format='PNG')
    comparison_base64 = base64.b64encode(buffer.getvalue()).decode()

    return comparison_base64


# Example usage function for testing
if __name__ == "__main__":
    print("Grad-CAM utilities loaded successfully!")
    print("This module provides:")
    print("  - GradCAM class for generating activation maps")
    print("  - create_heatmap_overlay for visualizing attention regions")
    print("  - generate_gradcam_visualization for complete analysis")