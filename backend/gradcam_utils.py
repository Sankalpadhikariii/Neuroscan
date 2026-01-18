"""
Grad-CAM Utilities for Brain Tumor Classification
Generates class activation heatmaps to visualize what the model is focusing on
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import io


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping for CNNs
    """

    def __init__(self, model, target_layer):
        """
        Args:
            model: trained PyTorch model
            target_layer: the convolutional layer to generate heatmap from
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        """Hook to save forward pass activations"""
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        """Hook to save backward pass gradients"""
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        """
        Generate Class Activation Map

        Args:
            input_tensor: preprocessed input image tensor
            target_class: class index to generate CAM for (if None, uses predicted class)

        Returns:
            heatmap: numpy array of the heatmap
            prediction: predicted class index
        """
        # Forward pass
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        # Get gradients and activations
        gradients = self.gradients[0].cpu().numpy()  # (C, H, W)
        activations = self.activations[0].cpu().numpy()  # (C, H, W)

        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))  # (C,)

        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        # Apply ReLU
        cam = np.maximum(cam, 0)

        # Normalize
        if cam.max() > 0:
            cam = cam / cam.max()

        return cam, target_class

    def overlay_heatmap_on_image(self, image, heatmap, alpha=0.5, colormap=cv2.COLORMAP_JET):
        """
        Overlay heatmap on original image

        Args:
            image: original image (PIL Image or numpy array)
            heatmap: CAM heatmap
            alpha: transparency of overlay
            colormap: OpenCV colormap

        Returns:
            overlaid_image: numpy array of the overlaid image
        """
        # Convert PIL to numpy if needed
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Ensure image is RGB
        if len(image.shape) == 2:  # Grayscale
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:  # RGBA
            image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)

        # Resize heatmap to match image size
        h, w = image.shape[:2]
        heatmap_resized = cv2.resize(heatmap, (w, h))

        # Convert heatmap to RGB using colormap
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized),
            colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Ensure image is uint8
        if image.dtype != np.uint8:
            image = np.uint8(255 * (image - image.min()) / (image.max() - image.min()))

        # Overlay
        overlaid = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

        return overlaid


def generate_gradcam_for_model(model, image_path, target_layer_name='features',
                               target_class=None, save_path=None):
    """
    Convenience function to generate and save Grad-CAM

    Args:
        model: trained PyTorch model
        image_path: path to input image or PIL Image
        target_layer_name: name of the target layer
        target_class: class to generate CAM for (None = use prediction)
        save_path: where to save the result (None = don't save)

    Returns:
        overlaid_image: numpy array of the Grad-CAM visualization
        prediction: predicted class
    """
    import torchvision.transforms as transforms

    # Get target layer
    if hasattr(model, target_layer_name):
        target_layer = getattr(model, target_layer_name)
        # For VGG-style models, get the last conv layer
        if hasattr(target_layer, '__getitem__'):
            # Find last conv layer
            for i in range(len(target_layer) - 1, -1, -1):
                if isinstance(target_layer[i], torch.nn.Conv2d):
                    target_layer = target_layer[i]
                    break
    else:
        raise ValueError(f"Layer {target_layer_name} not found in model")

    # Load and preprocess image
    if isinstance(image_path, str):
        image = Image.open(image_path).convert('RGB')
    else:
        image = image_path

    original_image = image.copy()

    # Preprocessing
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = preprocess(image).unsqueeze(0)

    # Move to same device as model
    device = next(model.parameters()).device
    input_tensor = input_tensor.to(device)

    # Generate CAM
    grad_cam = GradCAM(model, target_layer)
    heatmap, prediction = grad_cam.generate_cam(input_tensor, target_class)

    # Overlay on original image
    overlaid = grad_cam.overlay_heatmap_on_image(original_image, heatmap)

    # Save if requested
    if save_path:
        Image.fromarray(overlaid).save(save_path)

    return overlaid, prediction


def generate_gradcam_from_tensor(model, input_tensor, original_image,
                                 target_layer_name='features', target_class=None):
    """
    Generate Grad-CAM from already preprocessed tensor

    Args:
        model: trained model
        input_tensor: preprocessed input tensor
        original_image: original PIL Image for overlay
        target_layer_name: layer name
        target_class: target class (None = use prediction)

    Returns:
        overlaid_image: numpy array
        prediction: predicted class index
    """
    # Get target layer
    if hasattr(model, target_layer_name):
        target_layer = getattr(model, target_layer_name)
        if hasattr(target_layer, '__getitem__'):
            for i in range(len(target_layer) - 1, -1, -1):
                if isinstance(target_layer[i], torch.nn.Conv2d):
                    target_layer = target_layer[i]
                    break
    else:
        raise ValueError(f"Layer {target_layer_name} not found")

    # Generate CAM
    grad_cam = GradCAM(model, target_layer)
    heatmap, prediction = grad_cam.generate_cam(input_tensor, target_class)

    # Overlay
    overlaid = grad_cam.overlay_heatmap_on_image(original_image, heatmap)

    return overlaid, prediction


if __name__ == "__main__":
    # Example usage
    print("Grad-CAM utilities loaded successfully")
    print("Use generate_gradcam_for_model() or generate_gradcam_from_tensor()")