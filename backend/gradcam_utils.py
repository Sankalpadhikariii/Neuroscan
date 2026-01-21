# gradcam_utils.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # Register hooks
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_full_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_tensor, target_class=None):
        # Forward pass
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Backward pass
        self.model.zero_grad()
        class_loss = output[0, target_class]
        class_loss.backward()

        # Generate CAM
        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = F.relu(cam)

        # Normalize
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.squeeze().cpu().numpy()


def generate_gradcam_from_tensor(model, input_tensor, original_image, target_class=None):
    """
    Generate GradCAM visualization

    Args:
        model: PyTorch model
        input_tensor: Preprocessed input tensor
        original_image: PIL Image (original)
        target_class: Target class index (None = predicted class)

    Returns:
        overlaid_image: numpy array (RGB)
        prediction_idx: predicted class index
    """
    try:
        # Get target layer (last conv layer)
        if hasattr(model, 'features'):
            target_layer = model.features[-1]
        elif hasattr(model, 'vgg19'):
            # For VGG-based models
            target_layer = list(model.vgg19.features.children())[-1]
        else:
            # Fallback: try to find last convolutional layer
            conv_layers = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
            if not conv_layers:
                raise ValueError("No convolutional layers found in model")
            target_layer = conv_layers[-1]

        logger.info(f"Using target layer: {target_layer.__class__.__name__}")

        # Generate CAM
        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(input_tensor, target_class)

        # Get prediction
        with torch.no_grad():
            output = model(input_tensor)
            prediction_idx = output.argmax(dim=1).item()

        # Resize CAM to match original image
        original_array = np.array(original_image)
        cam_resized = cv2.resize(cam, (original_array.shape[1], original_array.shape[0]))

        # Apply colormap
        heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

        # Overlay
        overlay = heatmap * 0.4 + original_array * 0.6
        overlaid_image = np.uint8(overlay)

        logger.info(f"✅ GradCAM generated successfully for class {prediction_idx}")

        return overlaid_image, prediction_idx

    except Exception as e:
        logger.error(f"❌ GradCAM generation failed: {e}")
        raise


# Test function (only runs when script is executed directly)
if __name__ == "__main__":
    print("🧪 Testing gradcam_utils.py...")
    print("✅ All imports successful!")
    print("✅ GradCAM class defined")
    print("✅ generate_gradcam_from_tensor function defined")
    print("\n📝 This module is ready to be imported by app.py")
    print("   Usage: from gradcam_utils import generate_gradcam_from_tensor")