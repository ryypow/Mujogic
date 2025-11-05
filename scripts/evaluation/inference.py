"""
Single image inference script.

Usage:
    python scripts/evaluation/inference.py --model models/best_model.pth --image test.jpg
"""

import argparse
import os
import sys
import torch
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

from src.models.example_model import ExampleCNN
from src.data.dataset import get_transforms


def predict(model, image_path, transform, device, class_names):
    """Predict class for a single image."""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Inference
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)[0]
        predicted_class = torch.argmax(probabilities).item()

    return predicted_class, probabilities


def main(args):
    """Main inference function."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Class names (you should load this from your dataset or config)
    class_names = ['class1', 'class2', 'class3']  # Replace with your classes

    # Load model
    model = ExampleCNN(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    print(f"Loaded model from {args.model}")

    # Get transform
    transform = get_transforms('test')

    # Predict
    predicted_class, probabilities = predict(model, args.image, transform, device, class_names)

    print(f"\nPrediction for: {args.image}")
    print(f"Predicted class: {class_names[predicted_class]}")
    print(f"Confidence: {probabilities[predicted_class]:.4f}")

    if args.show_top_k > 1:
        print(f"\nTop-{args.show_top_k} predictions:")
        top_k_probs, top_k_indices = torch.topk(probabilities, min(args.show_top_k, len(class_names)))
        for prob, idx in zip(top_k_probs, top_k_indices):
            print(f"  {class_names[idx]}: {prob:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run inference on a single image")
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--show-top-k', type=int, default=3,
                       help='Show top K predictions')

    args = parser.parse_args()
    main(args)
