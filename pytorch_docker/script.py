import torch
from PIL import Image
from torchvision import transforms
import sys
import urllib.request
import os

def download_imagenet_classes(filename='imagenet_classes.txt'):
    url = 'https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt'
    urllib.request.urlretrieve(url, filename)

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <model_file.pt> <image_file>")
        sys.exit(1)

    model_file = sys.argv[1]
    image_file = sys.argv[2]

    # Load the model
    print("Loading model")
    model = torch.jit.load(model_file)
    model.eval()
    print("Model loaded successfully")

    # Preprocess the image
    preprocess = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),  # Converts image to PyTorch tensor and scales pixel values from [0, 255] to [0.0, 1.0]
        transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Normalization for ImageNet models
                             std=[0.229, 0.224, 0.225]),
    ])

    print("Processing image")
    img = Image.open(image_file).convert('RGB')
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0)  # Create a mini-batch as expected by the model
    print("Image processed successfully")

    # Move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    # Run inference
    with torch.no_grad():
        print("Running inference")
        output = model(input_batch)
    print("Inference completed")

    # Get probabilities
    probabilities = torch.nn.functional.softmax(output[0], dim=0)

    # Get the top 5 results
    top5_prob, top5_catid = torch.topk(probabilities, 5)

    # Download ImageNet class names if not already present
    if not os.path.exists('imagenet_classes.txt'):
        print("Downloading ImageNet class names")
        download_imagenet_classes()
        print("Downloaded ImageNet class names")

    # Load class names
    with open('imagenet_classes.txt') as f:
        categories = [s.strip() for s in f.readlines()]

    # Print the results
    print("Top 5 predictions:")
    for i in range(top5_prob.size(0)):
        print(f"   {i + 1}.) [{top5_catid[i].item()}] ({top5_prob[i].item():.4f}) {categories[top5_catid[i]]}")

if __name__ == '__main__':
    main()
