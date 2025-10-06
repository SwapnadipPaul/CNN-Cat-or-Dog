import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from main import models, train_data, device  # import classes and device
import torch.nn as nn
import os

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, 2)
model.load_state_dict(torch.load('cat_dog_resnet18.pth', map_location=device))
model = model.to(device)
model.eval()

# Transform
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# Prediction function
def predict_image(img_path):
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        pred = output.argmax(dim=1).item()

    print(f"Prediction: {train_data.classes[pred]}")
    plt.imshow(img)
    plt.title(f"Predicted: {train_data.classes[pred]}")
    plt.axis('off')
    plt.show()

# -------------------- INTERACTIVE LOOP --------------------
while True:
    img_path = input("Enter image path (or 'q' to quit): ").strip().strip('"').strip("'")
    if img_path.lower() == 'q':
        break

    if not os.path.isfile(img_path):
        print(f"File not found: {img_path}. Try again.")
        continue

    try:
        predict_image(img_path)
    except Exception as e:
        print(f"Error opening or processing image: {e}")
        continue

