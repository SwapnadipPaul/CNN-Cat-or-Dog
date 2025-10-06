import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# -------------------- TRANSFORMS --------------------
train_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
])

val_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# -------------------- DATASET --------------------
train_data = datasets.ImageFolder('training_set', transform=train_transform)
val_data   = datasets.ImageFolder('test_set', transform=val_transform)

train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=32, shuffle=False)

# -------------------- MODEL --------------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = models.resnet18(pretrained=True)  # pretrained model
model.fc = nn.Linear(model.fc.in_features, 2)  # cats vs dogs
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# -------------------- TRAINING --------------------
num_epochs = 5
print_every = 20

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, (inputs, labels) in enumerate(train_loader, 1):
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if batch_idx % print_every == 0:
            avg_loss = running_loss / print_every
            print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}], Loss: {avg_loss:.4f}")
            running_loss = 0.0

    # -------------------- VALIDATION --------------------
    model.eval()
    val_loss = 0.0
    correct = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            val_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

    val_epoch_loss = val_loss / len(val_data)
    val_accuracy = correct / len(val_data)
    print(f"Validation Loss: {val_epoch_loss:.4f}, Accuracy: {val_accuracy:.2%}\n")

# -------------------- SAVE MODEL --------------------
torch.save(model.state_dict(), 'cat_dog_resnet18.pth')
print("Model saved as cat_dog_resnet18.pth")
