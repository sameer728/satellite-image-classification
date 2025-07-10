import joblib
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.models import AlexNet_Weights, ResNet50_Weights, ResNet18_Weights
from PIL import Image
import os
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Load models and tools
classifier = joblib.load("svm_model.pkl")
selector = joblib.load("selector.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Load saved features & labels (from training step)
features = np.load("features.npy")
labels = np.load("labels.npy")

# Apply same feature selection
features_selected = selector.transform(features)

# Re-split test set (same split ratio as training)
_, X_test, _, y_test = train_test_split(features_selected, labels, test_size=0.2, random_state=42)

# Evaluate classifier
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load same CNNs as training
alexnet = models.alexnet(weights=AlexNet_Weights.DEFAULT)
alexnet = torch.nn.Sequential(*list(alexnet.children())[:-1]).to(device).eval()

resnet18 = models.resnet18(weights=ResNet18_Weights.DEFAULT)
resnet18 = torch.nn.Sequential(*list(resnet18.children())[:-1]).to(device).eval()

resnet50 = models.resnet50(weights=ResNet50_Weights.DEFAULT)
resnet50 = torch.nn.Sequential(*list(resnet50.children())[:-1]).to(device).eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def load_image_from_path(img_path):
    try:
        image = Image.open(img_path).convert('RGB')
        return image
    except Exception as e:
        print(f"❌ Failed to load image: {e}")
        return None

def extract_features(image):
    tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feat1 = alexnet(tensor).view(1, -1).cpu().numpy()
        feat2 = resnet18(tensor).view(1, -1).cpu().numpy()
        feat3 = resnet50(tensor).view(1, -1).cpu().numpy()
    combined = np.hstack([feat1, feat2, feat3])
    selected = selector.transform(combined)
    return selected

def classify_image_path(path):
    path = path.strip()
    if not os.path.exists(path):
        return "❌ File path doesn't exist."

    image = load_image_from_path(path)
    if image is None:
        return "❌ Could not process the image."

    features = extract_features(image)
    prediction = classifier.predict(features)
    label = label_encoder.inverse_transform(prediction)[0]
    return label

# === MAIN ===
if __name__ == "__main__":
    path = input("📁 Enter the full image file path: ").strip()
    result = classify_image_path(path)
    print(f"\n📌 Predicted Class: {result}")
    print(f"🎯 Model Test Accuracy: {accuracy:.4f}")
