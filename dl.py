import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import AlexNet_Weights, ResNet18_Weights, ResNet50_Weights
import joblib

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset class
class SatelliteDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        self.classes = sorted(os.listdir(root_dir))
        for label in self.classes:
            class_path = os.path.join(root_dir, label)
            if os.path.isdir(class_path):
                for img in os.listdir(class_path):
                    img_path = os.path.join(class_path, img)
                    if img_path.lower().endswith(('.jpg', '.jpeg', '.png')):
                        self.image_paths.append(img_path)
                        self.labels.append(label)
        self.le = LabelEncoder()
        self.labels = self.le.fit_transform(self.labels)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = cv2.imread(image_path)
        if image is None:
            print(f"[WARNING] Could not read image: {image_path}")
            image = np.zeros((224, 224, 3), dtype=np.uint8)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

# Transforms
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Paths
dataset_root = r'C:/Users/saisa/Downloads/NWPU/NWPU'
feature_file = "features.npy"
label_file = "labels.npy"

# Load dataset if features aren't saved yet
if not os.path.exists(feature_file) or not os.path.exists(label_file):
    print("🔄 Extracting features (first-time only)...")
    dataset = SatelliteDataset(root_dir=dataset_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)

    models_dict = {
        "AlexNet": models.alexnet(weights=AlexNet_Weights.DEFAULT),
        "LeNet": models.resnet18(weights=ResNet18_Weights.DEFAULT),  # placeholder for LeNet
        "ResNet": models.resnet50(weights=ResNet50_Weights.DEFAULT),
    }

    def extract_features(model, dataloader):
        model = torch.nn.Sequential(*list(model.children())[:-1])  # remove classifier
        model = model.to(device)
        model.eval()
        features = []
        labels = []
        with torch.no_grad():
            for images, lbls in dataloader:
                images = images.to(device)
                outputs = model(images)
                features.append(outputs.view(outputs.size(0), -1).cpu().numpy())
                labels.extend(lbls.numpy())
        return np.vstack(features), np.array(labels)

    # Extract and concatenate features
    all_features = []
    for name, model in models_dict.items():
        print(f"🔍 Using {name}...")
        feats, lbls = extract_features(model, dataloader)
        all_features.append(feats)
    combined_features = np.hstack(all_features)
    np.save(feature_file, combined_features)
    np.save(label_file, lbls)
    print("✅ Features saved to disk.")
else:
    print("✅ Features found on disk. Skipping extraction...")
    combined_features = np.load(feature_file)
    lbls = np.load(label_file)

# Feature selection (you can change this)
selector = SelectKBest(score_func=f_classif, k=min(500, combined_features.shape[1]))
features_selected = selector.fit_transform(combined_features, lbls)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    features_selected, lbls, test_size=0.2, random_state=42)

# Train SVM
classifier = SVC(kernel='linear', probability=True)
classifier.fit(X_train, y_train)

# Accuracy
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🎯 Classification Accuracy: {accuracy:.4f}")

# Save everything for later use
joblib.dump(classifier, "svm_model.pkl")
joblib.dump(selector, "selector.pkl")

# Save label encoder (from dataset if available)
if 'dataset' not in locals():
    # Recreate dataset to get label encoder
    dataset = SatelliteDataset(root_dir=dataset_root, transform=transform)

# Save label encoder
joblib.dump(dataset.le, "label_encoder.pkl")
print("✅ Model, selector, and label encoder saved.")
