import os
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

# === Use the correct project folder path ===
project_path = r"C:\Users\saisa\OneDrive\Desktop\satellite-image-classification\Project"

# === Load model and data ===
features = np.load(os.path.join(project_path, "features.npy"))
labels = np.load(os.path.join(project_path, "labels.npy"))
classifier = joblib.load(os.path.join(project_path, "svm_model.pkl"))
selector = joblib.load(os.path.join(project_path, "selector.pkl"))
label_encoder = joblib.load(os.path.join(project_path, "label_encoder.pkl"))

# === Apply feature selection ===
features_selected = selector.transform(features)

# === Train-Test Split (same as training step) ===
X_train, X_test, y_train, y_test = train_test_split(
    features_selected, labels, test_size=0.2, random_state=42)

# === Predict & Evaluate ===
y_pred = classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"🎯 Model Test Accuracy: {accuracy:.4f}")

# === Confusion Matrix ===
class_names = label_encoder.classes_
cm = confusion_matrix(y_test, y_pred, labels=range(len(class_names)))

# === Plot the Confusion Matrix ===
plt.figure(figsize=(16, 14))
sns.heatmap(cm, annot=False, fmt="d", cmap="YlGnBu",
            xticklabels=class_names, yticklabels=class_names)
plt.title("📊 Confusion Matrix - Satellite Image Classification")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()
