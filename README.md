# 🛰️ Satellite Image Classification using CNNs + SVM

This project classifies satellite images into 44+ categories (like forest, runway, stadium, etc.) using a deep learning pipeline with CNN-based feature extraction and SVM classification.

---

## 📁 Project Structure

```bash
.
├── dl.py                  # Main training code: feature extraction, feature selection, model training
├── predict_from_url.py    # Predicts the class of a satellite image from a given URL
├── Project/
│   └── (utils, models, etc. if any)
├── results/
│   ├── accuracy.txt
│   └── confusion_matrix.png
├── requirements.txt
├── .gitignore
└── README.md

Data Set Link:  https://drive.google.com/file/d/12eDQH0viFRpmr13WIMMe-S4JAuNLq18p/view?usp=sharing
