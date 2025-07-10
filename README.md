# satellite-image-classification
CNN-based satellite image classifier using feature selection and LightGBM
# Satellite Image Classification using CNN + SVM

This project classifies satellite images into 44+ land cover classes using deep CNN feature extraction and SVM classification.

## 🔍 Architecture
- Feature Extractors: ResNet, MobileNet, AlexNet
- Feature Selection: CCGSA
- Classifier: SVM
- Dataset: NWPU VHR-10 (Landsat)

## 📂 Google Drive Link for Full Files
- Trained model files (`.pkl`)
- `features.npy`
- Sample dataset

🔗 [Download from Drive](https://drive.google.com/your-link-here)

## 🧠 Prediction Script

Run prediction on a new image:

```bash
python Project/predict.py --img_path "sample.jpg"
