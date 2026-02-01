# Pneumonia Detection from Chest X-Rays Using MobileNetV2 and Grad-CAM

[Live Demo] :- https://nagane09-pneumonia-detection-using-mobilenetv2-with--app-jerbqg.streamlit.app/

This repository presents a deep learning approach for **automatic pneumonia detection** from chest X-ray images. The project combines a **MobileNetV2-based classifier** with **Grad-CAM visualizations** to provide interpretable predictions, suitable for research purposes.

---

## Dataset

The dataset contains chest X-ray images categorized into two classes:

- **Normal**: 1000+ images of healthy lungs  
- **Pneumonia**: 1000+ images of infected lungs (viral or bacterial)  

The dataset was randomly split into training, validation, and test sets to ensure unbiased model evaluation:

- **Training set**: 900+ images per class  
- **Validation set**: 300+ images per class  
- **Test set**: 250+ images per class  

This splitting ensures that the model learns generalizable features rather than memorizing specific images.

---

## Data Preprocessing

All images were resized to **224x224 pixels** to match the input size expected by the CNN. For training images, **random horizontal flips** were applied to augment the dataset and reduce overfitting. All images were normalized using ImageNet mean and standard deviation values, aligning with the pre-trained MobileNetV2 backbone.

---

## Model Architecture

The model uses **MobileNetV2** as the backbone for feature extraction. The classification head consists of:

1. A fully connected layer reducing features to 128 units  
2. ReLU activation  
3. Dropout layer with 0.3 probability  
4. Final fully connected layer producing a single output  
5. Sigmoid activation to output the probability of pneumonia  

This architecture enables efficient learning while maintaining high accuracy for binary classification.

---

## Training

- **Loss Function**: Binary Cross-Entropy (`BCELoss`)  
- **Optimizer**: Adam (learning rate 1e-4)  
- **Batch Size**: 16  
- **Epochs**: 5  

Training logs were recorded to monitor loss and ensure convergence. The trained model was saved as `mobilenetv2_pneumonia.pth`.

---

## Evaluation

The model was evaluated on the train, validation, and test sets using multiple metrics:

- **Accuracy**  
- **F1-Score**  
- **ROC-AUC**  

Additional evaluation included **confusion matrices**, **ROC curves**, and **Precision-Recall curves** to provide a comprehensive understanding of the model's performance.

**Test set performance**:

| Metric    | Value      |
|-----------|------------|
| Accuracy  | 0.9966     |
| F1-Score  | 0.9967     |
| ROC-AUC   | 0.9999     |

---

## Grad-CAM Explainability

Grad-CAM was used to generate heatmaps highlighting regions in the X-ray that contributed most to the model’s predictions. This provides **interpretability**, allowing medical practitioners to verify that the model focuses on relevant lung regions.

---

## Key Findings

- MobileNetV2 can achieve high accuracy for pneumonia detection with moderate computational resources.  
- Data augmentation and proper normalization improved generalization.  
- Grad-CAM visualizations confirm that the model focuses on meaningful areas in the lungs.  
- The pipeline demonstrates the feasibility of combining deep learning with explainable AI in medical imaging research.

---

## Requirements

```text
Python >= 3.9
PyTorch >= 2.0
Torchvision >= 0.15
PIL
numpy
scikit-learn
matplotlib
seaborn
opencv-python
pytorch-grad-cam
```
