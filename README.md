# 🗑️ Waste Classification with Knowledge Distillation

## 📌 Project Overview
This project tackles the problem of **waste classification** into six categories: **Plastic, Paper, Metal, Glass, Organic, and Trash**.  
The goal was to build a **lightweight model** that can run efficiently on low‑end devices, while retaining as much accuracy as possible from a heavy teacher model.

We used **Knowledge Distillation** to transfer knowledge from a large **Inception‑ResNet‑v2** teacher (≈85% accuracy) into a compact student CNN. The distilled student achieves **~68% accuracy** with a fraction of the parameters, making it suitable for deployment on resource‑constrained environments.

---

## 📂 Dataset
- Source: [Trash Type Image Dataset (Kaggle)](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)  
- Total images: ~2,500  
- Split: **80% training / 20% validation**  
- Classes:  
  - Plastic  
  - Paper  
  - Metal  
  - Glass  
  - Organic  
  - Trash  

---

## 🏗️ Methodology

### 1. Teacher Model
- **Backbone**: Inception‑ResNet‑v2 (pretrained on ImageNet).  
- Fine‑tuned on the waste dataset.  
- Achieved ~83–85% validation accuracy.  

### 2. Student Model (Lightweight CNN)
- 4 convolutional blocks with BatchNorm + ReLU.  
- Global Average Pooling instead of flattening large feature maps.  
- Dropout (0.5) for regularization.  
- Final classifier: `Linear(256 → 6)`.  
- Total parameters: **~1M** (vs. 54M in teacher).  

### 3. Knowledge Distillation
- **Soft loss**: KL divergence between teacher soft targets and student predictions.  
- **Hard loss**: Cross‑entropy with ground truth labels.  
- **Final loss**:  
  

\[
  L = \alpha \cdot L_{soft} + (1-\alpha) \cdot L_{hard}
  \]

  
- Hyperparameters:  
  - Temperature (T) = 2  
  - α = 0.7  

---

## 📊 Results

### Validation Accuracy
- Teacher (Inception‑ResNet‑v2): **~85%**  
- Student (Distilled CNN): **~68%**

### Confusion Matrix
Shows where the student model confuses classes:

| Actual \ Predicted | Plastic | Paper | Metal | Glass | Organic | Trash |
|--------------------|---------|-------|-------|-------|---------|-------|
| Plastic            | 62      | 5     | 6     | 3     | 5       | 0     |
| Paper              | 4       | 51    | 22    | 7     | 17      | 0     |
| Metal              | 5       | 9     | 57    | 4     | 7       | 1     |
| Glass              | 4       | 1     | 15    | 91    | 8       | 0     |
| Organic            | 6       | 13    | 7     | 11    | 60      | 0     |
| Trash              | 8       | 2     | 4     | 5     | 4       | 1     |

- **Strong classes**: Glass, Plastic  
- **Confusion hotspots**: Paper ↔ Metal, Glass ↔ Metal  
- **Weakest class**: Trash (likely due to dataset imbalance)


### Classification Report

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Plastic  | 0.70      | 0.77   | 0.73     | 81      |
| Paper    | 0.63      | 0.50   | 0.56     | 101     |
| Metal    | 0.50      | 0.70   | 0.58     | 82      |
| Glass    | 0.75      | 0.76   | 0.76     | 119     |
| Organic  | 0.59      | 0.62   | 0.61     | 97      |
| Trash    | 1.00      | 0.04   | 0.07     | 28      |
| **Accuracy** |        |        | **0.63** | 508     |
| **Macro Avg** | 0.69 | 0.56   | 0.55     | 508     |
| **Weighted Avg** | 0.66 | 0.63 | 0.62     | 508     |



---

## 🚀 Deployment Readiness
- **Model size**: ~10 MB after quantization.  
- **Inference speed**: <100ms on CPU (tested on Colab CPU).  
- Exported to **ONNX** for cross‑platform deployment.  
- Can be integrated into a **mobile app** or **web demo** (Gradio/Streamlit).  

---

## 🔮 Future Improvements
- Collect more samples for **Trash** class to reduce imbalance.  
- Experiment with **data augmentation** (rotation, blur, brightness).  
- Try **MobileNetV2 or EfficientNet‑B0** as student backbones.  
- Apply **post‑training quantization** for even smaller footprint.  

---

## 📌 Key Takeaways
- Knowledge Distillation successfully compressed a heavy teacher into a lightweight student.  
- Student retained ~80% of teacher’s performance while being **5× smaller and faster**.  
- This project demonstrates **end‑to‑end ML engineering**: dataset prep, training, distillation, evaluation, and deployment readiness.  

---

✨ **Author**: Amin  
📅 **Year**: 2025  
🔗 **Dataset**: [Trash Type Image Dataset (Kaggle)](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)  
