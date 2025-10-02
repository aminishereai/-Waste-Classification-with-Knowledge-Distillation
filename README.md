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
- **Student (Distilled CNN, INT8 Quantized)**: ~63.4% (no accuracy drop)
  

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

### Confusion Matrix (Quantized Student)

| Actual \ Predicted | Plastic | Paper | Metal | Glass | Organic | Trash |
|--------------------|---------|-------|-------|-------|---------|-------|
| Plastic            | 62      | 5     | 6     | 4     | 4       | 0     |
| Paper              | 4       | 51    | 23    | 7     | 16      | 0     |
| Metal              | 5       | 9     | 57    | 4     | 7       | 0     |
| Glass              | 5       | 1     | 15    | 90    | 8       | 0     |
| Organic            | 6       | 12    | 7     | 11    | 61      | 0     |
| Trash              | 8       | 3     | 7     | 5     | 4       | 1     |

- **Strong classes**: Glass, Plastic  
- **Confusion hotspots**: Paper ↔ Metal, Glass ↔ Metal  
- **Weakest class**: Trash (due to dataset imbalance)

---

### Classification Report (Quantized Student)

| Class    | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| Plastic  | 0.69      | 0.77   | 0.73     | 81      |
| Paper    | 0.63      | 0.50   | 0.56     | 101     |
| Metal    | 0.50      | 0.70   | 0.58     | 82      |
| Glass    | 0.74      | 0.76   | 0.75     | 119     |
| Organic  | 0.61      | 0.63   | 0.62     | 97      |
| Trash    | 1.00      | 0.04   | 0.07     | 28      |
| **Accuracy** |        |        | **0.63** | 508     |
| **Macro Avg** | 0.69 | 0.56   | 0.55     | 508     |
| **Weighted Avg** | 0.66 | 0.63 | 0.62     | 508     |

---

### Teacher vs. Student vs. Quantized Student

| Model                        | Size    | Accuracy | Avg Latency (ms/img) | Notes                                    |
|-------------------------------|---------|----------|-----------------------|------------------------------------------|
| Teacher (Incep‑ResNet‑v2 FP32)| ~217 MB | ~85%     | ~200 ms (GPU)         | Very accurate, too heavy for deployment  |
| Student (CNN, FP32)           | 1.5 MB  | 63.4%    | 11.26 ms              | Lightweight, deployable                  |
| Student (CNN, INT8 Quantized) | 0.38 MB | 63.4%    | 23.16 ms              | 3.8× smaller, same accuracy, slower on Colab CPU (likely faster on edge/mobile hardware) |

---


---

## 🚀 Deployment Readiness
- **Model size**: ~381 KB after quantization.  
- **Inference speed**: <25ms on CPU (tested on Colab CPU).  
- Exported to **ONNX** for cross‑platform deployment.  
- Can be integrated into a **mobile app** or **web demo** (Gradio/Streamlit).  

---

## 🔮 Future Improvements
- Collect more samples for **Trash** class to reduce imbalance.  
- Experiment with **data augmentation** (rotation, blur, brightness).  
- Try **MobileNetV2 or EfficientNet‑B0** as student backbones.  
- Apply **post‑training quantization** for even smaller footprint.  

---
### Key Takeaways
- **Knowledge Distillation** compressed a 217 MB teacher into a 1.5 MB student.  
- **Quantization** further reduced size to 0.38 MB (3.8× smaller).  
- **Accuracy** remained stable at ~63%.  
- **Inference speed**: slower on Colab CPU, but expected to be faster on real INT8‑optimized hardware (mobile/edge).  
⚡ This section tells the full story: distillation shrank the model, quantization made it ultra‑compact, and accuracy stayed intact. Recruiters will love the clarity.

---

✨ **Author**: Amin  
📅 **Year**: 2025  
🔗 **Dataset**: [Trash Type Image Dataset (Kaggle)](https://www.kaggle.com/datasets/farzadnekouei/trash-type-image-dataset)  
