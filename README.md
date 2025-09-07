# ICU-Mortality-Prediction-MIMIC-III
This project focuses on predicting 30-day ICU mortality using clinical discharge summaries from the MIMIC-III dataset. The goal is to evaluate different deep learning architectures for text classification, leveraging both custom trainable embeddings and pretrained GloVe embeddings.  
# 🏥 ICU Mortality Prediction (MIMIC-III)

This project focuses on **predicting 30-day ICU mortality** using clinical discharge summaries from the **MIMIC-III dataset**.  
The goal is to evaluate different **deep learning architectures** for text classification, leveraging both **custom trainable embeddings** and **pretrained GloVe embeddings**.  

⚠️ **Note**: Due to confidentiality, the raw medical data cannot be shared.  
Instead, this repository provides:
- The full preprocessing and modeling **Jupyter Notebooks** (for reference and learning).
- A **PDF report** with experimental results, metrics, and plots.

---

## 🚀 Project Overview

### 🔹 Problem
Predict whether a patient will **survive within 30 days** of ICU discharge based on clinical text notes.

### 🔹 Dataset
- **Source**: [MIMIC-III Clinical Database](https://physionet.org/content/mimiciii/1.4/) (confidential medical data, not shared here).  
- **Features Used**:  
  - `subject_id` (patient identifier)  
  - `hadm_id` (hospital admission ID)  
  - `text` (discharge summary)  
  - `mortality_30d` (binary target: 1 = death within 30 days, 0 = survival)

### 🔹 Approach
1. **Data Preprocessing**  
   - Text cleaning (stopwords, punctuation, digits).  
   - Synonym-based **data augmentation** (`nlpaug`).  
   - Tokenization & padding with `Keras Tokenizer`.  

2. **Embeddings**  
   - **Custom trainable embeddings**  
   - **Pretrained GloVe embeddings (100d)** (both frozen & fine-tuned)

3. **Models Implemented**
   - LSTM with custom embeddings  
   - BiLSTM with custom embeddings  
   - LSTM with frozen GloVe embeddings  
   - GRU with trainable GloVe embeddings  

4. **Training Optimization**
   - Early stopping & model checkpointing  
   - Dropout for regularization  
   - Validation split (20%)  
   - Comparison across multiple architectures

---

## 📊 Results

### Model Performance (Test Set, 30k samples → 6k test)
| Model                          | Accuracy | Precision | Recall | F1-score |
|--------------------------------|----------|-----------|--------|----------|
| LSTM (Custom Embeddings)       | **0.86** | 0.85      | 0.88   | 0.86     |
| BiLSTM (Custom Embeddings)     | **0.87** | 0.89      | 0.84   | 0.87     |
| LSTM (Frozen GloVe)            | **0.86** | 0.86      | 0.86   | 0.86     |
| GRU (Trainable GloVe)          | **0.86** | 0.87      | 0.85   | 0.86     |

🔹 The **BiLSTM with custom embeddings** achieved the **highest accuracy (87%)**.

---

## 📈 Visual Results

### Custom LSTM
- Accuracy & Loss  
  [<img width="512" height="385" alt="image" src="https://github.com/user-attachments/assets/8da9e68a-305e-4c21-9290-189c14d6a81c" />](https://github.com/mbakos95/ICU-Mortality-Prediction-MIMIC-III-/blob/main/custom_lstm_accuracy.png)  
  ![Loss](https://github.com/mbakos95/ICU-Mortality-Prediction-MIMIC-III-/blob/main/custom_lstm_loss.png)  

- Confusion Matrix  
  ![[Confusion Matrix](results/conf_matrix_custom.png)](https://github.com/mbakos95/ICU-Mortality-Prediction-MIMIC-III-/blob/main/conf_matrix_custom.png)

### BiLSTM
- Confusion Matrix  
  ![https://github.com/mbakos95/ICU-Mortality-Prediction-MIMIC-III-/blob/main/conf_matrix_bilstm.png](https://github.com/mbakos95/ICU-Mortality-Prediction-MIMIC-III-/blob/main/conf_matrix_bilstm.png)

### GloVe LSTM
- Accuracy & Loss  
  ![[Accuracy](results/glove_lstm_accuracy.png)](https://github.com/mbakos95/ICU-Mortality-Prediction-MIMIC-III-/blob/main/GLOVE%20LSTM%20ACCURACY.png)  
  ![[Loss](results/glove_lstm_loss.png)  ](https://github.com/mbakos95/ICU-Mortality-Prediction-MIMIC-III-/blob/main/GLOVE%20LSTM%20LOSS.png)

- Confusion Matrix  
  ![[Confusion Matrix](results/conf_matrix_glove.png)](https://github.com/mbakos95/ICU-Mortality-Prediction-MIMIC-III-/blob/main/conf_matrix_glove.png)

### GloVe GRU (Trainable)
- Confusion Matrix  
  ![[Confusion Matrix](results/conf_matrix_gru.png)](https://github.com/mbakos95/ICU-Mortality-Prediction-MIMIC-III-/blob/main/conf_matrix_gru.png)

### Final Comparison
- Precision / Recall / F1-score across models  
  ![[Comparison](results/comparison_metrics.png)](https://github.com/mbakos95/ICU-Mortality-Prediction-MIMIC-III-/blob/main/comparison_metrics.png)


---

## 🛠️ Tech Stack

- **Languages**: Python (Jupyter Notebook, Colab)  
- **Libraries**:  
  - `TensorFlow / Keras` (LSTM, BiLSTM, GRU models)  
  - `scikit-learn` (metrics, train-test split)  
  - `NLTK`, `nlpaug` (text cleaning & augmentation)  
  - `matplotlib`, `seaborn` (visualizations)  
  - `swifter`, `tqdm`, `pandas`, `numpy`  

---

## 📂 Repository Structure
```
icu-mortality-prediction/
│
├── ICU_Mortality_Prediction_-_Notebook_1_Dataset_Preprocessing.ipynb
├── ICU_Mortality_Prediction_-_Notebook_2.ipynb
│
├── results/                  # Plots & confusion matrices
│   ├── custom_lstm_accuracy.png
│   ├── custom_lstm_loss.png
│   ├── GLOVE_LSTM_ACCURACY.png
│   ├── GLOVE_LSTM_LOSS.png
│   ├── conf_matrix_custom.png
│   ├── conf_matrix_bilstm.png
│   ├── conf_matrix_glove.png
│   ├── conf_matrix_gru.png
│   └── comparison_metrics.png
│
├── README.md
└── requirements.txt

```






---

## 🔮 Future Work
- Explore **Transformer-based models** (BERT, ClinicalBERT).  
- Apply **explainability methods** (e.g., LIME, SHAP).  
- Extend to **multi-label predictions** (comorbidities, risk stratification).  

---

## 📌 References
- [MIMIC-III Dataset](https://physionet.org/content/mimiciii/1.4/)  
- [GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/projects/glove/)  
- Coursework: MSc Artificial Intelligence & Deep Learning (AIDL_A02)

---

