#  Disaster Tweet Classification: Real vs Metaphorical Disaster Detection
A deep learning NLP project that distinguishes between real disaster tweets and metaphorical disaster language using fine-tuned DistilBERT. The model achieves **83.94% accuracy** in identifying genuine emergencies from social media noise.

---
##  Table of Contents

- Overview
- Problem Statement
- Dataset
- Methodology
- Results
- Installation
- Usage
- Project Structure
- Model Performance
- Future Improvements
- Contributing


---
##  Overview

This project tackles the challenge of **automatic disaster detection** from Twitter data. Not all tweets mentioning disasters refer to actual emergencies—people frequently use disaster terminology metaphorically (e.g., "My presentation was a disaster" or "My code is on fire!"). 

Our solution uses **state-of-the-art NLP** to distinguish literal from figurative language, enabling emergency response systems to filter genuine disasters from the noise.

### Key Features

-  **Two-tiered modeling approach**: Baseline (TF-IDF + Logistic Regression) and Advanced (DistilBERT)
-  **High accuracy**: 83.94% on validation set
-  **Fast training**: ~4 minutes on GPU
-  **Production-ready**: Complete pipeline from data preprocessing to deployment
-  **Comprehensive evaluation**: Multiple metrics, error analysis, and visualizations

---

##  Problem Statement

### The Challenge

Social media platforms like Twitter generate massive volumes of disaster-related content. However, distinguishing between **real emergencies** and **metaphorical usage** is difficult:

**Real Disasters** (Target = 1):
```
"Massive earthquake hits California, buildings collapsed"
"Forest fire spreading rapidly in the region"
"Flood warning issued for coastal areas"
```

**Metaphorical Usage** (Target = 0):
```
"My presentation was an absolute disaster"
"My code is on fire today! Absolutely ABLAZE with productivity"
"This traffic is killing me"
```

### Solution

Build a binary classification model using transformer-based deep learning to automatically categorize tweets with high accuracy.

---

##  Dataset

### Source
Kaggle Disaster Tweets Dataset

### Statistics

| Metric | Value |
|--------|-------|
| **Total Tweets** | 7,613 |
| **After Cleaning** | 7,503 (removed 110 duplicates) |
| **Training Set** | 6,002 samples (80%) |
| **Validation Set** | 1,501 samples (20%) |
| **Class Distribution** | Non-Disaster: 57.38% / Disaster: 42.62% |
| **Average Tweet Length** | 15-20 words |

### Data Characteristics

- **Features**: Tweet text only
- **Labels**: Binary (0 = Non-Disaster, 1 = Disaster)
- **Challenges**: Sarcasm, metaphors, figurative language, slang, abbreviations
- **Noise**: URLs, mentions (@), hashtags (#)


---

##  Methodology

### Pipeline Architecture

```
Data Loading → EDA → Preprocessing → Baseline Model → DistilBERT → Evaluation → Deployment
```

### 1. Data Preprocessing

- Removed 110 duplicate tweets
- Handled missing values
- Stratified train/validation split (80/20)
- Minimal text cleaning (preserving context for transformers)

### 2. Baseline Model

**TF-IDF + Logistic Regression**
- TF-IDF vectorization (max 5,000 features, unigrams + bigrams)
- Balanced class weights
- Quick benchmark: **78.48% accuracy**

### 3. Advanced Model

**Fine-tuned DistilBERT**
- Pre-trained model: `distilbert-base-uncased`
- Tokenization: Max length 128 tokens
- Training: 3 epochs, batch size 16, learning rate 2e-5
- Optimization: AdamW with weight decay 0.01
- Evaluation: Per epoch

### 4. Training Configuration

| Hyperparameter | Value |
|----------------|-------|
| Model | DistilBERT Base Uncased |
| Epochs | 3 |
| Batch Size | 16 |
| Learning Rate | 2e-5 |
| Max Sequence Length | 128 |
| Optimizer | AdamW |
| Weight Decay | 0.01 |
| Training Time | 242.78 seconds (~4 minutes) |
| Samples/Second | 74.17 |

---
##  Results

### Model Comparison

| Model | Accuracy | F1-Score | Precision | Recall |
|-------|----------|----------|-----------|--------|
| **Logistic Regression (Baseline)** | 78.48% | 0.7455 | 0.7520 | 0.7391 |
| **DistilBERT (Fine-tuned)** | **83.94%** | **0.8052** | **0.8342** | **0.7781** |
| **Improvement** | **+5.46%** | **+5.97%** | **+8.22%** | **+3.90%** |

### Detailed Classification Report

#### DistilBERT Performance

```
               precision    recall  f1-score   support

Non-Disaster       0.84      0.89      0.86       861
    Disaster       0.83      0.78      0.81       640

    accuracy                           0.84      1501
   macro avg       0.84      0.83      0.83      1501
weighted avg       0.84      0.84      0.84      1501
```

### Key Metrics

- **ROC-AUC Score**: 0.8936 (Excellent discrimination ability)
- **Training Loss**: 0.3417
- **Validation Loss**: 0.4509
- **Total Misclassifications**: 241 out of 1,501
  - False Positives: 99 (predicted disaster, actually non-disaster)
  - False Negatives: 142 (predicted non-disaster, actually disaster)

### Confidence Analysis

- **Correct Predictions**: Average confidence 93.07%
- **Incorrect Predictions**: Average confidence 84.95%
- **Insight**: Model shows appropriate uncertainty on difficult cases

---

##  Error Analysis

### Common Misclassification Patterns

**False Positives** (Predicted Disaster, Actually Non-Disaster):
- Strong emotional/violent words used metaphorically
- Examples: "The game was a bloodbath", "Sales are exploding"

**False Negatives** (Predicted Non-Disaster, Actually Disaster):
- Downplayed severity or resolved situations
- Examples: "Small fire contained quickly", "Minor flooding expected"

### Model Strengths

- Correctly identifies "ABLAZE" in productivity context  
-  Understands multi-word phrases and context  
- Robust to noise (URLs, hashtags, mentions)  
- High confidence on clear-cut cases  

---

##  Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn

### Setup

# Install dependencies
pip install -r requirements.txt

# Or install individually
pip install torch transformers datasets accelerate
pip install scikit-learn pandas numpy matplotlib seaborn
```

##  Project Structure

```
disaster-tweet-classification/
│
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── disaster_tweet_classifier.ipynb    # Main Jupyter notebook
│
├── data/
│   └── train.csv                      # Training dataset
│
├── models/
│   └── disaster_tweet_classifier/     # Saved model and tokenizer
│
├── results/
│   ├── predictions.csv                # Validation predictions
│   ├── confusion_matrix.png           # Confusion matrix visualization
│   └── roc_curve.png                  # ROC curve plot
│
├── notebooks/
│   └── exploratory_analysis.ipynb     # Additional EDA
│
└── src/
    ├── preprocessing.py               # Data preprocessing utilities
    ├── model.py                       # Model architecture
    └── utils.py                       # Helper functions      
    ```


---

##  Model Performance

### Confusion Matrix

```
                Predicted
              Non-Dis  Disaster
Actual  Non-Dis   762      99
        Disaster  142     498
```

### Performance Highlights

-  **83.94%** overall accuracy
-  **86%** F1-score for Non-Disaster class
-  **81%** F1-score for Disaster class
-  **89.36%** ROC-AUC score
-  Balanced performance across both classes

### Speed

- **Training**: 242 seconds (~4 minutes on GPU)
- **Inference**: ~30ms per tweet (batch of 32)
- **Throughput**: 74 samples/second during training

---

##  Future Improvements

### Short-term

- [ ] Implement ensemble methods (combine baseline + DistilBERT)
- [ ] Add LIME/SHAP for model interpretability
- [ ] Create REST API for real-time predictions
- [ ] Deploy on cloud platform (AWS/GCP/Azure)

### Long-term

- [ ] Expand to multi-language support (Spanish, French, etc.)
- [ ] Integrate geolocation data for better context
- [ ] Implement active learning pipeline
- [ ] Add confidence calibration
- [ ] Create mobile app for emergency responders
- [ ] Real-time Twitter stream integration

### Model Enhancements

- [ ] Experiment with larger models (BERT, RoBERTa, GPT)
- [ ] Multi-task learning (disaster type classification)
- [ ] Few-shot learning for rare disaster types
- [ ] Incorporate user metadata and temporal features

---

##  Acknowledgments

- **Dataset**: Kaggle Disaster Tweets Dataset
- **Model**: Hugging Face Transformers (DistilBERT)
- **Inspiration**: Real-world emergency response challenges
- **Community**: Thanks to all contributors and the NLP community


## 🌟 Star History

If you find this project useful, please consider giving it a ⭐!

---

**Last Updated**: January 2025  
