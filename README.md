# SJ-DECEN 
CYBERBULLYING DETECTION USING A CUSTOM DECEN-INSPIRED DEEP LEARNING APPROACH
==============================================================================

This repository presents a deep learning pipeline for cyberbullying detection,
inspired by the DECEN framework that was originally used for depression detection and has been adapted here to incorporate emotion-aware features for enhanced performance in cyberbullying detection. The model is trained on Wikipedia Comments which contains user-generated comments labeled for toxicity. 

DATASET
-------
**Primary Source**: [Jigsaw Toxic Comment Classification Challenge](https://www.kaggle.com/competitions/jigsaw-toxic-comment-classification-challenge/data)

The original dataset includes three files:

* `train.csv`
* `test.csv`
* `test_labels.csv`

These files were loaded and merged to isolate labeled and unlabeled samples. The original task involved **multi-label classification** with labels:

* toxic, severe toxic, obscene, threat, insult, identity hate.

For this project, the task has been **simplified to binary classification**:

* **Malicious (1)**: 10%
* **Non-Malicious (0)**: 90%
  
### Preprocessed Datasets Used:

After cleaning and processing, the following refined datasets were used for training and evaluation:

* `train_cleaned_no_punkt.csv`
* `test_labelled_cleaned_no_punkt.csv`
* `test_unlabelled_cleaned_no_punkt.csv`

These preprocessed datasets were adapted from the reference paper:
**Zinovyeva, E., Härdle, W. K., & Lessmann, S. (2020). Antisocial online behavior detection using deep learning. *Decision Support Systems, 138*, 113362.**
🔗 [DOI: 10.1016/j.dss.2020.113362](https://doi.org/10.1016/j.dss.2020.113362)

**We acknowledge and thank the authors for providing this valuable resource**, which served as the foundation for the models used in both the baseline and proposed architectures.



  
DATA PREPARATION STEPS
-----------------------
1. Load and Merge:
   - Loaded train.csv, test.csv, and test_labels.csv.
   - Merged test and test_labels to isolate labeled/unlabeled samples.

2. Text Cleaning:
   - Lowercasing
   - Expanding contractions
   - Removing stopwords and special characters
   - Optionally preserving sentence punctuation

3. Noise Reduction:
   - Removed URLs
   - Replaced repeated characters (e.g., "soooo" -> "so")

4. Export:
   - Saved cleaned datasets for training and evaluation.

BACKGROUND AND RELATED WORK
---------------------------
This project is motivated by prior research in two complementary areas:
1. **Antisocial Behavior Detection Using Deep Learning**
   Zinovyeva et al. (2020) explored CNNs, LSTMs, and GRUs for detecting antisocial behavior in online forums. Their methods provided baseline benchmarks.

   *Zinovyeva, E., Härdle, W. K., & Lessmann, S.* (2020). Antisocial online behavior detection using deep learning. *Decision Support Systems, 138*, 113362.
   [DOI](https://doi.org/10.1016/j.dss.2020.113362)

3. **Emotion-Aware Deep Learning for Depression Detection**
   DECEN integrates token-level emotion recognition with contextual embeddings to detect depressive content. Its architecture served as the foundation for our SJ-DECEN adaptation.
   
   *Yan, Z., Peng, F., & Zhang, D.* (2025). DECEN: A deep learning model enhanced by depressive emotions for depression detection from social media content. *Decision Support Systems*, 114421.
   [DOI](https://doi.org/10.1016/j.dss.2025.114421)

**We gratefully acknowledge the contributions of both sets of authors whose work directly influenced the methodology and direction of our research.**

BASELINE MODELS AND RESULTS
---------------------------
Model     | Precision(1) | Recall(1) | F1-Score(1) | Avg Precision | ROC AUC
--------- | -------------| ----------| ------------| --------------|---------
BiLSTM    |     0.75     |   0.75    |    0.75     |     0.81      |    -
LSTM      |     0.73     |   0.78    |    0.75     |     0.82      |    -
GRU       |     0.74     |   0.77    |    0.75     |       -       |    -
BGRU      |     0.76     |   0.75    |    0.76     |     0.826     |  0.960
CNN       |     0.81     |   0.66    |    0.73     |       -       |    -

CUSTOM MODEL (DECEN-INSPIRED ARCHITECTURE)
------------------------------------------
Adapted from the DECEN framework, SJ-DECEN consists of three modules:

1. DER MODULE (Detection-Emotion Recognition)
   - BiLSTM-CRF for token-level emotion labeling
   - Tasks: NER, POS tagging, Chunking

2. ECR MODULE (Emotion-Context Representation)
   - Inputs: 
     * Emotion label embeddings (from DER)
     * Post embeddings (BERT)
   - Fusion Strategies:
     * Concatenation
     * Standardization of emotion vectors (Improvement 1)
     * Residual Dot Product Fusion (Improvement 2)

3. DETECTION MODULE
   - Initial: BiLSTM-FFNN
   - Final: Hybrid CNN + BiLSTM (Single timestep classification)

CUSTOM MODEL RESULTS
--------------------
ECR Strategy                  | Detection Module      | Precision(1) | Recall(1) | F1-score(1)
-----------------------------|------------------------|--------------|-----------|-------------
Emotion + Post Concatenation | BiLSTM                 |     0.75     |   0.58    |     0.65
Standardized Emotion Vectors| BiLSTM                |     0.78     |   0.63    |     0.70
Residual Dot Product Fusion | BiLSTM                |     0.81     |   0.57    |     0.67
Final Fusion + Hybrid Model   | CNN + BiLSTM Classifier|    0.83     |   0.55    |     0.66

TECHNOLOGIES USED
-----------------
- Python
- TensorFlow / Keras
- BERT (via HuggingFace Transformers)
- CRF Layer (for sequence tagging)
- Scikit-learn
- NLTK, regex
- Pandas, NumPy

KEY TAKEAWAYS
-------------
- Emotion labels enhance cyberbullying detection.
- Fusion strategies like residual dot product fusion improve performance.
- CNN + BiLSTM hybrid architecture showed better results than pure sequence models.
