# CS 410 – Text Information Systems: HW3

**University of Illinois Urbana-Champaign | Spring 2026**

This repository contains four machine programming assignments (MPs) for CS 410 HW3, covering core NLP and information retrieval techniques.

---

## MPs Overview

### MP1 – LSTM + Word2Vec: Next-Word Prediction
Trains a Word2Vec embedding on AG News article content and feeds it into a sequence-to-sequence LSTM to predict the next two word vectors. Compares `tanh` vs `relu` activations and evaluates the effect of removing stopwords on RMSE.

- Architecture: 2-layer LSTM (200 units each), RepeatVector, TimeDistributed Dense
- Embedding: Word2Vec (Skip-gram, 100-dim, window=5)
- Training: 20 epochs, batch_size=256, Adam optimizer
- Key finding: removing stopwords lowers RMSE (0.1288 vs 0.1903)

---

### MP2 – BERT Fill-Mask & GPT-2 Next-Token Prediction
Uses HuggingFace `bert-base-uncased` (fill-mask) and `gpt2` (text generation) to evaluate masked token recovery accuracy.

- BERT (no punctuation): ~0.005% exact match
- BERT (with period after mask): ~26.9% — right-side context dramatically helps
- GPT-2 causal prediction: ~22.9–24.2% top-1 accuracy
- Key finding: BERT's bidirectional attention is sensitive to punctuation context; GPT-2 predicts causally with no peek at future tokens

---

### MP3 – LDA Topic Modeling
Applies Latent Dirichlet Allocation (LDA) via Gensim to AG News articles to discover 10 latent topics.

- Preprocessing: tokenization, stopword removal, `no_below=2` frequency filter
- Training: 100 passes for convergence
- Outputs: top-5 term bar chart for Topic 0, KL-divergence heatmap across all topic pairs
- Key finding: topics clearly separate into business, sports, tech, and world news clusters

---

### MP4 – Learning to Rank + PageRank + HITS
Implements a full learning-to-rank pipeline on web graph data, combining TF-IDF retrieval scores with PageRank and HITS authority scores as features.

- Features: TF-IDF score, PageRank (d=0.85), HITS authority
- Classifier: Logistic Regression, 60/40 stratified train/test split
- Results: F1 = 1.0 on both train and test
- Key finding: sports pages have high HITS authority (~0.46) due to mutual cross-linking; finance pages score near 0.0 on HITS

---

## Files

| File | Description |
|------|-------------|
| `hw3_mp1.ipynb` | LSTM + Word2Vec notebook (executed) |
| `hw3_mp2.ipynb` | BERT + GPT-2 notebook (executed) |
| `hw3_mp3.ipynb` | LDA Topic Modeling notebook (executed) |
| `hw3_mp4.ipynb` | Learning to Rank + PageRank + HITS notebook (executed) |
| `HW3_CS_410_Spring2026_Instructions.pdf` | Official assignment instructions |

---

## Requirements

- Python 3.8+
- TensorFlow / Keras
- Gensim
- HuggingFace Transformers
- scikit-learn
- NumPy, Pandas, Matplotlib
