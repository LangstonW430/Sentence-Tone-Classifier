---

# Sentence Tone / Emotion Classifier

## Overview

This project implements a **text-based emotion classifier** that predicts emotions from input sentences using a **TF-IDF + Logistic Regression** model.

The classifier can detect multiple emotions, including:

* `love`
* `hate`
* `neutral`
* `sadness`
* `enthusiasm`
* `worry`
* `happiness`
* `relief`
* `anger`

The model reads data from a CSV file containing sentences and their associated emotion labels, trains a classifier, and allows **interactive text input** for real-time predictions.

---

## Features

- **CSV dataset loader**: Handles datasets with columns `index,text,Emotion`.
- **Text preprocessing**: Lowercasing, punctuation removal, and whitespace normalization.
- **TF-IDF feature extraction**: Uses unigrams, bigrams, and trigrams to vectorize text.
- **Logistic Regression classifier**: Multi-class classifier for predicting emotions.
- **Top-K predictions**: Returns the top probable emotions with confidence scores.
- **Interactive CLI mode**: Allows users to type text and see emotion predictions visually.
- **Model persistence**: Save and load trained models using pickle.

---

## Setup

### Requirements

- Python 3.10+
- scikit-learn
- numpy

Install dependencies with pip:

```bash
pip install scikit-learn numpy
```

### Dataset Format

The CSV file should follow this format:

```
index,text,Emotion
0,"I am so happy today!","love"
1,"I feel sad and lonely","sadness"
2,"I hate this situation","hate"
...
```

- **index**: ignored by the loader.
- **text**: the input sentence.
- **Emotion**: the label for the sentence.

---

## Usage

1. **Train and Run the Model**

```bash
python main.py
```

- The script will load the dataset, train the classifier, save the model as `emotion_classifier.pkl`, and launch an interactive mode.

2. **Interactive Mode Example**

```
Enter text: I love you
love       [█████████████████████████████ ] 99.4%
empty      [                              ] 0.1%
enthusiasm [                              ] 0.1%
sadness    [                              ] 0.1%
worry      [                              ] 0.1%
```

- Top-5 predicted emotions are displayed with a bar graph representing the confidence scores.
- Type `quit`, `exit`, or `q` to leave interactive mode.

---

## Observations

- The model tends to **favor certain words**, which can lead to biased predictions.

  - Example: `i dont love you` may still return `love` due to how the training dataset represents the word “love”.

- Words like `hate` or `love` are strongly weighted, sometimes overshadowing context.
- Neutral sentences often dominate in predictions if the dataset contains more neutral examples.
- TF-IDF + Logistic Regression is simple and fast, but cannot understand complex sentence structures, sarcasm, or negations reliably.

---

## Limitations

- **Dataset bias**: Model predictions depend heavily on the training dataset.
- **Context awareness**: Model cannot handle negations or subtle sentiments effectively.
- **Rare emotions**: Classes with fewer examples may have lower prediction accuracy.
- **Bag-of-words representation**: Does not fully capture word order or meaning beyond n-grams.

---

## Future Improvements

- Replace TF-IDF + Logistic Regression with **transformer-based models** (e.g., BERT) for better context understanding.
- Augment dataset to balance rare emotions.
- Add **confusion matrix** and per-class F1-score evaluation.
- Implement multi-label classification for sentences expressing multiple emotions.

---

## File Structure

```
Sentence Tone Classifier/
│
├── main.py                         # Main script for training and interactive mode
├── emotion_sentimen_dataset.csv    # Example CSV dataset
├── emotion_classifier.pkl          # Saved trained model
└── README.md                       # Project documentation
```
