"""
Emotion Classification from CSV dataset
Detects emotions: joy, sadness, anger, fear, love
"""

import re
import pickle
import numpy as np
import csv
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


class EmotionClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 3),
            min_df=2,
            stop_words="english"
        )

        self.classifier = LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=42
        )

        self.is_trained = False

    @staticmethod
    def preprocess(text: str) -> str:
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def load_csv_dataset(self, filepath: str, verbose=True):
        """
        Load dataset from CSV file.

        Expected CSV format:
        index, text, Emotion
        """
        sentences, labels = [], []

        with open(filepath, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header

            for row in reader:
                if len(row) < 3:
                    continue

                text = row[1].strip()
                label = row[2].strip()

                if text and label:
                    sentences.append(text)
                    labels.append(label)

        if not sentences:
            raise ValueError("No valid samples found in dataset.")

        if verbose:
            print(f"Loaded {len(sentences)} samples")
            emotion_counts = Counter(labels)
            for k, v in emotion_counts.most_common(10):
                print(f"  {k}: {v}")
            print(f" ... {len(emotion_counts)} total labels")

        return sentences, labels


    def train(self, sentences, labels, test_size=0.2, verbose=True):
        processed = [self.preprocess(s) for s in sentences]

        X_train, X_test, y_train, y_test = train_test_split(
            processed,
            labels,
            test_size=test_size,
            stratify=labels,
            random_state=42
        )

        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)

        self.classifier.fit(X_train_vec, y_train)

        train_pred = self.classifier.predict(X_train_vec)
        test_pred = self.classifier.predict(X_test_vec)

        if verbose:
            print("\n Training complete")
            print(f"  Train accuracy: {accuracy_score(y_train, train_pred):.2%}")
            print(f"  Test accuracy : {accuracy_score(y_test, test_pred):.2%}")

        self.is_trained = True

    def predict(self, sentence, top_k=3):
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        vec = self.vectorizer.transform([self.preprocess(sentence)])
        probs = self.classifier.predict_proba(vec)[0]
        classes = self.classifier.classes_

        top_idx = np.argsort(probs)[::-1][:top_k]
        return [(classes[i], float(probs[i])) for i in top_idx]

    def predict_single(self, sentence):
        if not self.is_trained:
            raise RuntimeError("Model not trained")

        vec = self.vectorizer.transform([self.preprocess(sentence)])
        return self.classifier.predict(vec)[0]

    def save_model(self, path="emotion_model.pkl"):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        print(f"Model saved -> {path}")

    @staticmethod
    def load_model(path="emotion_model.pkl"):
        with open(path, "rb") as f:
            return pickle.load(f)


def interactive_mode(model):
    print("\n" + "=" * 50)
    print("EMOTION CLASSIFIER - Interactive Mode")
    print("=" * 50)
    print("Type 'quit' to exit\n")

    while True:
        text = input("Enter text: ").strip()
        if text.lower() in {"quit", "exit", "q"}:
            print("Goodbye!")
            break

        if not text:
            continue

        preds = model.predict(text, top_k=5)
        print()
        for label, score in preds:
            bar = "█" * int(score * 30)
            print(f"{label:10s} [{bar:<30}] {score:.1%}")
        print()


def main():
    classifier = EmotionClassifier()

    # 🔹 Change this path to your CSV file
    dataset_path = "emotion_sentimen_dataset.csv"

    sentences, labels = classifier.load_csv_dataset(dataset_path)
    classifier.train(sentences, labels)

    classifier.save_model("emotion_classifier.pkl")
    interactive_mode(classifier)


if __name__ == "__main__":
    main()
