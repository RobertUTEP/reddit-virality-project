import pandas as pd
import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)

import matplotlib.pyplot as plt

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("reddit_posts.csv")

df = df[["Title", "Body", "Upvotes", "num_comments", "Subreddit"]]

df = df.dropna()

# -----------------------------
# Combine text
# -----------------------------
df["text"] = df["Title"] + " " + df["Body"]

# -----------------------------
# Viral label
# Top 10% of upvotes = viral
# -----------------------------
threshold = df["Upvotes"].quantile(0.90)

df["viral"] = (df["Upvotes"] >= threshold).astype(int)

print("Viral threshold:", threshold)

print("\nClass counts:")
print(df["viral"].value_counts())

# -----------------------------
# Tagged documents for Doc2Vec
# -----------------------------
documents = [
    TaggedDocument(
        words=text.lower().split(),
        tags=[str(i)]
    )
    for i, text in enumerate(df["text"])
]

# -----------------------------
# Train Doc2Vec
# -----------------------------
model = Doc2Vec(
    vector_size=100,
    window=5,
    min_count=2,
    workers=4,
    epochs=20
)

model.build_vocab(documents)

model.train(
    documents,
    total_examples=model.corpus_count,
    epochs=model.epochs
)

# -----------------------------
# Create Doc2Vec vectors
# -----------------------------
X = np.array([
    model.dv[str(i)]
    for i in range(len(documents))
])

y = df["viral"]

# -----------------------------
# Split data
# stratify keeps viral/non-viral ratio balanced
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Logistic Regression
# class_weight balances viral/non-viral classes
# -----------------------------
lr_model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced"
)

lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)

lr_accuracy = accuracy_score(y_test, lr_preds)

print("\n=== Logistic Regression ===")
print("Accuracy:", lr_accuracy)

print(classification_report(
    y_test,
    lr_preds,
    zero_division=0
))

# -----------------------------
# Random Forest
# class_weight balances viral/non-viral classes
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

rf_accuracy = accuracy_score(y_test, rf_preds)

print("\n=== Random Forest ===")
print("Accuracy:", rf_accuracy)

print(classification_report(
    y_test,
    rf_preds,
    zero_division=0
))

# -----------------------------
# Confusion Matrix - Logistic Regression
# -----------------------------
lr_cm = confusion_matrix(y_test, lr_preds)

lr_display = ConfusionMatrixDisplay(
    confusion_matrix=lr_cm,
    display_labels=["Non-Viral", "Viral"]
)

lr_display.plot()

plt.title("Logistic Regression Confusion Matrix")

plt.savefig(
    "logistic_regression_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# Confusion Matrix - Random Forest
# -----------------------------
rf_cm = confusion_matrix(y_test, rf_preds)

rf_display = ConfusionMatrixDisplay(
    confusion_matrix=rf_cm,
    display_labels=["Non-Viral", "Viral"]
)

rf_display.plot()

plt.title("Random Forest Confusion Matrix")

plt.savefig(
    "random_forest_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

# -----------------------------
# Accuracy Comparison Bar Chart
# -----------------------------
model_names = [
    "Logistic Regression",
    "Random Forest"
]

accuracies = [
    lr_accuracy,
    rf_accuracy
]

plt.figure(figsize=(8, 5))

plt.bar(model_names, accuracies)

plt.ylim(0, 1)

plt.ylabel("Accuracy")

plt.title("Model Accuracy Comparison")

for i, acc in enumerate(accuracies):
    plt.text(
        i,
        acc + 0.02,
        f"{acc:.2f}",
        ha="center"
    )

plt.savefig(
    "accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()