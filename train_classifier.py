import pandas as pd
import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

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
# -----------------------------
threshold = df["Upvotes"].quantile(0.90)

df["viral"] = (df["Upvotes"] >= threshold).astype(int)

# -----------------------------
# Tagged documents
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
# Create vectors
# -----------------------------
X = np.array([
    model.dv[str(i)]
    for i in range(len(documents))
])

# Target labels
y = df["viral"]

# -----------------------------
# Split data
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# -----------------------------
# Logistic Regression
# -----------------------------
lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(X_train, y_train)

lr_preds = lr_model.predict(X_test)

print("\n=== Logistic Regression ===")

print("Accuracy:",
      accuracy_score(y_test, lr_preds))

print(classification_report(y_test, lr_preds))

# -----------------------------
# Random Forest
# -----------------------------
rf_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf_model.fit(X_train, y_train)

rf_preds = rf_model.predict(X_test)

print("\n=== Random Forest ===")

print("Accuracy:",
      accuracy_score(y_test, rf_preds))

print(classification_report(y_test, rf_preds))