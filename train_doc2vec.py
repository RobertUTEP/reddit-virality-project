import pandas as pd
import numpy as np

from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# -----------------------------
# Load dataset
# -----------------------------
df = pd.read_csv("reddit_posts.csv")

# Keep only needed columns
df = df[["Title", "Body", "Upvotes", "num_comments", "Subreddit"]]

# Remove missing rows
df = df.dropna()

# -----------------------------
# Combine title + body
# -----------------------------
df["text"] = df["Title"] + " " + df["Body"]

# -----------------------------
# Create viral label
# Top 10% upvotes = viral
# -----------------------------
threshold = df["Upvotes"].quantile(0.90)

df["viral"] = (df["Upvotes"] >= threshold).astype(int)

print("Viral Threshold:", threshold)
print(df["viral"].value_counts())

# -----------------------------
# Prepare Doc2Vec documents
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

print("Doc2Vec training complete.")

# -----------------------------
# Create vectors
# -----------------------------
vectors = np.array([
    model.dv[str(i)]
    for i in range(len(documents))
])

print("Vector shape:", vectors.shape)

# -----------------------------
# t-SNE visualization
# -----------------------------
tsne = TSNE(
    n_components=2,
    random_state=42,
    perplexity=30
)

tsne_result = tsne.fit_transform(vectors)

# -----------------------------
# Plot
# -----------------------------
plt.figure(figsize=(10, 8))

scatter = plt.scatter(
    tsne_result[:, 0],
    tsne_result[:, 1],
    c=df["viral"],
    alpha=0.6
)

plt.title("t-SNE Visualization of Reddit Post Embeddings")

plt.xlabel("Dimension 1")
plt.ylabel("Dimension 2")

plt.colorbar(label="Viral")

plt.show()