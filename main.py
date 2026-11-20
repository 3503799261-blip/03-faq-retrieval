"""Chinese FAQ retrieval: NumPy BM25 versus character TF-IDF."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
SEED = 42


def tokenize(text):
    """Keep Latin words; split each Chinese span into overlapping bigrams."""
    tokens = []
    for span in re.findall(r"[a-z0-9]+|[\u4e00-\u9fff]+", text.lower()):
        if re.fullmatch(r"[a-z0-9]+", span) or len(span) == 1:
            tokens.append(span)
        else:
            tokens.extend(span[i:i + 2] for i in range(len(span) - 1))
    return tokens


class BM25:
    def __init__(self, documents, k1=1.5, b=0.75):
        self.counts = [Counter(tokenize(doc)) for doc in documents]
        self.lengths = np.array([sum(c.values()) for c in self.counts], dtype=float)
        self.avg_length = max(float(self.lengths.mean()), 1.0)
        self.k1, self.b = k1, b
        document_frequency = Counter(token for count in self.counts for token in count)
        n = len(documents)
        self.idf = {term: float(np.log1p((n - df + 0.5) / (df + 0.5))) for term, df in document_frequency.items()}

    def score(self, query):
        scores = np.zeros(len(self.counts), dtype=float)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.avg_length)
        for token in set(tokenize(query)):
            tf = np.array([count.get(token, 0) for count in self.counts], dtype=float)
            scores += self.idf.get(token, 0.0) * tf * (self.k1 + 1) / (tf + norm)
        return scores


def evaluate(scorer, queries, documents):
    reciprocal, recall, top1, details = [], [], [], []
    ids = [doc["id"] for doc in documents]
    for row in queries:
        scores = scorer(row["query"])
        ranking = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0]
        relevant = set(row["relevant_ids"])
        positions = [rank + 1 for rank, i in enumerate(ranking) if ids[i] in relevant]
        reciprocal.append(1 / min(positions) if positions else 0.0)
        recall.append(len(relevant & {ids[i] for i in ranking[:3]}) / len(relevant))
        top1.append(float(bool(ranking) and ids[ranking[0]] in relevant))
        details.append({"query": row["query"], "relevant_ids": row["relevant_ids"], "top_ids": [ids[i] for i in ranking[:3]]})
    return {"mrr": float(np.mean(reciprocal)), "recall_at_3": float(np.mean(recall)), "top1_accuracy": float(np.mean(top1))}, details


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", default="忘记登录密码要怎么找回？")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    args = parser.parse_args()
    if not args.query.strip() or args.top_k < 1:
        parser.error("Provide a nonempty --query and positive --top-k")
    documents = json.loads((ROOT / "data" / "faqs.json").read_text(encoding="utf-8"))
    queries = json.loads((ROOT / "data" / "queries.json").read_text(encoding="utf-8"))
    if len({d["id"] for d in documents}) != len(documents):
        raise ValueError("FAQ ids must be unique")
    if any(not set(q["relevant_ids"]) <= {d["id"] for d in documents} for q in queries):
        raise ValueError("Query references unknown FAQ id")
    texts = [d["question"] + " " + d["answer"] for d in documents]
    bm25 = BM25(texts)
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
    matrix = vectorizer.fit_transform(texts)

    def tfidf_score(query):
        return (matrix @ vectorizer.transform([query]).T).toarray().ravel()

    bm_metrics, bm_details = evaluate(bm25.score, queries, documents)
    tf_metrics, tf_details = evaluate(tfidf_score, queries, documents)
    report = {"project": ROOT.name, "seed": SEED,
              "data_source": "Repository-authored fictional service FAQ and manually labeled paraphrase queries; not a public benchmark.",
              "faq_count": len(documents), "query_count": len(queries),
              "metrics": {"bm25": bm_metrics, "tfidf_baseline": tf_metrics}}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.output_dir / "evaluation.json").write_text(json.dumps({"bm25": bm_details, "tfidf": tf_details}, ensure_ascii=False, indent=2), encoding="utf-8")
    scores = bm25.score(args.query)
    ranking = [int(i) for i in np.argsort(-scores, kind="stable") if scores[i] > 0][:args.top_k]
    results = [{**documents[i], "score": float(scores[i])} for i in ranking]
    demo = {"query": args.query, "results": results,
            "message": "以下是词面匹配的候选，分数不是正确概率。" if results else "没有匹配词，请换个问法或补充 FAQ。"}
    (args.output_dir / "demo.json").write_text(json.dumps(demo, ensure_ascii=False, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 4))
    keys = ["mrr", "recall_at_3", "top1_accuracy"]
    x = np.arange(len(keys))
    ax.bar(x - 0.18, [bm_metrics[k] for k in keys], 0.36, label="BM25")
    ax.bar(x + 0.18, [tf_metrics[k] for k in keys], 0.36, label="TF-IDF baseline")
    ax.set(xticks=x, xticklabels=["MRR", "Recall@3", "Top-1"], ylim=(0, 1.08), title="FAQ retrieval: authored query set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir / "retrieval_metrics.png", dpi=150)
    plt.close(fig)
    print(json.dumps({"metrics": report["metrics"], "demo": demo}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
