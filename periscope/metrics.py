# Retrieval metrics, shared by the probe runner and the baselines so both are scored identically.

import math


def rank(scores):
    return sorted(scores, key=scores.get, reverse=True)


def ndcg(gold, ranked, k):
    dcg = sum((1.0 if d in gold else 0.0) / math.log2(i + 2) for i, d in enumerate(ranked[:k]))
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold), k)))
    return dcg / idcg if idcg else 0.0


def recall(gold, ranked, k):
    return len(gold & set(ranked[:k])) / len(gold) if gold else 0.0


def mrr(gold, ranked):
    for i, d in enumerate(ranked):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


def report(scores, golds):
    qs = [q for q in scores if golds.get(q)]
    rows = {q: rank(scores[q]) for q in qs}
    return {
        "n": len(qs),
        "NDCG@10": sum(ndcg(golds[q], rows[q], 10) for q in qs) / max(len(qs), 1),
        "MRR": sum(mrr(golds[q], rows[q]) for q in qs) / max(len(qs), 1),
        "R@1": sum(recall(golds[q], rows[q], 1) for q in qs) / max(len(qs), 1),
        "R@10": sum(recall(golds[q], rows[q], 10) for q in qs) / max(len(qs), 1),
    }
