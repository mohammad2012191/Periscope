# Reproduce the retrieval baselines under the same protocol and metrics as run_retrieval.py.
#   bm25            standard BM25, k1 = 0.9, b = 0.4
#   embed           one vector per document, truncated to --max-doc-tokens
#   embed-chunkmax  every chunk embedded, a document scored by its best chunk
#
#   python scripts/baselines.py --method bm25
#   python scripts/baselines.py --method embed --model Qwen/Qwen3-Embedding-4B

import argparse
import json
import math
import os
import re
from collections import Counter

import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from periscope import metrics  # noqa: E402

DOMAINS = ["biology", "earth_science", "economics", "pony", "psychology",
           "robotics", "sustainable_living"]

ap = argparse.ArgumentParser()
ap.add_argument("--method", default="bm25", choices=["bm25", "embed", "embed-chunkmax"])
ap.add_argument("--model", default="", help="embedding model, for the two embed methods")
ap.add_argument("--domain", default="ALL")
ap.add_argument("--queries", type=int, default=50, help="must match the run_retrieval.py value")
ap.add_argument("--max-doc-tokens", type=int, default=2048)
ap.add_argument("--chunk-tokens", type=int, default=512)
ap.add_argument("--batch", type=int, default=8)
ap.add_argument("--outdir", default="results/retrieval")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

from datasets import load_dataset

WORD = re.compile(r"[A-Za-z0-9]+")


def bm25(doc_ids, texts, queries, k1=0.9, b=0.4):
    docs = [[w.lower() for w in WORD.findall(t)] for t in texts]
    lengths = np.array([len(d) for d in docs], dtype=float)
    avg = lengths.mean() or 1.0
    tf = [Counter(d) for d in docs]
    df = Counter()
    for c in tf:
        df.update(c.keys())
    n = len(docs)
    idf = {w: math.log(1 + (n - k + 0.5) / (k + 0.5)) for w, k in df.items()}
    out = {}
    for qid, text in queries.items():
        s = np.zeros(n)
        for w in {w.lower() for w in WORD.findall(text)} & idf.keys():
            weight = idf[w]
            for i, c in enumerate(tf):
                f = c.get(w, 0)
                if f:
                    s[i] += weight * f * (k1 + 1) / (f + k1 * (1 - b + b * lengths[i] / avg))
        out[qid] = {doc_ids[i]: float(s[i]) for i in range(n)}
    return out


def embed(doc_ids, texts, queries):
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModel.from_pretrained(args.model, torch_dtype=torch.bfloat16).cuda().eval()

    @torch.no_grad()
    def encode(items, max_len):
        out = []
        for i in range(0, len(items), args.batch):
            batch = tok(items[i:i + args.batch], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt").to("cuda")
            h = model(**batch).last_hidden_state
            # last-token pooling with left padding, then L2 normalization
            out.append(torch.nn.functional.normalize(h[:, -1], p=2, dim=1).float().cpu())
        return torch.cat(out)

    qids = list(queries)
    instruction = "Instruct: Given a query, retrieve relevant documents.\nQuery: "
    Q = encode([instruction + queries[q] for q in qids], 512)

    if args.method == "embed":
        C = encode(texts, args.max_doc_tokens)
        sims = (Q @ C.T).numpy()
        return {q: {doc_ids[j]: float(sims[i, j]) for j in range(len(doc_ids))}
                for i, q in enumerate(qids)}

    pieces, owner = [], []
    for j, t in enumerate(texts):
        ids = tok(t, add_special_tokens=False)["input_ids"]
        for i in range(0, max(len(ids), 1), args.chunk_tokens):
            pieces.append(tok.decode(ids[i:i + args.chunk_tokens]))
            owner.append(j)
    C = encode(pieces, args.chunk_tokens + 16)
    sims = Q @ C.T
    owner = torch.tensor(owner).unsqueeze(0).expand(len(qids), -1)
    best = torch.full((len(qids), len(doc_ids)), -1.0).scatter_reduce(
        1, owner, sims, reduce="amax").numpy()
    return {q: {doc_ids[j]: float(best[i, j]) for j in range(len(doc_ids))}
            for i, q in enumerate(qids)}


domains = DOMAINS if args.domain == "ALL" else [args.domain]
summary = []
for domain in domains:
    corpus = {x["id"]: x["content"] for x in
              load_dataset("xlangai/BRIGHT", "long_documents")[domain]}
    examples = list(load_dataset("xlangai/BRIGHT", "examples")[domain])
    if args.queries:
        examples = examples[:args.queries]
    doc_ids = sorted(corpus)
    texts = [corpus[d] for d in doc_ids]
    queries = {e["id"]: e["query"] for e in examples}
    golds = {e["id"]: {g for g in e.get("gold_ids_long", e.get("gold_ids", [])) if g in corpus}
             for e in examples}
    print("\n%s: %d documents, %d queries, %s" % (domain, len(doc_ids), len(queries), args.method),
          flush=True)

    scores = bm25(doc_ids, texts, queries) if args.method == "bm25" else embed(doc_ids, texts, queries)
    m = metrics.report(scores, golds)
    m.update(domain=domain, arm=args.method)
    summary.append(m)
    print("  NDCG@10 %.3f  MRR %.3f  R@1 %.3f  R@10 %.3f" % (
        m["NDCG@10"], m["MRR"], m["R@1"], m["R@10"]), flush=True)

print("\nmean over %d domains" % len(domains))
print("  %-14s NDCG@10 %.3f  MRR %.3f  R@1 %.3f  R@10 %.3f" % (
    args.method, *[sum(r[k] for r in summary) / len(summary)
                   for k in ("NDCG@10", "MRR", "R@1", "R@10")]))
json.dump(summary, open(os.path.join(args.outdir, "summary_%s.json" % args.method), "w"), indent=1)
