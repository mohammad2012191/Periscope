# Reproduce the retrieval results: BRIGHT long documents, full corpus, no candidate pooling.
#
# Arms
#   periscope  the ranking readout, score(Yes) = max_i L_i + max_j S_j
#   local      local probes only (ablation)
#   maxp       one probe per chunk, max over chunks (ablation)
#   window     one read of the longest prefix that fits the window
#   firstp     one read of the first chunk
#
# Documents are scored document by document with every query in one batch, so the server's prefix
# cache prefills each span once.
#
#   python scripts/run_retrieval.py --server http://localhost:8000 --model Qwen/Qwen3-4B-Instruct-2507

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from periscope import core as C  # noqa: E402
from periscope import metrics  # noqa: E402
from periscope.reader import chat  # noqa: E402

DOMAINS = ["biology", "earth_science", "economics", "pony", "psychology", "robotics", "sustainable_living"]

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--domain", default="ALL")
ap.add_argument("--arms", default="periscope,window,firstp")
ap.add_argument("--queries", type=int, default=50, help="queries per domain (0 = all)")
ap.add_argument("--window", type=int, default=16384)
ap.add_argument("--chunk-size", type=int, default=100)
ap.add_argument("--batch", type=int, default=512)
ap.add_argument("--outdir", default="results/retrieval")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

from datasets import load_dataset  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(args.model)
client = C.Client(args.server, args.model, batch=args.batch)


def prompt(span, query):
    return chat(tok, "Document: %s\nQuery: %s\nIs this document relevant to the query? Answer Yes or No:"
                % (span, query))


def spans_of(text, arm):
    if arm == "window":
        ids = tok.encode(text, add_special_tokens=False)
        budget = max(args.window - 3072, args.chunk_size)       # room for the query and the template
        return [("l", tok.decode(ids[:budget], skip_special_tokens=True) if len(ids) > budget else text)]
    chunks = C.chunk(tok, text, args.chunk_size) or ["empty"]
    if arm == "firstp":
        return [("l", chunks[0])]
    if arm == "maxp":
        return [("l", c) for c in chunks]
    _, local, strided = C.grid(chunks)
    return [("l", s) for _, s in local] + ([("s", s) for _, s in strided] if arm == "periscope" else [])


domains = DOMAINS if args.domain == "ALL" else [args.domain]
summary = []
for domain in domains:
    corpus = {x["id"]: x["content"] for x in load_dataset("xlangai/BRIGHT", "long_documents")[domain]}
    examples = list(load_dataset("xlangai/BRIGHT", "examples")[domain])[:args.queries or None]
    queries = {e["id"]: e["query"] for e in examples}
    golds = {e["id"]: {g for g in e.get("gold_ids_long", e.get("gold_ids", [])) if g in corpus} for e in examples}
    overhead = {q: len(tok.encode(prompt("", t), add_special_tokens=False)) + 16 for q, t in queries.items()}
    print("\n%s: %d documents, %d queries" % (domain, len(corpus), len(queries)), flush=True)

    for arm in args.arms.split(","):
        path = os.path.join(args.outdir, "%s_%s_%d.json" % (domain, arm, args.window))
        state = json.load(open(path)) if os.path.exists(path) else {"scores": {q: {} for q in queries}, "done": []}
        done = set(state["done"])
        for n, d in enumerate(sorted(corpus)):
            if d in done:
                continue
            spans = spans_of(corpus[d], arm)
            span_len = [len(e) for e in tok([s for _, s in spans], add_special_tokens=False)["input_ids"]]
            prompts, owner = [], []
            for q, text in queries.items():
                for (kind, s), n_tok in zip(spans, span_len):
                    if n_tok + overhead[q] <= args.window:
                        prompts.append(prompt(s, text))
                        owner.append((q, kind))
            best = {}
            for (q, kind), top in zip(owner, client.top_logprobs(prompts) if prompts else []):
                v = C.log_odds(top, ["Yes"], "No")["Yes"]
                best[q, kind] = max(best.get((q, kind), -1e9), v)
            for q in queries:
                if (q, "l") in best or (q, "s") in best:
                    state["scores"][q][d] = best.get((q, "l"), -1e9) + (
                        best.get((q, "s"), -1e9) if arm == "periscope" else 0.0)
                elif arm in ("window", "firstp"):
                    state["scores"][q][d] = -1e9                    # the document does not fit one call
            done.add(d)
            if n % 25 == 0:
                state["done"] = sorted(done)
                json.dump(state, open(path, "w"))
        state["done"] = sorted(done)
        json.dump(state, open(path, "w"))
        m = metrics.report(state["scores"], golds)
        m.update(domain=domain, arm=arm)
        summary.append(m)
        print("  %-10s NDCG@10 %.3f  MRR %.3f  R@1 %.3f  R@10 %.3f" % (
            arm, m["NDCG@10"], m["MRR"], m["R@1"], m["R@10"]), flush=True)

print("\nmean over %d domains" % len(domains))
for arm in args.arms.split(","):
    rows = [m for m in summary if m["arm"] == arm]
    print("  %-10s NDCG@10 %.3f  MRR %.3f  R@1 %.3f  R@10 %.3f" % (
        arm, *[sum(r[k] for r in rows) / len(rows) for k in ("NDCG@10", "MRR", "R@1", "R@10")]), flush=True)
json.dump(summary, open(os.path.join(args.outdir, "summary_llm.json"), "w"), indent=1)
