# Reproduce the question-answering results: LongBench v2 and InfiniteBench En.MC.
#
# Arms
#   window     the window read: the longest prefix that fits the server's window
#   periscope  the 2K probes, then both readouts: direct (no further call) and select (one read of K chunks)
#   local      local probes only, direct readout (ablation: what the strided probes add)
#   first      one read of the first K chunks          (same budget as select, no map)
#   random     one read of K random chunks
#   embed      one read of the K chunks most similar to the question (needs --embed-server)
#   coa        Chain of Agents over the K local spans (ablation: independent probes or a chain)
#
#   python scripts/run_qa.py --server http://localhost:8000 --model Qwen/Qwen3.5-4B --window 131072 \
#       --arms window,periscope,first,random

import argparse
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from periscope import Periscope, core as C  # noqa: E402
from periscope.reader import chat, letters  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--server", required=True)
ap.add_argument("--model", required=True)
ap.add_argument("--dataset", default="longbench", choices=["longbench", "infinitebench"])
ap.add_argument("--split", default="", help="longbench: short, medium or long (default all)")
ap.add_argument("--arms", default="window,periscope,first,random")
ap.add_argument("--window", type=int, default=131072, help="must equal the server's --max-model-len")
ap.add_argument("--chunk-size", type=int, default=500)
ap.add_argument("--read-server", default=None, help="cross-model: a second model performs every read")
ap.add_argument("--read-model", default=None)
ap.add_argument("--read-window", type=int, default=None)
ap.add_argument("--embed-server", default=None, help="embed arm: `vllm serve <embedding model> --task embed`")
ap.add_argument("--embed-model", default="Qwen/Qwen3-Embedding-4B")
ap.add_argument("--coa-summary-tokens", type=int, default=512)
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--outdir", default="results/qa")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)

from datasets import load_dataset  # noqa: E402

preamble = "Read the book and answer the question.\n\n" if args.dataset == "infinitebench" else ""
P = Periscope(args.server, args.model, window=args.window, chunk_size=args.chunk_size, batch=args.batch,
              read_server=args.read_server, read_model=args.read_model, read_window=args.read_window,
              preamble=preamble)
embedder = C.Embedder(args.embed_server, args.embed_model) if args.embed_server else None


def load_examples():
    out = []
    if args.dataset == "infinitebench":
        url = ("https://huggingface.co/datasets/xinrongzhang2022/InfiniteBench/"
               "resolve/main/longbook_choice_eng.jsonl")
        for r in load_dataset("json", data_files=url, split="train"):
            options = list(r["options"])
            if len(options) != 4 or r["answer"][0] not in options:
                continue
            out.append({"id": str(r["id"]), "context": r["context"], "question": r["input"],
                        "options": options, "gold": "ABCD"[options.index(r["answer"][0])], "split": "enmc"})
    else:
        for r in load_dataset("THUDM/LongBench-v2", split="train"):
            if args.split and r["length"] != args.split:
                continue
            out.append({"id": r["_id"], "context": r["context"], "question": r["question"],
                        "options": [r["choice_A"], r["choice_B"], r["choice_C"], r["choice_D"]],
                        "gold": r["answer"].strip().upper(), "split": r["length"], "domain": r["domain"]})
    return out[:args.limit] if args.limit else out


def read_chunks(chunks, picked, ex):
    pred, scores, n = P.read(" ".join(chunks[i] for i in picked), ex["question"], ex["options"])
    return {"pred": pred, "scores": scores, "read_tokens": n, "selected": picked, "calls": 1}


# Chain of Agents, Zhang et al. (NeurIPS 2024), Table 9 prompts. The manager answers with the same
# first-token letter readout as every other arm.
def coa(ex):
    chunks = C.chunk(P.tok, ex["context"], args.chunk_size) or ["empty"]
    _, local, _ = C.grid(chunks)
    summary, calls = "", 0
    for _, span in local:
        body = (f"{span}\nHere is the summary of the previous source text: {summary}\nQuestion: {ex['question']}\n"
                "You need to read current source text and summary of previous source text (if any) and "
                "generate a summary to include them both. Later, this summary will be used for other agents "
                "to answer the Query, if any. So please write the summary that can include the evidence for "
                "answering the Query:")
        text = P.client.generate(chat(P.tok, body), args.coa_summary_tokens).strip()
        calls += 1
        summary = text or summary
    ys, _ = letters(4)
    opts = "\n".join("%s. %s" % (y, o) for y, o in zip(ys, ex["options"]))
    body = ("The following are given passages. However, the source text is too long and has been "
            f"summarized. You need to answer based on the summary:\n{summary}\n\nQuestion: {ex['question']}\n"
            f"{opts}\nAnswer with a single letter:")
    scores = C.read_scores(P.client.top_logprobs([chat(P.tok, body)])[0], ys)
    return {"pred": max(ys, key=scores.get), "scores": scores, "calls": calls + 1}


def run(arm, ex, rng):
    if arm == "window":
        r = P.window_read(ex["context"], ex["question"], ex["options"])
        return {"pred": r["pred"], **r}
    if arm in ("periscope", "local"):
        if arm == "local":
            st = P.probe(ex["context"], ex["question"], ex["options"], strided=False)
            if not st["L"]:
                return {"pred": None}
            s = C.compose(st["L"], {})
            return {"pred": max(s, key=s.get), "scores": s, "calls": st["calls"]}
        r = P.answer(ex["context"], ex["question"], ex["options"])
        return {"pred": r["select"], "direct": r["direct"], **{k: v for k, v in r.items() if k != "map"}}
    if arm == "coa":
        return coa(ex)
    chunks = C.chunk(P.tok, ex["context"], args.chunk_size) or ["empty"]
    n, k = len(chunks), math.ceil(math.sqrt(len(chunks)))
    if arm == "first":
        return read_chunks(chunks, list(range(min(k, n))), ex)
    if arm == "random":
        return read_chunks(chunks, sorted(rng.sample(range(n), min(k, n))), ex)
    if arm == "embed":
        e = embedder(["Instruct: Given a question, retrieve passages that answer it\nQuery: "
                      + ex["question"]] + chunks)
        order = [int(i) for i in np.argsort(-(e[1:] @ e[0]))]
        return read_chunks(chunks, sorted(order[:k]), ex)
    raise ValueError(arm)


def accuracy(rows, key):
    scored = [r for r in rows if r.get(key) is not None]
    return 100 * sum(r[key] == r["gold"] for r in scored) / max(len(scored), 1), len(scored)


examples = load_examples()
tag = args.model.split("/")[-1] + ("_read-" + args.read_model.split("/")[-1] if args.read_model else "")
print("%s: %d questions" % (args.dataset, len(examples)), flush=True)
for arm in args.arms.split(","):
    path = os.path.join(args.outdir, "%s_%s_%s_%s_%d.jsonl" % (
        args.dataset, args.split or "all", arm, tag, args.window))
    done = {}
    if os.path.exists(path):
        done = {r["id"]: r for r in map(json.loads, open(path, encoding="utf-8"))}
    rng = random.Random(args.seed)
    with open(path, "a", encoding="utf-8") as f:
        for ex in examples:
            if ex["id"] in done:
                continue
            t0 = time.time()
            rec = run(arm, ex, rng)
            rec.update(id=ex["id"], gold=ex["gold"], split=ex["split"], domain=ex.get("domain"),
                       n_tokens=len(P.tok.encode(ex["context"], add_special_tokens=False)),
                       latency=time.time() - t0)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done[ex["id"]] = rec
    rows = list(done.values())
    for key in (("pred", "direct") if arm == "periscope" else ("pred",)):
        name = {"pred": "select" if arm == "periscope" else arm, "direct": "direct"}[key]
        line = "%-10s all %.1f (n=%d)" % ((name,) + accuracy(rows, key))
        for split in ("short", "medium", "long"):
            sub = [r for r in rows if r["split"] == split]
            if sub:
                line += "  %s %.1f" % (split, accuracy(sub, key)[0])
        print(line, flush=True)
