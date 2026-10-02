# Command line for your own data.
#
#   python -m periscope qa   --server URL --model NAME --input questions.jsonl --output answers.jsonl
#   python -m periscope rank --server URL --model NAME --queries queries.jsonl --docs docs.jsonl --output ranking.jsonl
#
# questions.jsonl  {"id": ..., "text": ..., "question": ..., "options": [...]}   one per line
# queries.jsonl    {"id": ..., "query": ...}
# docs.jsonl       {"id": ..., "text": ...}

import argparse
import json
import os

from .reader import Periscope, letters


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def done_ids(path):
    return {r["id"] for r in read_jsonl(path)} if os.path.exists(path) else set()


def main():
    ap = argparse.ArgumentParser(prog="python -m periscope")
    sub = ap.add_subparsers(dest="task", required=True)
    for name in ("qa", "rank"):
        s = sub.add_parser(name)
        s.add_argument("--server", required=True, help="OpenAI-compatible endpoint, e.g. http://localhost:8000")
        s.add_argument("--model", required=True)
        s.add_argument("--window", type=int, default=131072, help="the server's --max-model-len")
        s.add_argument("--chunk-size", type=int, default=500 if name == "qa" else 100)
        s.add_argument("--batch", type=int, default=64)
        s.add_argument("--output", required=True)
    qa = sub.choices["qa"]
    qa.add_argument("--input", required=True)
    qa.add_argument("--readout", default="both", choices=["both", "direct"],
                    help="direct skips the selected read (no call after the probes)")
    qa.add_argument("--budget", type=int, default=0, help="chunks in the selected read (0 = K)")
    qa.add_argument("--read-server", default=None, help="a second model performs the selected read")
    qa.add_argument("--read-model", default=None)
    qa.add_argument("--preamble", default="", help="text placed before every prompt")
    qa.add_argument("--save-map", action="store_true", help="keep the evidence map of every question")
    rk = sub.choices["rank"]
    rk.add_argument("--queries", required=True)
    rk.add_argument("--docs", required=True)
    rk.add_argument("--top", type=int, default=100, help="documents kept per query in the output")
    a = ap.parse_args()

    p = Periscope(a.server, a.model, window=a.window, chunk_size=a.chunk_size, batch=a.batch,
                  read_server=getattr(a, "read_server", None), read_model=getattr(a, "read_model", None),
                  preamble=(a.preamble + "\n\n") if getattr(a, "preamble", "") else "")
    done = done_ids(a.output)

    with open(a.output, "a", encoding="utf-8") as out:
        if a.task == "qa":
            for ex in read_jsonl(a.input):
                if ex["id"] in done:
                    continue
                r = p.answer(ex["text"], ex["question"], ex["options"], budget=a.budget or None,
                             read=a.readout == "both")
                ys, _ = letters(len(ex["options"]))
                rec = {"id": ex["id"], "direct": r["direct"], "select": r["select"],
                       "selected_chunks": r.get("selected"), "peak_chunk": (
                           r["peak"][0] * r["k"] + r["peak"][1]) if "peak" in r else None,
                       "scores": r.get("scores"), "k": r["k"], "n_chunks": r["n_chunks"], "calls": r["calls"]}
                if a.save_map:
                    rec["map"] = r.get("map")
                if "answer" in ex:
                    gold = ex["answer"] if ex["answer"] in ys else ys[ex["options"].index(ex["answer"])]
                    rec["gold"] = gold
                out.write(json.dumps(rec) + "\n")
                out.flush()
                print(ex["id"], "direct", rec["direct"], "select", rec["select"], flush=True)
        else:
            docs = {d["id"]: d["text"] for d in read_jsonl(a.docs)}
            for q in read_jsonl(a.queries):
                if q["id"] in done:
                    continue
                scores = p.rank(q["query"], docs)
                ranked = sorted(scores, key=scores.get, reverse=True)[:a.top]
                out.write(json.dumps({"id": q["id"], "ranking": [[d, scores[d]] for d in ranked]}) + "\n")
                out.flush()
                print(q["id"], "top", ranked[:3], flush=True)


if __name__ == "__main__":
    main()
