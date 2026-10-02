# High-level API: read a long text with a frozen model through 2K bounded probes.
#
#   from periscope import Periscope
#   p = Periscope("http://localhost:8000", "Qwen/Qwen3.5-4B", window=131072)
#   out = p.answer(text, "Who ...?", ["option 1", "option 2", "option 3", "option 4"])
#   out["select"], out["direct"], out["selected"], out["peak"]
#   scores = p.rank("query", {"doc1": text1, "doc2": text2}, chunk_size=100)

import math
import string

from transformers import AutoTokenizer

from . import core as C


def letters(n):
    """Answer letters for n options and the abstain letter after them (4 options: A-D and E)."""
    if not 2 <= n <= 25:
        raise ValueError("between 2 and 25 options")
    return list(string.ascii_uppercase[:n]), string.ascii_uppercase[n]


def chat(tokenizer, body):
    # A first-token readout cannot survive a leading reasoning block, so thinking is disabled where the
    # template defines it. The argument is ignored by templates that do not.
    return tokenizer.apply_chat_template([{"role": "user", "content": body}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)


class Periscope:
    """Periscope over one served model.

    server, model      the probe model, served with `vllm serve <model> --max-model-len <window>`
    window             the server's --max-model-len, used to skip a call that would not fit
    chunk_size         tokens per chunk (500 for question answering, 100 for retrieval in the paper)
    read_server/model  optional second model that performs the selected read (cross-model reading)
    preamble           text placed before every question prompt (e.g. an instruction line)
    """

    def __init__(self, server, model, window=131072, chunk_size=500, batch=64,
                 read_server=None, read_model=None, read_window=None, preamble=""):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.client = C.Client(server, model, batch=batch)
        self.window = window
        self.chunk_size = chunk_size
        self.preamble = preamble
        if read_server:
            self.read_tok = AutoTokenizer.from_pretrained(read_model or model)
            self.read_client = C.Client(read_server, read_model or model, batch=batch)
            self.read_window = read_window or window
        else:
            self.read_tok, self.read_client, self.read_window = self.tok, self.client, window
        self._check()

    # ------------------------------------------------------------------ prompts
    def qa_prompt(self, text, question, options, abstain, tokenizer=None):
        ys, ab = letters(len(options))
        lines = ["%s. %s" % (y, o) for y, o in zip(ys, options)]
        if abstain:
            lines.append("%s. Unsure" % ab)
        body = "%s\n\nQuestion: %s\n%s\nAnswer with a single letter:" % (text, question, "\n".join(lines))
        return chat(tokenizer or self.tok, self.preamble + body)

    def relevance_prompt(self, span, query):
        # The document comes first so that a span is a prefix shared across queries.
        return chat(self.tok, "Document: %s\nQuery: %s\n"
                              "Is this document relevant to the query? Answer Yes or No:" % (span, query))

    def _check(self):
        top = self.client.top_logprobs([self.qa_prompt("The sky is blue.", "What colour is the sky?",
                                                       ["green", "blue", "red", "black"], True)])[0]
        if "B" not in C.collapse(top, {"A", "B", "C", "D", "E"})[0]:
            raise RuntimeError("no option letter in the top 20 of a trivial probe: check the chat template")

    # ------------------------------------------------------------------ probes
    def probe(self, text, question, options, strided=True):
        """The 2K probes of one text, issued as one batch. Returns the chunks, K, and the per-span
        log-odds L[i] and S[j] (dicts answer -> score)."""
        ys, ab = letters(len(options))
        chunks = C.chunk(self.tok, text, self.chunk_size) or ["empty"]
        k, local, strd = C.grid(chunks)
        spans = local + (strd if strided else [])
        prompts = [self.qa_prompt(s, question, options, abstain=True) for _, s in spans]
        lengths = [len(e) for e in self.tok(prompts, add_special_tokens=False)["input_ids"]]
        keep = [i for i, n in enumerate(lengths) if n < self.window]
        values = dict(zip(keep, self.client.top_logprobs([prompts[i] for i in keep]) if keep else []))
        L = {local[i][0]: C.log_odds(values[i], ys, ab) for i in range(len(local)) if i in values}
        S = {strd[j][0]: C.log_odds(values[len(local) + j], ys, ab)
             for j in range(len(spans) - len(local)) if len(local) + j in values}
        return {"chunks": chunks, "k": k, "L": L, "S": S, "calls": len(keep),
                "probe_tokens": sum(lengths[i] for i in keep)}

    def read(self, text, question, options):
        """One read by the read model, answers without the abstain option. None if it does not fit."""
        ys, _ = letters(len(options))
        p = self.qa_prompt(text, question, options, abstain=False, tokenizer=self.read_tok)
        n = len(self.read_tok.encode(p, add_special_tokens=False))
        if n >= self.read_window:
            return None, None, n
        scores = C.read_scores(self.read_client.top_logprobs([p])[0], ys)
        return max(ys, key=scores.get), scores, n

    # ------------------------------------------------------------------ readouts
    def answer(self, text, question, options, budget=None, read=True):
        """Both question-answering readouts from one set of probes.
        direct: argmax of score(y) = max_i L_i(y) + max_j S_j(y), no further call.
        select: one read of the `budget` chunks (default K) with the highest cells of the map."""
        ys, _ = letters(len(options))
        st = self.probe(text, question, options)
        out = {"k": st["k"], "n_chunks": len(st["chunks"]), "calls": st["calls"],
               "probe_tokens": st["probe_tokens"], "direct": None, "select": None}
        if not st["L"]:
            return out
        scores = C.compose(st["L"], st["S"])
        cells = C.evidence_map(st["L"], st["S"], ys)
        out.update(direct=max(ys, key=scores.get), scores=scores, peak=max(cells, key=cells.get),
                   map={"%d,%d" % ij: v for ij, v in cells.items()})
        if read:
            picked = C.select(cells, st["k"], len(st["chunks"]), budget or st["k"])
            pred, rs, n = self.read(" ".join(st["chunks"][i] for i in picked), question, options)
            out.update(select=pred, read_scores=rs, selected=picked, read_tokens=n)
            out["calls"] += 1
        return out

    def window_read(self, text, question, options, manner="prefix"):
        """The baseline: one read of the longest prefix that fits the window."""
        ids = self.read_tok.encode(text, add_special_tokens=False)
        kept = C.fit_prefix(self.read_tok, ids,
                            lambda t: self.qa_prompt(t, question, options, False, self.read_tok),
                            self.read_window, manner)
        pred, scores, n = self.read(self.read_tok.decode(kept, skip_special_tokens=True), question, options)
        return {"pred": pred, "scores": scores, "read_tokens": n, "truncated": int(len(kept) < len(ids))}

    def rank(self, query, documents, strided=True):
        """The ranking readout: one relevance score per document, score(Yes) = max_i L_i + max_j S_j."""
        out = {}
        for doc_id, text in documents.items():
            chunks = C.chunk(self.tok, text, self.chunk_size) or ["empty"]
            _, local, strd = C.grid(chunks)
            spans = [("l", s) for _, s in local] + ([("s", s) for _, s in strd] if strided else [])
            prompts = [self.relevance_prompt(s, query) for _, s in spans]
            lengths = [len(e) for e in self.tok(prompts, add_special_tokens=False)["input_ids"]]
            keep = [i for i, n in enumerate(lengths) if n <= self.window]
            best = {}
            for i, top in zip(keep, self.client.top_logprobs([prompts[i] for i in keep]) if keep else []):
                v = C.log_odds(top, ["Yes"], "No")["Yes"]
                best[spans[i][0]] = max(best.get(spans[i][0], -1e9), v)
            out[doc_id] = best.get("l", -1e9) + best.get("s", 0.0 if not strided else -1e9)
        return out
