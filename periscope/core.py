# Core of Periscope: chunking, the K x K grid, the probe readout, the composition rule, the evidence map.
# Every call goes to an OpenAI-compatible endpoint, e.g. a `vllm serve` process.

import math

import numpy as np
import requests


class Client:
    """One forward pass per prompt, read at the first answer token."""

    def __init__(self, server, model, batch=64, timeout=3600):
        self.url = server.rstrip("/") + "/v1/completions"
        self.model = model
        self.batch = batch
        self.timeout = timeout

    def _post(self, payload):
        r = requests.post(self.url, json={"model": self.model, **payload}, timeout=self.timeout)
        if r.status_code != 200:
            raise RuntimeError("server %d: %s" % (r.status_code, r.text[:400]))
        return sorted(r.json()["choices"], key=lambda c: c["index"])

    def top_logprobs(self, prompts):
        # Temperature rescales every log-odds by the same constant and cannot change a ranking.
        # Nucleus and top-k truncation can drop an answer token, so both are disabled.
        out = []
        for i in range(0, len(prompts), self.batch):
            for c in self._post({"prompt": prompts[i:i + self.batch], "max_tokens": 1, "logprobs": 20,
                                 "temperature": 1.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0}):
                out.append((c.get("logprobs") or {}).get("top_logprobs", [{}])[0] or {})
        return out

    def generate(self, prompt, max_tokens):
        return self._post({"prompt": prompt, "max_tokens": max_tokens, "temperature": 0.0})[0]["text"]


class Embedder:
    """An OpenAI-compatible embeddings endpoint, e.g. `vllm serve <embedding model> --task embed`."""

    def __init__(self, server, model, batch=64, timeout=3600):
        self.url = server.rstrip("/") + "/v1/embeddings"
        self.model = model
        self.batch = batch
        self.timeout = timeout

    def __call__(self, texts):
        out = []
        for i in range(0, len(texts), self.batch):
            r = requests.post(self.url, json={"model": self.model, "input": texts[i:i + self.batch]},
                              timeout=self.timeout)
            r.raise_for_status()
            out.extend(np.array(d["embedding"]) for d in sorted(r.json()["data"], key=lambda d: d["index"]))
        e = np.stack(out)
        return e / np.linalg.norm(e, axis=1, keepdims=True)


def collapse(topmap, keys):
    # The surface form of an answer token depends on the template ("A" against " A"),
    # so take the maximum over surface variants.
    out = {}
    for k, v in topmap.items():
        s = k.strip()
        if s in keys:
            out[s] = max(out.get(s, -1e9), float(v))
    floor = (min(float(v) for v in topmap.values()) - 2.0) if topmap else -1e9
    return out, floor


def log_odds(topmap, answers, abstain):
    """Eq. 1: the log-odds of each answer against the abstain option, with no default clamp."""
    lp, floor = collapse(topmap, set(answers) | {abstain})
    has_a = abstain in lp
    out = {}
    for y in answers:
        if y in lp and has_a:
            out[y] = lp[y] - lp[abstain]
        elif y in lp:
            out[y] = lp[y] - floor
        elif has_a:
            out[y] = floor - lp[abstain]
        else:
            out[y] = floor - 50.0
    return out


def read_scores(topmap, answers):
    """A read has no abstain option: the raw log-probability of each answer."""
    lp, floor = collapse(topmap, set(answers))
    return {y: lp.get(y, floor) for y in answers}


def chunk(tokenizer, text, chunk_size):
    ids = tokenizer.encode(text, add_special_tokens=False)
    pieces = [ids[i:i + chunk_size] for i in range(0, len(ids), chunk_size)]
    decoded = tokenizer.batch_decode(pieces, skip_special_tokens=True)
    return [t for t in (x.strip() for x in decoded) if t]


def grid(chunks):
    """K = ceil(sqrt(N)), padded to K^2 with empty chunks that are dropped from the spans, so no chunk
    is discarded and an empty span is not issued. Chunks of a span are joined by a single space.
    Returns K, the local spans [(i, text)] and the strided spans [(j, text)]."""
    k = math.ceil(math.sqrt(len(chunks)))
    padded = chunks + [""] * (k * k - len(chunks))
    cells = [padded[i * k:(i + 1) * k] for i in range(k)]
    local = [(i, " ".join(c for c in cells[i] if c)) for i in range(k)]
    strided = [(j, " ".join(cells[i][j] for i in range(k) if cells[i][j])) for j in range(k)]
    return k, [(i, s) for i, s in local if s], [(j, s) for j, s in strided if s]


def compose(local, strided):
    """Eq. 3: score(y) = max_i L_i(y) + max_j S_j(y)."""
    answers = sorted({y for d in list(local.values()) + list(strided.values()) for y in d})
    out = {}
    for y in answers:
        s = max(d[y] for d in local.values())
        if strided:
            s += max(d[y] for d in strided.values())
        out[y] = s
    return out


def evidence_map(local, strided, answers):
    """Eq. 4: M[i,j] = max_y (L_i(y) + S_j(y)), cell (i, j) being chunk i*K + j (0-based).
    The largest cell for y equals score(y), so the peak is the chunk behind the answer."""
    return {(i, j): max(li[y] + sj[y] for y in answers)
            for i, li in local.items() for j, sj in strided.items()}


def select(cells, k, n_chunks, budget):
    """The highest cells of the map, as chunk indices in their original reading order."""
    picked = []
    for i, j in sorted(cells, key=cells.get, reverse=True):
        if i * k + j < n_chunks:
            picked.append(i * k + j)
        if len(picked) == budget:
            break
    return sorted(picked)


def fit_prefix(tokenizer, ids, build, window, manner="prefix"):
    """The window read: the longest prefix whose whole prompt fits the window.
    manner="middle" keeps head and tail instead."""
    def cut(n):
        if manner == "prefix" or n >= len(ids):
            return ids[:n]
        return ids[:n // 2] + ids[-(n - n // 2):]

    overhead = len(tokenizer.encode(build(""), add_special_tokens=False)) + 16
    n = min(len(ids), max(0, window - overhead))
    while n > 0:
        text = tokenizer.decode(cut(n), skip_special_tokens=True)
        if len(tokenizer.encode(build(text), add_special_tokens=False)) < window:
            break
        n = int(n * 0.98)
    return cut(n)
