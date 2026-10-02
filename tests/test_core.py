# Self-test of the composition rule, the map and the readout. Needs no GPU and no server.
#   python tests/test_core.py

import math
import random

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from periscope import core as P  # noqa: E402

random.seed(0)
ANSWERS = ["A", "B", "C", "D"]

# the grid covers every chunk exactly once in each family
for n in [1, 2, 4, 5, 9, 10, 17, 100, 101]:
    chunks = ["c%d" % i for i in range(n)]
    k, local, strided = P.grid(chunks)
    assert k == math.ceil(math.sqrt(n)), n
    for spans in (local, strided):
        seen = [c for _, s in spans for c in s.split(" ")]
        assert sorted(seen) == sorted(chunks), (n, spans)
    assert all(len(s.split(" ")) <= k for _, s in local + strided)

# the peak of the map equals the winning answer's composed score
for _ in range(200):
    k = random.randint(2, 6)
    L = {i: {y: random.uniform(-9, 9) for y in ANSWERS} for i in range(k)}
    S = {j: {y: random.uniform(-9, 9) for y in ANSWERS} for j in range(k)}
    composed = P.compose(L, S)
    cells = P.evidence_map(L, S, ANSWERS)
    assert abs(max(cells.values()) - max(composed.values())) < 1e-9

    # the peak's cell is the local and strided span that produced the answer
    best = max(composed, key=composed.get)
    i, j = max(cells, key=cells.get)
    assert abs(L[i][best] + S[j][best] - composed[best]) < 1e-9

# select returns chunk indices in the original reading order, inside the grid
cells = {(i, j): random.uniform(0, 1) for i in range(4) for j in range(4)}
picked = P.select(cells, 4, 13, 4)
assert picked == sorted(picked) and len(picked) == 4 and max(picked) < 13

# the four cases of the log-odds, with no default clamp
assert abs(P.log_odds({"A": -1.0, "E": -3.0}, ANSWERS, "E")["A"] - 2.0) < 1e-9
floor_only = P.log_odds({"B": -2.0, "E": -4.0}, ANSWERS, "E")
assert floor_only["A"] < floor_only["B"]
# surface variants collapse to the larger of the two
assert abs(P.log_odds({" A": -1.0, "A": -5.0, "E": -2.0}, ANSWERS, "E")["A"] - 1.0) < 1e-9

print("ok")
