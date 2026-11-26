"""Texto: predicción online del próximo byte (tiny-shakespeare), sin vocabulario previo.

La red ve los últimos 8 bytes por la línea de retardos. Referencia: modelo de
n-gramas con backoff, entrenado online sobre el mismo flujo (mismo test-then-train).

  python experiments/exp_text.py [n_bytes]
"""
from __future__ import annotations

import sys
import time
from collections import Counter, defaultdict

from common import CsvLog
from bioelastic import BioElasticNet, TextEncoder
from bioelastic.datasets import load_text

WIN = 20000


class NGram:
    """Backoff al contexto más largo ya visto; predice el sucesor más frecuente."""

    def __init__(self, order: int):
        self.order = order
        self.tabs = [defaultdict(Counter) for _ in range(order + 1)]

    def step(self, b: bytes, t: int) -> int:
        pred = 32
        top = min(self.order, t + 1)
        for o in range(top, -1, -1):
            c = self.tabs[o].get(b[t + 1 - o:t + 1])
            if c:
                pred = c.most_common(1)[0][0]
                break
        y = b[t + 1]
        for o in range(top + 1):
            self.tabs[o][b[t + 1 - o:t + 1]][y] += 1
        return pred

    def size(self) -> int:
        return sum(len(t) for t in self.tabs)


def main():
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 400000
    raw = load_text()[:N]
    data = TextEncoder()(raw)
    net = BioElasticNet(256, 256, context=8, delay=0.1, layer=dict(max_neurons=30000))
    grams = {o: NGram(o) for o in (2, 3, 5)}
    log = CsvLog("text_online", ["byte", "acc_win", "acc_2gram", "acc_3gram", "acc_5gram",
                                 "neurons", "created", "pruned", "ctx_5gram"])
    win = 0; gw = dict.fromkeys(grams, 0); t0 = time.time()
    for t in range(N - 1):
        y = int(data[t + 1])
        pred = net.forward(int(data[t]))
        net.teach(y)
        win += pred == y
        for o, g in grams.items():
            gw[o] += g.step(raw, t) == y
        if (t + 1) % WIN == 0:
            st = net.stats()
            log.row(t + 1, win / WIN, gw[2] / WIN, gw[3] / WIN, gw[5] / WIN,
                    st["neurons"], st["created"], st["pruned"], grams[5].size())
            win = 0; gw = dict.fromkeys(grams, 0)
    log.close()
    print(f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
