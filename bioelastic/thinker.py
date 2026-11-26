"""Thinker: bucle de pensamiento sobre un espacio de trabajo.

En `Cortex` la información va en un solo sentido: entrada -> áreas -> decisión. Acá la
red relee un espacio de trabajo en cada paso interno y decide qué hacer con él:

    contenido:  foco (lo que se está considerando) | registros
    estado:     consigna | reloj interno | señales de la memoria

Piezas:
  * Memoria rápida (`FastMemory`): liga en un disparo cada símbolo con el que llegó
    después y, ante un símbolo, recupera lo que lo siguió. Dura un episodio. Es la
    operación de una «induction head»: [A][B] ... [A] -> [B].
  * Habla interna: lo que la red se dice vuelve a entrar por el mismo canal que la
    percepción. Por eso lo aprendido sobre la percepción sirve para encadenar.
  * Control aprendido. En cada paso interno elige:
        DECIR    responder con su conocimiento de largo plazo
        COPIAR   responder con lo recordado
        SEGUIR   llevar el recuerdo al foco y volver a consultar la memoria
        HABLAR   decirse algo y ponerlo en el foco (sin responder todavía)
    y escribe bits en sus registros. La red no recibe cuántos pasos pensar ni qué
    guardar: sólo la recompensa al final de cada respuesta.

Aprendizaje, todo local:
  * Qué decir: prototipos elásticos sobre el contenido (`BioElasticNet`), con el
    objetivo como tercer factor.
  * Control y registros: actor-crítico TD(lambda). Cada sinapsis lleva una traza de
    elegibilidad y una sola señal global (el error de predicción de recompensa) decide
    el cambio. Es la forma de asignar crédito a lo largo de los pasos sin retropropagar
    en el tiempo.

El controlador ve el *estado* del proceso (consigna, reloj, señales de memoria,
registros), no el contenido: decide cómo pensar, no qué pensar.
"""
from __future__ import annotations

import numpy as np

from .network import BioElasticNet

SAY, COPY, FOLLOW, SPEAK = 0, 1, 2, 3
ACTION_NAMES = ("decir", "copiar", "seguir", "hablar")
_NONE = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))


class FastMemory:
    """Memoria episódica de un disparo: símbolo -> símbolos que lo siguieron."""

    def __init__(self, n: int):
        self.M = np.zeros((n, n), dtype=np.float32)        # M[después, antes] = cuándo
        self.prev: tuple = ()
        self.t = 0

    def reset(self):
        self.M[:] = 0.0
        self.prev = (); self.t = 0

    def bind(self, tokens: tuple):
        """Liga lo anterior con lo actual. Un silencio (tokens vacío) corta la adyacencia."""
        self.t += 1
        if tokens and self.prev:
            if len(tokens) == 1 and len(self.prev) == 1:      # caso común: sin np.ix_
                self.M[tokens[0], self.prev[0]] = self.t
            else:
                self.M[np.ix_(list(tokens), list(self.prev))] = self.t
        self.prev = tuple(tokens)

    def recall(self, keys):
        """Símbolos que siguieron a `keys`; el ligado más reciente llega con fuerza 1."""
        if len(keys) == 0:
            return _NONE
        s = self.M[:, list(keys)].max(axis=1)
        idx = np.flatnonzero(s)
        if len(idx) == 0:
            return _NONE
        return idx, (s[idx] / s[idx].max()).astype(np.float32)

    def has_successor(self, sym: int) -> bool:
        return bool(self.M[:, sym].any())


class Thinker:
    def __init__(self, n_symbols: int, *, n_cues: int = 1, memory: bool = False,
                 speak: bool = False, registers: int = 0, max_steps: int = 4,
                 gamma: float = 0.95, lam: float = 0.9, lr: float = 0.5,
                 temperature: float = 1.0, epsilon: float = 0.1, reg_lr: float = 0.5,
                 hindsight: bool = True, decay: float = 1e-2, seed: int = 0,
                 say: dict | None = None):
        N = self.N = int(n_symbols)
        self.R = int(registers)
        self.max_steps = int(max_steps)
        self.gamma, self.lam, self.lr = float(gamma), float(lam), float(lr)
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)          # exploración uniforme mínima del control
        self.reg_lr = float(reg_lr)
        self.hindsight = bool(hindsight)
        self.decay = float(decay)              # contracción L1 del control: gana la explicación más simple
        self.blame = np.zeros(2, dtype=np.int64)       # errores atribuidos a [estrategia, conocimiento]
        self.rng = np.random.default_rng(seed)
        self.fast = FastMemory(N) if memory else None
        acts = [SAY] + ([COPY, FOLLOW] if memory else []) + ([SPEAK] if speak else [])
        self.actions = np.array(acts)
        # contenido (lo que ve la memoria de largo plazo): foco | registros. El recuerdo no
        # entra: para usarlo hay que copiarlo o llevarlo al foco (acciones del control).
        self.o_reg = N
        self.D_content = N + 2 * self.R
        # estado (lo que ve el controlador): consigna | reloj | señales de memoria | registros
        self.o_cue = 0
        self.o_clock = self.o_cue + int(n_cues)
        self.o_mem = self.o_clock + self.max_steps + 1
        self.o_sreg = self.o_mem + 5
        self.D_state = self.o_sreg + 2 * self.R
        self.say = BioElasticNet(self.D_content, N, seed=seed, **(say or {}))
        self.Wc = np.zeros((4, self.D_state))              # control
        # escritura de registros: pesos iniciales al azar para romper la simetría
        self.Wr = self.rng.normal(0.0, 1.0, (self.R, 2, self.D_content + self.D_state))
        self.wv = np.zeros(self.D_state)                   # crítico
        self.plastic = True
        self.decisions = np.zeros(4, dtype=np.int64)
        self._V = None
        self.reset()

    # ------------------------------------------------------------------ episodio
    def reset(self):
        """Empieza un episodio (y cierra el anterior con su última actualización TD)."""
        if self.plastic and self._V is not None:
            self._update(self._r - self._V)
        self.Ec = np.zeros_like(self.Wc); self.Er = np.zeros_like(self.Wr); self.ev = np.zeros_like(self.wv)
        self._V = None; self._r = 0.0
        self.reg = np.zeros(self.R, dtype=np.int64)
        self._out = -1; self._said = False
        self._wx = None                                    # entrada con la que se escribieron los registros
        self._cf = None                                    # contenido de la última respuesta
        if self.fast is not None:
            self.fast.reset()
        self.say.reset_context()

    @staticmethod
    def _tok(tokens) -> tuple:
        if tokens is None:
            return ()
        if isinstance(tokens, (int, np.integer)):
            return (int(tokens),)
        return tuple(int(t) for t in tokens)

    # ------------------------------------------------------------------ espacio de trabajo
    def _content(self, focus, recall=None, reg=None):
        reg = self.reg if reg is None else reg
        idx = list(focus); val = [1.0] * len(focus)
        for b in range(self.R):
            idx.append(self.o_reg + 2 * b + int(reg[b])); val.append(1.0)
        return np.array(idx, dtype=np.int64), np.array(val, dtype=np.float32)

    def _state(self, cue, clock, recall, reg=None):
        reg = self.reg if reg is None else reg
        idx = [self.o_cue + cue, self.o_clock + clock]
        n = len(recall[0])
        idx.append(self.o_mem + min(n, 2))                                # nada / uno / varios
        if n:
            best = int(recall[0][np.argmax(recall[1])])
            idx.append(self.o_mem + 3 + int(self.fast.has_successor(best)))  # ¿el recuerdo sigue?
        for b in range(self.R):
            idx.append(self.o_sreg + 2 * b + int(reg[b]))
        idx = np.array(idx, dtype=np.int64)
        return idx, np.full(len(idx), 1.0 / np.sqrt(len(idx)), dtype=np.float32)

    # ------------------------------------------------------------------ TD(lambda)
    def _update(self, delta: float):
        step = self.lr * delta
        self.Wc += step * self.Ec; self.Wr += step * self.Er; self.wv += step * self.ev

    def _softmax(self, logits):
        z = logits / self.temperature
        p = np.exp(z - z.max())
        return p / p.sum()

    def _decide(self, s_idx, s_val, allowed, explore):
        """Cierra la transición anterior (TD) y elige la acción de control."""
        if self.plastic:
            if self.decay:                                # L1: lo redundante tiende a cero
                np.copysign(np.maximum(np.abs(self.Wc) - self.decay, 0.0), self.Wc, out=self.Wc)
            V = float(self.wv[s_idx] @ s_val)
            if self._V is not None:
                self._update(self._r + self.gamma * V - self._V)
            self._r = 0.0; self._V = V
            decay = self.gamma * self.lam
            self.Ec *= decay; self.Er *= decay; self.ev *= decay
            self.ev[s_idx] += s_val
        p = self._softmax(self.Wc[allowed][:, s_idx] @ s_val)
        if explore:
            q = (1.0 - self.epsilon) * p + self.epsilon / len(allowed)
            k = int(self.rng.choice(len(allowed), p=q))
        else:
            k = int(np.argmax(p))
        if self.plastic:
            g = -p; g[k] += 1.0
            self.Ec[allowed[:, None], s_idx[None, :]] += g[:, None] * s_val[None, :]
        return int(allowed[k])

    def _write_registers(self, c_idx, c_val, s_idx, s_val, explore):
        """Escribe los registros. Qué escribir se aprende en retrospectiva (ver `feedback`)."""
        if not self.R:
            return
        idx = np.concatenate([c_idx, self.D_content + s_idx])
        val = np.concatenate([c_val / np.sqrt(max(len(c_val), 1)), s_val])
        new = np.zeros(self.R, dtype=np.int64)
        for b in range(self.R):
            p = self._softmax(self.Wr[b][:, idx] @ val)
            bit = int(self.rng.choice(2, p=p)) if explore else int(np.argmax(p))
            new[b] = bit
        self._wx = (idx, val)
        self.reg = new

    def _hindsight(self, target: int):
        """Aprendizaje retrospectivo de los registros (un paso hacia atrás, sin BPTT).

        Con la respuesta ya corregida, la memoria de largo plazo evalúa qué habría dicho
        si cada bit leído hubiera sido el otro. Si sólo uno de los dos valores lleva al
        objetivo, ese es el valor que debió escribirse en el paso anterior, y el escritor
        se corrige hacia él con una regla delta sobre la entrada que tenía entonces."""
        if self._prev_wx is None or self._cf is None:
            return
        c_idx, c_val = self._cf
        idx, val = self._prev_wx
        for b in range(self.R):
            outs = []
            for bit in (0, 1):
                ci = c_idx.copy()
                ci[ci == self.o_reg + 2 * b + (1 - bit)] = self.o_reg + 2 * b + bit
                outs.append(self.say.peek((ci, c_val)))
            ok = [o == target for o in outs]
            if ok[0] != ok[1]:
                want = 1 if ok[1] else 0
                p = self._softmax(self.Wr[b][:, idx] @ val)
                g = -p; g[want] += 1.0
                self.Wr[b][:, idx] += self.reg_lr * g[:, None] * val[None, :]

    # ------------------------------------------------------------------ percibir y responder
    def observe(self, tokens=None):
        """Percibe sin responder. `None` es un silencio."""
        if self.fast is not None:
            self.fast.bind(self._tok(tokens))

    def answer(self, tokens, cue: int = 0, explore: bool = False) -> int:
        """Percibe, piensa los pasos que decida y responde. Devuelve el símbolo (-1 = nada)."""
        focus = self._tok(tokens)
        recall = self.fast.recall(focus) if self.fast is not None else _NONE
        self._prev_wx = self._wx
        self._start = (focus, recall, cue, self.reg.copy())
        self._path = []
        clock = 0
        while True:
            s_idx, s_val = self._state(cue, clock, recall)
            allowed = self.actions
            if clock >= self.max_steps:
                allowed = allowed[(allowed != FOLLOW) & (allowed != SPEAK)]
            a = self._decide(s_idx, s_val, allowed, explore)
            self.decisions[a] += 1
            self._path.append(a)
            c_idx, c_val = self._content(focus, recall)
            if a == FOLLOW:                                   # el recuerdo pasa al foco
                focus = (int(recall[0][np.argmax(recall[1])]),) if len(recall[0]) else ()
            elif a == SPEAK:                                  # habla interna
                focus = (int(self.say.forward((c_idx, c_val))),)
            else:
                break
            recall = self.fast.recall(focus) if self.fast is not None else _NONE
            clock += 1
        if a == COPY:
            out = int(recall[0][np.argmax(recall[1])]) if len(recall[0]) else -1
            self._said = False
        else:
            out = int(self.say.forward((c_idx, c_val)))
            self._said = True
        self._cf = (c_idx, c_val)
        self._write_registers(c_idx, c_val, s_idx, s_val, explore)
        self._out = out
        return out

    def feedback(self, target: int | None = None, reward: float | None = None):
        """Consecuencia de la última respuesta: el objetivo correcto y/o una recompensa.
        Con sólo el objetivo, la recompensa es +1 si acertó y -1 si no."""
        if not self.plastic:
            return
        if target is not None:
            target = int(target)
            if reward is None:
                reward = 1.0 if self._out == target else -1.0
            strategy_error = False
            if self._out != target and self.hindsight:
                path = self._search(target)
                if path is not None:                       # se podía con lo que ya sabe:
                    self._teach_control(path)              # el error es de estrategia
                    strategy_error = True
            self.blame[0 if strategy_error else 1] += int(self._out != target)
            if not strategy_error:
                if self._out != target and (len(self._path) > 1 or not self._said):
                    # Falta conocimiento y la respuesta no salió directa de lo percibido:
                    # se aprende qué decir ante lo percibido.
                    focus0, _, _, reg0 = self._start
                    self.say.forward(self._content(focus0, None, reg0))
                    self._said = True
                if self._said:
                    self.say.teach(target)
                if self.R:
                    self._hindsight(target)
        self._r += float(reward or 0.0)

    # ------------------------------------------------------------------ simulación interna
    def _search(self, target: int):
        """Busca, con el conocimiento actual y sin efectos, las secuencias de pasos internos
        más cortas que habrían llevado al objetivo. Devuelve una lista de caminos
        [(estado, permitidas, acción)], o None si ninguna funciona.

        Sólo cuenta conocimiento establecido (una neurona que ya acertó antes): un acierto
        por casualidad con conocimiento inmaduro no es prueba de que el error fue de estrategia."""
        focus0, recall0, cue, reg = self._start
        frontier = [(focus0, recall0, [])]
        for clock in range(self.max_steps + 1):
            nxt = []; found = []
            for focus, recall, path in frontier:
                s = self._state(cue, clock, recall, reg)
                allowed = self.actions
                if clock >= self.max_steps:
                    allowed = allowed[(allowed != FOLLOW) & (allowed != SPEAK)]
                c = self._content(focus, recall, reg)
                peek = None                                # DECIR y HABLAR consultan lo mismo: una vez
                for a in allowed:
                    step = path + [(s, allowed, int(a))]
                    if a == SAY:
                        out, sure = peek = peek or self.say.peek(c, confidence=True)
                        if sure and out == target:
                            found.append(step)
                    if a == COPY and len(recall[0]) and int(recall[0][np.argmax(recall[1])]) == target:
                        found.append(step)
                    if a == FOLLOW and len(recall[0]):
                        f = (int(recall[0][np.argmax(recall[1])]),)
                        nxt.append((f, self.fast.recall(f), step))
                    if a == SPEAK:
                        out, sure = peek = peek or self.say.peek(c, confidence=True)
                        if sure:                           # sólo se encadena conocimiento establecido
                            f = (out,)
                            nxt.append((f, self.fast.recall(f) if self.fast is not None else _NONE, step))
            if found:
                return found
            frontier = nxt
            if not frontier:
                break
        return None

    def _teach_control(self, paths):
        """Corrige el control hacia las secuencias que habrían funcionado (regla delta local,
        repartida entre los caminos igual de cortos)."""
        w = self.lr / len(paths)
        for path in paths:
            for (s_idx, s_val), allowed, a in path:
                p = self._softmax(self.Wc[allowed][:, s_idx] @ s_val)
                g = -p; g[list(allowed).index(a)] += 1.0
                self.Wc[allowed[:, None], s_idx[None, :]] += w * g[:, None] * s_val[None, :]

    def stats(self) -> dict:
        return dict(neurons=self.say.layer.n,
                    **{name: int(n) for name, n in zip(ACTION_NAMES, self.decisions)})
