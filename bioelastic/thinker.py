"""Thinker: bucle de pensamiento sobre un espacio de trabajo.

En `Cortex` la información va en un solo sentido: entrada -> áreas -> decisión. Acá la
red relee un espacio de trabajo en cada paso interno y decide qué hacer con él:

    contenido:  foco (lo que se está considerando) | registros
    estado:     consigna | copia eferente (qué hizo recién) | señales de la memoria
                | coincidencias con lo sostenido | registros

Piezas:
  * Memoria rápida (`FastMemory`): liga en un disparo cada símbolo con el que llegó
    después y, ante un símbolo, recupera lo que lo siguió. Dura un episodio. Es la
    operación de una «induction head»: [A][B] ... [A] -> [B]. Con varias claves a la
    vez la recuperación es conjuntiva (gana lo ligado a más claves): ligadura de roles.
  * Habla interna: lo que la red se dice vuelve a entrar por el mismo canal que la
    percepción. Por eso lo aprendido sobre la percepción sirve para encadenar.
  * Sostén: una pregunta puede traer, aparte del foco, algo que sostener (`probe`,
    p. ej. el segundo término de «¿X > Y?»). El controlador no ve qué es: sólo si el
    foco o el recuerdo coinciden con eso (detección de coincidencia).
  * Control aprendido. En cada paso interno elige:
        DECIR    responder con su conocimiento de largo plazo sobre el contenido
        COPIAR   responder con lo recordado
        SEGUIR   llevar el recuerdo al foco y volver a consultar la memoria
        HABLAR   decirse algo y ponerlo en el foco (sin responder todavía)
        EMITIR   responder con lo que tiene en el foco
        JUZGAR   responder con un juicio sobre el *estado* del proceso (sí/no, ...)
    y escribe bits en sus registros. La red no recibe cuántos pasos pensar ni qué
    guardar: sólo el objetivo o la recompensa al final de cada respuesta.

Aprendizaje, todo local:
  * Qué decir: prototipos elásticos sobre el contenido (`BioElasticNet`), con el
    objetivo como tercer factor. Qué juzgar: regla delta sobre el estado.
  * Control y registros: actor-crítico TD(lambda) con trazas de elegibilidad y una sola
    señal global (el error de predicción de recompensa).
  * Retrospectiva: ante un error, una búsqueda interna acotada y guiada por la propia
    política (de lo más probable a lo menos, con presupuesto fijo) simula, sin efectos,
    qué secuencia de pasos habría llevado al objetivo con el conocimiento que ya tiene.
    Si la encuentra, el error fue de estrategia: se corrige el control y no se toca el
    conocimiento. Si no, falta conocimiento: se enseña al módulo que respondió (y sólo
    a él), sobre lo que tenía delante cuando respondió.

El controlador ve el *estado* del proceso (consigna, copia eferente, señales de memoria,
coincidencias, registros), no el contenido: decide cómo pensar, no qué pensar. No ve un
reloj absoluto: sabe qué hizo recién (copia eferente) y qué le devuelve la memoria. Con
varias consignas, cada señal llega conjugada con la consigna (selectividad mixta, como en
corteza prefrontal): por defecto *sólo* conjugada, para que una tarea no pise el control
de otra (con pesos compartidos, una semilla de cada tres colapsaba en multitarea).

Conocimiento establecido: una neurona que ya acertó y cuya lectura es consistente. Lo
establecido para esta misma situación no se pisa con una sola contradicción (si no, los
errores de estrategia en tareas episódicas escriben objetivos al azar sobre los mismos
símbolos que guardan el conocimiento permanente).
"""
from __future__ import annotations

import heapq
import itertools

import numpy as np

from .network import BioElasticNet

SAY, COPY, FOLLOW, SPEAK, EMIT, JUDGE = range(6)
ACTION_NAMES = ("decir", "copiar", "seguir", "hablar", "emitir", "juzgar")
_N_ACTIONS = len(ACTION_NAMES)
_INTERNAL = (FOLLOW, SPEAK)
_NONE = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))


class FastMemory:
    """Memoria episódica de un disparo: cada transición (lo anterior -> lo actual) es un
    evento con su propia clave conjuntiva, como un índice hipocampal. Dura un episodio."""

    def __init__(self, n: int):
        self.n = int(n)
        self.events: list = []                             # (clave: frozenset, símbolos, cuándo)
        self.prev: tuple = ()
        self.t = 0

    def reset(self):
        self.events = []
        self.prev = (); self.t = 0

    def bind(self, tokens: tuple):
        """Liga lo anterior con lo actual. Un silencio (tokens vacío) corta la adyacencia."""
        self.t += 1
        if tokens and self.prev:
            self.events.append((frozenset(self.prev), tuple(tokens), self.t))
        self.prev = tuple(tokens)

    def recall(self, keys):
        """Símbolos que siguieron a `keys`. Completa el patrón: ganan los eventos cuya clave
        comparte más símbolos con `keys` (conjunción) y, entre lo recuperado, lo ligado más
        recientemente llega con fuerza 1."""
        if len(keys) == 0 or not self.events:
            return _NONE
        keys = set(keys)
        over = [len(k & keys) for k, _, _ in self.events]
        top = max(over)
        if top == 0:
            return _NONE
        when: dict = {}
        for (k, toks, t), o in zip(self.events, over):
            if o == top:
                for x in toks:
                    when[x] = max(when.get(x, 0), t)
        idx = np.fromiter(when.keys(), dtype=np.int64, count=len(when))
        s = np.fromiter(when.values(), dtype=np.float32, count=len(when))
        return idx, s / s.max()

    def has_successor(self, sym: int) -> bool:
        return any(sym in k for k, _, _ in self.events)


class Thinker:
    def __init__(self, n_symbols: int, *, n_cues: int = 1, memory: bool = False,
                 speak: bool = False, judge: bool = False, copy: bool = True,
                 registers: int = 0, max_steps: int = 4, budget: int = 64,
                 gamma: float = 0.95, lam: float = 0.9, lr: float = 0.5,
                 temperature: float = 1.0, epsilon: float = 0.1, reg_lr: float = 0.5,
                 say_sure: float = 0.7, branch: float = 0.0, reliability: float = 3.0, protect: bool = True, patience: int = 3, hindsight: bool = True,
                 decay: float = 1e-2, mixed: bool | str = "solo", clock: bool = False,
                 lookahead: bool = False, search: str = "guiada", credit: str = "eslabon",
                 seed: int = 0, say: dict | None = None):
        N = self.N = int(n_symbols)
        self.R = int(registers)
        self.max_steps = int(max_steps)
        self.budget = int(budget)              # nodos que puede expandir la búsqueda interna
        self.gamma, self.lam, self.lr = float(gamma), float(lam), float(lr)
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)          # exploración uniforme mínima del control
        self.reg_lr = float(reg_lr)
        self.say_sure = float(say_sure)
        self.branch = float(branch)            # >0: la búsqueda abre hipótesis sobre conocimiento inconsistente
        # >0: la búsqueda paga -reliability*log(fiabilidad) por responder con una acción en un
        # estado donde esa respuesta acertó poco (estado ambiguo): prefiere juntar información
        self.reliability = float(reliability)
        self.rel: dict = {}                    # (estado, respuesta) -> [aciertos, intentos]
        # Proteger el conocimiento choca con registros cuyo significado se aprende a la par
        # (el lector se congela antes que el escritor: T3 cae de 5/5 a 2/5 semillas). Con
        # registros no se protege.
        self.protect = bool(protect) and self.R == 0
        self.patience = int(patience)
        self._contra: dict = {}                # contradicciones consistentes pendientes por situación
        self.hindsight = bool(hindsight)
        self.decay = float(decay)              # contracción L1 del control: gana la explicación más simple
        self.search = str(search)              # "guiada" (mejor primero, acotada) o "amplitud" (exhaustiva)
        # a quién se enseña si falta conocimiento: "eslabon" (al primer paso que se apoyó en
        # conocimiento no establecido), "estricto" (ídem, pero si ese paso es intermedio no se
        # enseña nada) o "percepcion" (siempre a lo percibido, como la versión anterior)
        self.credit = str(credit)
        self.blame = np.zeros(2, dtype=np.int64)       # errores atribuidos a [estrategia, conocimiento]
        self.expanded = 0                              # nodos expandidos por la búsqueda (costo)
        self.rng = np.random.default_rng(seed)
        self.fast = FastMemory(N) if memory else None
        acts = [SAY]
        if memory:
            acts += ([COPY] if copy else []) + [FOLLOW, EMIT]
        acts += ([SPEAK] if speak else []) + ([JUDGE] if judge else [])
        self.actions = np.array(acts)
        # contenido (lo que ve la memoria de largo plazo): foco | registros. El recuerdo no
        # entra: para usarlo hay que copiarlo o llevarlo al foco (acciones del control).
        self.o_reg = N
        self.D_content = N + 2 * self.R
        # estado (lo que ve el controlador)
        self.n_cues = int(n_cues)
        self.clock = bool(clock)               # reloj absoluto (sólo para comparar con la versión anterior)
        self.lookahead = bool(lookahead)       # «¿el recuerdo sigue?» (ídem)
        o = 1                                  # 0: sesgo
        self.o_cue = o; o += self.n_cues
        self.o_last = o; o += _N_ACTIONS + 1   # copia eferente: nada / acción anterior
        self.o_mem = o; o += 3                 # recuerdo: nada / uno / varios
        self.o_match = o; o += 4               # ¿foco sostenido? ¿recuerdo sostenido?
        self.o_sreg = o; o += 2 * self.R
        self.o_clock = o; o += (self.max_steps + 1) if self.clock else 0
        self.o_look = o; o += 2 if self.lookahead else 0
        self.D_base = o
        self.mixed = mixed if (mixed and self.n_cues > 1) else False
        self._shared = self.mixed != "solo"            # "solo": sólo señales conjugadas con la consigna
        self.D_state = self.D_base * (1 + self.n_cues) if self.mixed else self.D_base
        self.say = BioElasticNet(self.D_content, N, seed=seed, **(say or {}))
        self.Wc = np.zeros((_N_ACTIONS, self.D_state))     # control
        # juicio (estado -> respuesta): prototipos conjuntivos sobre el estado, para que un
        # estado ambiguo quede ambiguo y no herede confianza de los vecinos
        self.judge = BioElasticNet(self.D_state, N, seed=seed + 1) if judge else None
        # escritura de registros: pesos iniciales al azar para romper la simetría
        self.Wr = self.rng.normal(0.0, 1.0, (self.R, 2, self.D_content + self.D_state))
        self.wv = np.zeros(self.D_state)                   # crítico
        self.plastic = True
        self.decisions = np.zeros(_N_ACTIONS, dtype=np.int64)
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
        self._out = -1; self._path = []
        self._wx = None                                    # entrada con la que se escribieron los registros
        self._cf = None                                    # contenido de la última respuesta dicha
        self._js = None                                    # estado de la última respuesta juzgada
        if self.fast is not None:
            self.fast.reset()
        self.say.reset_context()
        if self.judge is not None:
            self.judge.reset_context()

    @staticmethod
    def _tok(tokens) -> tuple:
        if tokens is None:
            return ()
        if isinstance(tokens, (int, np.integer)):
            return (int(tokens),)
        return tuple(int(t) for t in tokens)

    # ------------------------------------------------------------------ espacio de trabajo
    def _recall(self, focus):
        return self.fast.recall(focus) if self.fast is not None else _NONE

    @staticmethod
    def _best(recall):
        return int(recall[0][np.argmax(recall[1])]) if len(recall[0]) else -1

    def _content(self, focus, recall=None, reg=None):
        reg = self.reg if reg is None else reg
        idx = list(focus); val = [1.0] * len(focus)
        for b in range(self.R):
            idx.append(self.o_reg + 2 * b + int(reg[b])); val.append(1.0)
        return np.array(idx, dtype=np.int64), np.array(val, dtype=np.float32)

    def _state(self, cue, last, clock, focus, recall, probe, reg=None):
        reg = self.reg if reg is None else reg
        best = self._best(recall)
        idx = [0, self.o_cue + cue, self.o_last + last + 1,
               self.o_mem + min(len(recall[0]), 2)]                       # nada / uno / varios
        if probe:
            idx.append(self.o_match + int(any(t in probe for t in focus)))
            idx.append(self.o_match + 2 + int(best in probe))
        for b in range(self.R):
            idx.append(self.o_sreg + 2 * b + int(reg[b]))
        if self.clock:
            idx.append(self.o_clock + clock)
        if self.lookahead and best >= 0:
            idx.append(self.o_look + int(self.fast.has_successor(best)))
        if self.mixed:                                     # cada señal, también conjugada con la consigna
            idx = (idx if self._shared else []) + [(1 + cue) * self.D_base + i for i in idx]
        idx = np.array(idx, dtype=np.int64)
        return idx, np.full(len(idx), 1.0 / np.sqrt(len(idx)), dtype=np.float32)

    def _allowed(self, clock):
        if clock >= self.max_steps:
            return self.actions[(self.actions != FOLLOW) & (self.actions != SPEAK)]
        return self.actions

    # ------------------------------------------------------------------ TD(lambda)
    def _update(self, delta: float):
        step = self.lr * delta
        self.Wc += step * self.Ec; self.Wr += step * self.Er; self.wv += step * self.ev

    def _softmax(self, logits):
        z = logits / self.temperature
        p = np.exp(z - z.max())
        return p / p.sum()

    def _policy(self, s_idx, s_val, allowed):
        return self._softmax(self.Wc[allowed][:, s_idx] @ s_val)

    def _decide(self, s_idx, s_val, allowed, explore):
        """Cierra la transición anterior (TD) y elige la acción de control."""
        if self.plastic:
            if self.decay:                                # L1 local: sólo sinapsis con presináptica activa
                W = self.Wc[:, s_idx]
                self.Wc[:, s_idx] = np.sign(W) * np.maximum(np.abs(W) - self.decay, 0.0)
            V = float(self.wv[s_idx] @ s_val)
            if self._V is not None:
                self._update(self._r + self.gamma * V - self._V)
            self._r = 0.0; self._V = V
            decay = self.gamma * self.lam
            self.Ec *= decay; self.Er *= decay; self.ev *= decay
            self.ev[s_idx] += s_val
        p = self._policy(s_idx, s_val, allowed)
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

    def _know(self, c, net=None):
        """Qué diría un módulo de conocimiento (`say` por defecto, o el juicio) ante `c`, sin
        efectos, y si es conocimiento establecido: una neurona que ya acertó antes y cuya
        lectura es consistente (probabilidad >= `say_sure`). Lo que la red memorizó con
        objetivos que se contradicen no cuenta como prueba de nada."""
        net = self.say if net is None else net
        v = net.layer.forward(*c)
        V = net.readout.potentials(v.winners, net._votes(v))
        out = int(np.argmax(V))
        if v.novel or not len(v.winners) or net.layer.hits[int(v.winners[0])] < 1:
            return out, False
        return out, bool(net.readout.probs(V)[out] >= self.say_sure)

    def _protected(self, c, net, target):
        """¿Es conocimiento consistente establecido para *esta misma* situación (no por
        parecido con otra)? Eso no se pisa con una contradicción suelta ni con objetivos que
        cambian cada vez (lo episódico), pero sí cuando el mismo objetivo alternativo se
        repite `patience` veces seguidas: eso es evidencia de que el conocimiento era otro.
        Lo generalizado desde un vecino siempre se corrige."""
        if not self.protect or not self._know(c, net)[1]:
            return False
        v = net.layer.forward(*c)
        if v.cos[0] < net.rho_sup:
            return False
        key = (id(net), c[0].tobytes())
        alt, n = self._contra.get(key, (None, 0))
        n = n + 1 if alt == target else 1
        if n >= self.patience:
            self._contra.pop(key, None)
            return False
        self._contra[key] = (target, n)
        return True

    # ------------------------------------------------------------------ percibir y responder
    def observe(self, tokens=None):
        """Percibe sin responder. `None` es un silencio."""
        if self.fast is not None:
            self.fast.bind(self._tok(tokens))

    def answer(self, tokens, cue: int = 0, explore: bool = False, probe=None) -> int:
        """Percibe, piensa los pasos que decida y responde. Devuelve el símbolo (-1 = nada).
        `probe`: lo que la pregunta pide sostener aparte del foco (opcional)."""
        focus = self._tok(tokens)
        probe = frozenset(self._tok(probe))
        recall = self._recall(focus)
        self._prev_wx = self._wx
        self._start = (focus, recall, cue, self.reg.copy(), probe)
        self._path = []; self._consults = []
        clock, last = 0, -1
        while True:
            s_idx, s_val = self._state(cue, last, clock, focus, recall, probe)
            a = self._decide(s_idx, s_val, self._allowed(clock), explore)
            self.decisions[a] += 1
            self._path.append(a)
            c_idx, c_val = self._content(focus)
            if a == FOLLOW:                                   # el recuerdo pasa al foco
                b = self._best(recall)
                focus = (b,) if b >= 0 else ()
            elif a == SPEAK:                                  # habla interna
                self._consults.append(((c_idx, c_val), self._know((c_idx, c_val))[1]))
                focus = (int(self.say.forward((c_idx, c_val))),)
            else:
                break
            recall = self._recall(focus)
            clock += 1; last = a
        if a == SAY:
            self._consults.append(((c_idx, c_val), self._know((c_idx, c_val))[1]))
            out = int(self.say.forward((c_idx, c_val)))
            self._cf = (c_idx, c_val)
        elif a == COPY:
            out = self._best(recall)
        elif a == EMIT:
            out = focus[0] if focus else -1
        else:                                                 # JUDGE
            out = int(self.judge.forward((s_idx, s_val)))
            self._js = (s_idx, s_val)
        self._write_registers(c_idx, c_val, s_idx, s_val, explore)
        self._key = (s_idx.tobytes(), a)
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
            wrong = self._out != target
            if self.reliability:
                r = self.rel.setdefault(self._key, [0, 0])
                r[0] += int(not wrong); r[1] += 1
            strategy_error = False
            if wrong and self.hindsight:
                paths = self._search(target)
                if paths is not None:                      # se podía con lo que ya sabe:
                    self._teach_control(paths)             # el error es de estrategia
                    strategy_error = True
            self.blame[0 if strategy_error else 1] += int(wrong)
            if not strategy_error:
                # Se enseña al módulo que respondió, sobre lo que tenía delante al responder.
                # Si respondió la memoria episódica, no hay conocimiento que corregir.
                last = self._path[-1] if self._path else -1
                if self.credit == "percepcion" and wrong and (len(self._path) > 1 or last != SAY):
                    focus0, _, _, reg0, _ = self._start     # versión anterior: a lo percibido
                    self._cf = self._content(focus0, reg=reg0)
                    self.say.forward(self._cf)
                    last = SAY
                elif self.credit in ("eslabon", "estricto") and wrong and self._consults:
                    weak = [i for i, (_, sure) in enumerate(self._consults) if not sure]
                    i = weak[0] if weak else len(self._consults) - 1
                    if last != SAY:
                        last = -1
                    elif i < len(self._consults) - 1:                # eslabón débil intermedio
                        if self.credit == "estricto":
                            last = -1
                        else:
                            self._cf = self._consults[i][0]
                            self.say.forward(self._cf)
                            last = SAY
                # conocimiento consistente: una sola contradicción no lo pisa
                if last == SAY:
                    if not (wrong and self._protected(self._cf, self.say, target)):
                        self.say.teach(target)
                    if self.R:                              # si el conocimiento está protegido, la sospecha
                        self._hindsight(target)             # recae justamente en lo que leyó de los registros
                elif last == JUDGE and not (wrong and self._protected(self._js, self.judge, target)):
                    self.judge.teach(target)
        self._r += float(reward or 0.0)

    # ------------------------------------------------------------------ simulación interna
    def _outcome(self, a, focus, recall, c, s):
        """Qué habría respondido la acción `a` y si sale de conocimiento establecido."""
        if a == SAY:
            return self._know(c)
        if a == COPY:
            return self._best(recall), True
        if a == EMIT:
            return (focus[0] if focus else -1), True
        return self._know(s, self.judge)                   # JUDGE

    def _successors(self, a, focus, recall, c):
        """Focos posibles tras un paso interno (lista vacía si no lleva a nada establecido).
        HABLAR con conocimiento establecido pero inconsistente abre una rama por cada salida
        plausible (probabilidad >= `branch`): hipótesis intermedias, no respuestas."""
        if a == FOLLOW:
            b = self._best(recall)
            return [(b,)] if b >= 0 else []
        out, sure = self._know(c)
        if sure:
            return [(out,)]
        if self.branch <= 0.0:
            return []
        net = self.say
        v = net.layer.forward(*c)
        if v.novel or not len(v.winners) or net.layer.hits[int(v.winners[0])] < 1:
            return []
        p = net.readout.probs(net.readout.potentials(v.winners, net._votes(v)))
        return [(int(o),) for o in np.flatnonzero(p >= self.branch)]

    def _search(self, target: int):
        """Busca, con el conocimiento actual y sin efectos, una secuencia de pasos internos
        que habría llevado al objetivo. Devuelve una lista de caminos
        [(estado, permitidas, acción)], o None si no encuentra ninguno.

        "guiada": mejor primero según la propia política (costo = -log prob, con el piso
        de exploración), con a lo sumo `budget` nodos expandidos. Devuelve el camino más
        probable para el control actual: el que pide menos cambio. Los caminos largos
        cuestan más, así que también prefiere lo más corto (navaja de Occam).
        "amplitud": la búsqueda exhaustiva por niveles de la versión anterior (todos los
        caminos más cortos; exponencial en `max_steps`).

        Sólo cuenta conocimiento establecido (una neurona que ya acertó antes, o un juicio
        seguro): un acierto por casualidad no prueba que el error fue de estrategia."""
        focus0, recall0, cue, reg, probe = self._start
        if self.search == "amplitud":
            return self._search_bfs(target)
        tie = itertools.count()
        heap = [(0.0, next(tie), focus0, recall0, -1, 0, [])]
        n = 0
        while heap and n < self.budget:
            cost, _, focus, recall, last, clock, path = heapq.heappop(heap)
            if focus is None:                              # se sacó una solución: es la más probable
                return [path]
            n += 1; self.expanded += 1
            s = self._state(cue, last, clock, focus, recall, probe, reg)
            allowed = self._allowed(clock)
            q = (1.0 - self.epsilon) * self._policy(*s, allowed) + self.epsilon / len(allowed)
            c = self._content(focus, reg=reg)
            for k, a in enumerate(allowed):
                a = int(a)
                step = path + [(s, allowed, a)]
                cst = cost - float(np.log(q[k]))
                if a in _INTERNAL:
                    for f in self._successors(a, focus, recall, c):
                        heapq.heappush(heap, (cst, next(tie), f, self._recall(f), a, clock + 1, step))
                else:
                    out, sure = self._outcome(a, focus, recall, c, s)
                    if sure and out == target:
                        if self.reliability:
                            ok, n_try = self.rel.get((s[0].tobytes(), a), (0, 0))
                            cst -= self.reliability * float(np.log((ok + 1.0) / (n_try + 2.0)))
                        heapq.heappush(heap, (cst, next(tie), None, None, a, clock, step))
        return None

    def _search_bfs(self, target: int):
        focus0, recall0, cue, reg, probe = self._start
        frontier = [(focus0, recall0, -1, [])]
        for clock in range(self.max_steps + 1):
            nxt = []; found = []
            for focus, recall, last, path in frontier:
                self.expanded += 1
                s = self._state(cue, last, clock, focus, recall, probe, reg)
                allowed = self._allowed(clock)
                c = self._content(focus, reg=reg)
                for a in allowed:
                    a = int(a)
                    step = path + [(s, allowed, a)]
                    if a in _INTERNAL:
                        for f in self._successors(a, focus, recall, c):
                            nxt.append((f, self._recall(f), a, step))
                    else:
                        out, sure = self._outcome(a, focus, recall, c, s)
                        if sure and out == target:
                            found.append(step)
            if found:
                return found
            frontier = nxt
            if not frontier:
                break
        return None

    def _teach_control(self, paths):
        """Corrige el control hacia las secuencias que habrían funcionado (regla delta local,
        repartida entre los caminos)."""
        w = self.lr / len(paths)
        for path in paths:
            for (s_idx, s_val), allowed, a in path:
                p = self._policy(s_idx, s_val, allowed)
                g = -p; g[list(allowed).index(a)] += 1.0
                self.Wc[allowed[:, None], s_idx[None, :]] += w * g[:, None] * s_val[None, :]

    def stats(self) -> dict:
        return dict(neurons=self.say.layer.n, expanded=int(self.expanded),
                    judge_neurons=self.judge.layer.n if self.judge is not None else 0,
                    **{name: int(n) for name, n in zip(ACTION_NAMES, self.decisions)})
