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
  * Compuerta de la memoria de trabajo (`gate`, como PBWM: O'Reilly y Frank, 2006). La
    pregunta llega como una secuencia de ítems; para cada uno el control elige ENFOCAR
    (sumarlo al foco) o SOSTENER (guardarlo aparte). Nadie le dice cuál es el término a
    sostener: la compuerta se aprende con la misma señal que el resto del control. El
    controlador no ve qué se sostiene: sólo si el foco o el recuerdo coinciden con eso
    (detección de coincidencia). Sin `gate`, lo que se sostiene lo da la pregunta
    (`probe`), como en la versión anterior.
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
    objetivo como tercer factor. Qué juzgar: prototipos sobre el estado.
  * Control, compuerta y registros: actor-crítico TD(lambda) con trazas de elegibilidad
    y una sola señal global (el error de predicción de recompensa); es la forma de
    e-prop con recompensa (Bellec et al., 2020) y del rastro sináptico de PBWM.
  * Retrospectiva: ante un error, una búsqueda interna acotada y guiada por la propia
    política (de lo más probable a lo menos, con presupuesto fijo) simula, sin efectos,
    qué secuencia de pasos habría llevado al objetivo con el conocimiento que ya tiene.
    Si la encuentra, el error fue de estrategia: se corrige el control y no se toca el
    conocimiento. Si no, falta conocimiento: se corrige el módulo que respondió (y sólo
    él), sobre lo que tenía delante cuando respondió.
  * Modelo interno (`model`). La búsqueda no ejecuta la red: consulta un modelo hacia
    adelante de los propios módulos de conocimiento (`SelfModel`), aprendido de la copia
    eferente (qué respondió de verdad cada módulo ante cada contenido) y de las
    consecuencias (si esa respuesta resultó correcta). Puede estar desactualizado o no
    cubrir un contenido; entonces la búsqueda no puede contar con él. Con
    `model="perfecto"` la búsqueda ejecuta la red misma (versión anterior).
  * Consolidación y revisión (`consolidate="explicacion"`). Una corrección no se escribe
    en el conocimiento de largo plazo hasta que el mismo objetivo se repite para el mismo
    contenido en la misma consigna (marca y captura sináptica: Frey y Morris, 1997); lo
    episódico (objetivos que cambian cada vez) nunca se consolida. Lo establecido se
    reabre cuando la contradicción consistente viene de la consigna donde se confirmó
    (error de predicción: Sevenster et al., 2013). Si viene de *otra* consigna, las dos
    versiones son hipótesis rivales (causas latentes: Gershman et al., 2017): gana la que
    deja explicada por estrategia la evidencia de la otra, probándolo con la búsqueda
    sobre los episodios que las sostienen; si ninguna explica a la otra, no se cambia.
  * Crédito retrospectivo también en los aciertos (como en «hindsight credit assignment»:
    Harutyunyan et al., 2019, el crédito va a lo que hizo probable el resultado). Si una
    respuesta acertó pero salió de algo que el modelo interno no da por fiable, el acierto
    no prueba la estrategia: si había un camino fiable al mismo objetivo, se enseña ese.
    La búsqueda sólo acepta caminos que un control reactivo puede seguir (en el mismo
    estado, la misma acción).
  * Confianza (`confidence`): el control ve, además, si el modelo interno daría por
    fiable un juicio sobre el estado actual y lo que sabe del foco (metacognición). Así
    «seguir buscando si el juicio no sería fiable» es una regla que generaliza a estados
    que en el entrenamiento casi no aparecieron (cadenas largas en la transitiva).
  * Repetición (`replay`, Dyna: Sutton, 1990). Los errores que no se pudieron explicar
    quedan guardados; cuando el conocimiento o el modelo cambian, se vuelven a buscar y,
    si ahora se explican, se enseña el control sin volver a vivir el episodio (ganancia
    de repetir: Mattar y Daw, 2018).

El controlador ve el *estado* del proceso, no el contenido: decide cómo pensar, no qué
pensar. No ve un reloj absoluto: sabe qué hizo recién (copia eferente) y qué le devuelve
la memoria. Con varias consignas, cada señal llega conjugada con la consigna
(selectividad mixta): por defecto *sólo* conjugada, para que una tarea no pise el control
de otra.
"""
from __future__ import annotations

import heapq
import itertools
from collections import deque

import numpy as np

from .network import BioElasticNet

SAY, COPY, FOLLOW, SPEAK, EMIT, JUDGE, FOCUS, HOLD = range(8)
ACTION_NAMES = ("decir", "copiar", "seguir", "hablar", "emitir", "juzgar", "enfocar", "sostener")
_N_ACTIONS = len(ACTION_NAMES)
_INTERNAL = (FOLLOW, SPEAK)
_GATE = np.array([FOCUS, HOLD])
_NONE = (np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32))


def _recall_events(events, keys):
    """Símbolos que siguieron a `keys` en una lista de eventos (clave, símbolos, cuándo).
    Completa el patrón: ganan los eventos cuya clave comparte más símbolos con `keys`
    (conjunción) y, entre lo recuperado, lo ligado más recientemente llega con fuerza 1."""
    if len(keys) == 0 or not events:
        return _NONE
    keys = set(keys)
    over = [len(k & keys) for k, _, _ in events]
    top = max(over)
    if top == 0:
        return _NONE
    when: dict = {}
    for (k, toks, t), o in zip(events, over):
        if o == top:
            for x in toks:
                when[x] = max(when.get(x, 0), t)
    idx = np.fromiter(when.keys(), dtype=np.int64, count=len(when))
    s = np.fromiter(when.values(), dtype=np.float32, count=len(when))
    return idx, s / s.max()


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
        return _recall_events(self.events, keys)

    def has_successor(self, sym: int) -> bool:
        return any(sym in k for k, _, _ in self.events)


class SelfModel:
    """Modelo interno aprendido de los propios módulos de conocimiento (un modelo hacia
    adelante, como los del control motor: predice la consecuencia de un comando a partir de
    su copia eferente). Por cada (módulo, contenido) guarda qué respondió el módulo la
    última vez que se lo consultó de verdad, y cuántas veces esa respuesta resultó correcta
    o fue culpada de un error. Es tabular, como el modelo de Dyna-Q: no generaliza a
    contenidos que nunca procesó y queda desactualizado cuando el módulo cambia, hasta
    que lo vuelve a ver responder."""

    def __init__(self, sure: float):
        self.T: dict = {}
        self.sure = float(sure)
        self.version = 0                                   # cambia cuando cambia lo que el modelo daría por cierto

    def _is_sure(self, e) -> bool:
        return e[1] >= 1 and e[1] >= self.sure * (e[1] + e[2])

    def observe(self, key, out: int):
        """Copia eferente: el módulo respondió `out` ante este contenido."""
        e = self.T.get(key)
        if e is None or e[0] != out:
            if e is not None and self._is_sure(e):
                self.version += 1
            self.T[key] = [int(out), 0, 0]

    def outcome(self, key, out: int, ok: bool):
        """Consecuencia: esa respuesta resultó correcta (o fue la culpable de un error)."""
        e = self.T.get(key)
        if e is None or e[0] != out:
            return
        was = self._is_sure(e)
        e[1 if ok else 2] += 1
        if self._is_sure(e) != was:
            self.version += 1

    def predict(self, key):
        e = self.T.get(key)
        return (-1, False) if e is None else (e[0], self._is_sure(e))


class _Held(frozenset):
    """Lo sostenido, con la copia eferente de la compuerta que lo guardó: qué ítem de la
    pregunta era (el primero o uno posterior). El control no ve el ítem, sólo eso."""
    origin = 0


def _held(item, origin):
    h = _Held(item)
    h.origin = int(origin)
    return h


def _key(c) -> bytes:
    return np.sort(c[0]).tobytes()


class Thinker:
    def __init__(self, n_symbols: int, *, n_cues: int = 1, memory: bool = False,
                 speak: bool = False, judge: bool = False, copy: bool = True,
                 registers: int = 0, max_steps: int = 4, budget: int = 64,
                 gamma: float = 0.95, lam: float = 0.9, lr: float = 0.5,
                 temperature: float = 1.0, epsilon: float = 0.1, reg_lr: float = 0.5,
                 say_sure: float = 0.7, branch: float = 0.0, reliability: float = 3.0,
                 protect: bool = True, patience: int = 3, hindsight: bool = True,
                 consolidate: str = "explicacion", capture: int = 2,
                 replay: int = 2, replay_size: int = 32, gate: bool = False,
                 model: str = "aprendido", confidence: bool = True, reactive: bool = True,
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
        self.protect = bool(protect)
        # "explicacion": marca y captura + revisión por explicación (ver docstring).
        # "paciencia": la versión anterior (lo establecido se revisa tras `patience`
        # contradicciones iguales seguidas; con registros no se protege).
        self.consolidate = str(consolidate)
        if self.consolidate == "paciencia":
            self.protect = self.protect and self.R == 0
        self.patience = int(patience)
        self.capture = int(capture)            # repeticiones consistentes para consolidar una corrección
        self._contra: dict = {}                # contradicciones consistentes pendientes por situación
        self._tags: dict = {}                  # (módulo, contenido, consigna) -> (objetivo, repeticiones)
        self._claims: dict = {}                # (módulo, contenido) -> {consigna: [objetivo, episodio, contenido, confirmado]}
        self.replay = int(replay)              # episodios repetidos por respuesta (0 = sin repetición)
        self.replay_size = int(replay_size)
        self._pending: dict = {}               # consigna -> deque de episodios sin explicar
        self.kver = 0                          # versión del conocimiento (cambia al aprender)
        self.hindsight = bool(hindsight)
        self.decay = float(decay)              # contracción L1 del control: gana la explicación más simple
        self.search = str(search)              # "guiada" (mejor primero, acotada) o "amplitud" (exhaustiva)
        # a quién se enseña si falta conocimiento: "eslabon" (al primer paso que se apoyó en
        # conocimiento no establecido), "estricto" (ídem, pero si ese paso es intermedio no se
        # enseña nada) o "percepcion" (siempre a lo percibido, como la versión anterior)
        self.credit = str(credit)
        self.gate = bool(gate)
        self.model = str(model)                # "aprendido" (SelfModel) o "perfecto" (ejecuta la red)
        self.confidence = bool(confidence)     # el control ve la confianza del modelo interno
        self.reactive = bool(reactive)         # la búsqueda sólo acepta caminos que un control reactivo puede seguir
        self.M = SelfModel(say_sure)
        self.blame = np.zeros(2, dtype=np.int64)       # errores atribuidos a [estrategia, conocimiento]
        self.expanded = 0                              # nodos expandidos por la búsqueda (costo)
        self.replayed = 0                              # episodios repetidos
        self.revised = np.zeros(3, dtype=np.int64)     # revisiones [misma consigna, ganó la nueva, ganó la vieja]
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
        # compuerta: [es compuerta] ¿foco vacío? ¿sostén vacío? | sostén: nada / 1.er ítem / posterior
        self.o_gate = o; o += 8 if self.gate else 0
        self.o_conf = o; o += 4 if self.confidence else 0   # confianza: ¿juicio fiable? ¿conocimiento fiable?
        self.D_base = o
        self.mixed = mixed if (mixed and self.n_cues > 1) else False
        self._shared = self.mixed != "solo"            # "solo": sólo señales conjugadas con la consigna
        self.D_state = self.D_base * (1 + self.n_cues) if self.mixed else self.D_base
        self.say = BioElasticNet(self.D_content, N, seed=seed, **(say or {}))
        self.Wc = np.zeros((_N_ACTIONS, self.D_state))     # control (y compuerta)
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

    def _finish_state(self, idx, cue):
        if self.mixed:                                     # cada señal, también conjugada con la consigna
            idx = (idx if self._shared else []) + [(1 + cue) * self.D_base + i for i in idx]
        idx = np.array(idx, dtype=np.int64)
        return idx, np.full(len(idx), 1.0 / np.sqrt(len(idx)), dtype=np.float32)

    def _state(self, cue, last, clock, focus, recall, hold, reg=None, over=None):
        reg = self.reg if reg is None else reg
        best = self._best(recall)
        idx = [0, self.o_cue + cue, self.o_last + last + 1,
               self.o_mem + min(len(recall[0]), 2)]                       # nada / uno / varios
        if hold:
            idx.append(self.o_match + int(any(t in hold for t in focus)))
            idx.append(self.o_match + 2 + int(best in hold))
        if self.gate:                                      # qué ítem de la pregunta se sostuvo
            idx.append(self.o_gate + 5 + (0 if not hold else 1 + min(getattr(hold, "origin", 0), 1)))
        for b in range(self.R):
            idx.append(self.o_sreg + 2 * b + int(reg[b]))
        if self.clock:
            idx.append(self.o_clock + clock)
        if self.lookahead and best >= 0:
            idx.append(self.o_look + int(self.fast.has_successor(best)))
        if not self.confidence:
            return self._finish_state(idx, cue)
        # confianza (metacognición): ¿daría por fiable un juicio sobre este estado? ¿y lo que
        # sabe del foco? Sale del modelo interno; el control no ve el juicio ni el contenido.
        # (Sin la segunda señal, la compuerta aprendida en la transitiva generalizó en 2 de
        # 5 semillas en vez de 4 de 5.)
        conf = []
        if self.judge is not None:
            conf.append(self.o_conf + int(self._sim("judge", self._finish_state(idx, cue), over)[1]))
        if focus:
            conf.append(self.o_conf + 2 + int(self._sim("say", self._content(focus, reg=reg), over)[1]))
        return self._finish_state(idx + conf, cue)

    def _base(self, s):
        """El estado sin las señales de confianza: lo que ve el juicio."""
        if not self.confidence:
            return s
        i = s[0] % self.D_base
        idx = s[0][(i < self.o_conf) | (i >= self.o_conf + 4)]
        return idx, np.full(len(idx), 1.0 / np.sqrt(len(idx)), dtype=np.float32)

    def _gate_state(self, cue, last, focus, hold):
        """Estado de la compuerta ante un ítem de la pregunta: la consigna, qué hizo con el
        ítem anterior (copia eferente) y si el foco y el sostén están ocupados. No ve el ítem."""
        idx = [0, self.o_cue + cue, self.o_last + last + 1, self.o_gate,
               self.o_gate + 1 + int(bool(focus)), self.o_gate + 3 + int(bool(hold))]
        return self._finish_state(idx, cue)

    @staticmethod
    def _gated(a, item, focus, hold, i=0):
        if a == FOCUS:
            return tuple(focus) + tuple(t for t in item if t not in focus), hold
        return focus, _held(item, i)

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

        Con la respuesta ya corregida, se evalúa (con el modelo interno) qué habría dicho
        la memoria de largo plazo si cada bit leído hubiera sido el otro. Si sólo uno de los
        dos valores lleva al objetivo, ese es el valor que debió escribirse en el paso
        anterior, y el escritor se corrige hacia él con una regla delta sobre la entrada que
        tenía entonces."""
        if self._prev_wx is None or self._cf is None:
            return
        c_idx, c_val = self._cf
        idx, val = self._prev_wx
        for b in range(self.R):
            outs = []
            for bit in (0, 1):
                ci = c_idx.copy()
                ci[ci == self.o_reg + 2 * b + (1 - bit)] = self.o_reg + 2 * b + bit
                outs.append(self._peek_sim((ci, c_val)))
            ok = [o == target for o in outs]
            if ok[0] != ok[1]:
                want = 1 if ok[1] else 0
                p = self._softmax(self.Wr[b][:, idx] @ val)
                g = -p; g[want] += 1.0
                self.Wr[b][:, idx] += self.reg_lr * g[:, None] * val[None, :]

    # ------------------------------------------------------------------ conocimiento
    def _net(self, mod):
        return self.say if mod == "say" else self.judge

    def _know(self, c, net=None):
        """Qué diría un módulo de conocimiento (`say` por defecto, o el juicio) ante `c`, sin
        efectos, y si es conocimiento establecido: una neurona que ya acertó antes y cuya
        lectura es consistente (probabilidad >= `say_sure`). Es información local del módulo
        (la neurona ganadora y su lectura), no una simulación."""
        net = self.say if net is None else net
        v = net.layer.forward(*c)
        V = net.readout.potentials(v.winners, net._votes(v))
        out = int(np.argmax(V))
        if v.novel or not len(v.winners) or net.layer.hits[int(v.winners[0])] < 1:
            return out, False
        return out, bool(net.readout.probs(V)[out] >= self.say_sure)

    def _sim(self, mod, c, over=None):
        """Lo que la búsqueda cree que respondería el módulo `mod` ante `c`, y si lo da por
        establecido. Con el modelo aprendido sale de `SelfModel`; con "perfecto", de ejecutar
        el módulo. `over`: hipótesis {(módulo, contenido): respuesta} que se dan por ciertas."""
        if over:
            t = over.get((mod, _key(c)))
            if t is not None:
                return t, True
        if self.model == "perfecto":
            return self._know(c, self._net(mod))
        return self.M.predict((mod, _key(c)))

    def _reliable(self) -> bool:
        """¿La respuesta final salió de algo que el modelo interno da por fiable? (La memoria
        episódica y el foco se dan por fiables: no son conocimiento que pueda estar mal.)"""
        a = self._path[-1] if self._path else -1
        if a == SAY:
            return self.M.predict(("say", _key(self._cf)))[1]
        if a == JUDGE:
            return self.M.predict(("judge", _key(self._js)))[1]
        return True

    def _m_outcome(self, mod, c, ok):
        """El modelo interno registra la consecuencia de lo que el módulo respondió."""
        k = (mod, _key(c))
        self.M.outcome(k, self.M.predict(k)[0], ok)

    def _observe_self(self, mod, c, out):
        """Copia eferente hacia el modelo interno (sólo mientras aprende)."""
        if self.plastic:
            self.M.observe((mod, _key(c)), out)

    def _peek_sim(self, c):
        if self.model == "perfecto":
            return self.say.peek(c)
        return self.M.predict(("say", _key(c)))[0]

    def _protected(self, c, net, target):
        """Versión anterior ("paciencia"): ¿es conocimiento consistente establecido para
        *esta misma* situación? No se pisa salvo que el mismo objetivo alternativo se repita
        `patience` veces seguidas. Lo generalizado desde un vecino siempre se corrige."""
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

    def _labilize(self, mod, c):
        """Reconsolidación (Nader et al., 2000): el recuerdo reactivado vuelve a ser plástico.
        Las lecturas de las neuronas que responden a `c` pierden la consolidación acumulada
        (el contador que frena sus cambios en `Readout`), para que la revisión pueda
        escribirse; el resto del conocimiento no se toca."""
        net = self._net(mod)
        v = net.layer.forward(*c)
        if len(v.winners):
            net.readout.count[v.winners] = 0.0

    def _teach(self, mod, c, target, refresh=True):
        """Escribe en el módulo de conocimiento (y anota que el conocimiento cambió)."""
        net = self._net(mod)
        if refresh:
            net.forward(c)
        net.teach(int(target))
        self.kver += 1

    def _confirm(self, mod, c, out, cue, episode):
        """La respuesta del módulo resultó correcta en esta consigna: queda registrada como
        lo que esta consigna sostiene para ese contenido, y se borran las marcas de
        contradicción pendientes en esta consigna (lo confirmado se afianza)."""
        if self.consolidate != "explicacion" or not len(c[0]):
            return
        k = (mod, _key(c))
        self._tags.pop((k, cue), None)
        self._claims.setdefault(k, {})[cue] = [int(out), episode, c, True]

    def _correct(self, mod, c, target, cue, episode, refresh):
        """Corrección de conocimiento ante un error que no se explicó por estrategia."""
        net = self._net(mod)
        if self.consolidate != "explicacion":
            if not (self.protect and self._protected(c, net, target)):
                self._teach(mod, c, target, refresh)
            return None
        if not len(c[0]):                                  # sin presináptica no hay plasticidad
            return None
        k = (mod, _key(c))
        alt, n = self._tags.get((k, cue), (None, 0))
        n = n + 1 if alt == target else 1
        if n < self.capture:                               # marca: todavía no se consolida
            self._tags[(k, cue)] = (target, n)
            return None
        self._tags.pop((k, cue), None)
        # una contradicción consistente sí es evidencia contra lo que el módulo responde; un
        # error suelto en el conocimiento compartido puede ser episódico y no cuenta
        self._m_outcome(mod, c, False)
        claims = self._claims.setdefault(k, {})
        out, sure = self._know(c, net)
        mine = claims.get(cue)
        others = [q for q, cl in claims.items() if q != cue and cl[3] and cl[0] == out]
        claims[cue] = [int(target), episode, c, False]
        if not (self.protect and sure) or not others or (mine is not None and mine[3] and mine[0] == out):
            if self.protect and sure:                      # se reabre lo establecido
                self.revised[0] += 1
                self._labilize(mod, c)
            self._teach(mod, c, target, refresh)
            return None
        return (mod, cue, others[0])                       # dos consignas sostienen cosas distintas

    def _adjudicate(self, mod, new, old):
        """Dos consignas sostienen respuestas distintas para los mismos contenidos. Cada una
        es una hipótesis sobre el conocimiento compartido: gana la que, dada por cierta,
        deja explicada por estrategia (con la búsqueda) más evidencia de la otra. Si gana la
        nueva se reescribe el conocimiento; en cualquier caso se enseña el control con los
        caminos que explican a la perdedora."""
        S = [(k, cl) for k, cl in self._claims.items()
             if k[0] == mod and new in cl and old in cl and cl[new][0] != cl[old][0]]
        if not S:
            return

        def explain(win, lose):
            over = {k: cl[win][0] for k, cl in S}
            paths = []
            for _, cl in S:
                p = self._search(cl[lose][0], start=cl[lose][1], over=over)
                if p is not None:
                    paths.append(p[0])
            return paths

        p_new, p_old = explain(new, old), explain(old, new)
        if len(p_new) > len(p_old):
            self.revised[1] += 1
            for k, cl in S:
                c, t = cl[new][2], cl[new][0]
                self._labilize(mod, c)
                for _ in range(6):                          # reconsolidar hasta que quede establecido
                    out, sure = self._know(c, self._net(mod))
                    if out == t and sure:
                        break
                    self._teach(mod, c, t)
                cl[new][3] = True
                self._remember(cl[old][1], cl[old][0])     # la evidencia de la perdedora: a repetir
                del cl[old]
            for p in p_new:
                self._teach_control([p])
        elif len(p_old) > len(p_new):
            self.revised[2] += 1
            for _, cl in S:
                self._remember(cl[new][1], cl[new][0])
                del cl[new]
            for p in p_old:
                self._teach_control([p])

    # ------------------------------------------------------------------ percibir y responder
    def observe(self, tokens=None):
        """Percibe sin responder. `None` es un silencio."""
        if self.fast is not None:
            self.fast.bind(self._tok(tokens))

    def answer(self, tokens, cue: int = 0, explore: bool = False, probe=None) -> int:
        """Percibe la pregunta, piensa los pasos que decida y responde. Devuelve el símbolo
        (-1 = nada). `probe`: un segundo término de la pregunta (p. ej. Y en «¿X > Y?»).
        Sin compuerta, es lo que se sostiene; con compuerta, es un ítem más de la pregunta
        y la red decide qué hacer con él."""
        items = [self._tok(tokens)]
        if probe is not None and self.gate:
            items.append(self._tok(probe))
        given = frozenset() if self.gate else frozenset(self._tok(probe))
        self._prev_wx = self._wx
        self._q = (tuple(items), cue, self.reg.copy(), given)
        self._path = []; self._consults = []
        last = -1
        if self.gate:                                         # compuerta de la memoria de trabajo
            focus, hold = (), frozenset()
            for i, it in enumerate(items):
                s_idx, s_val = self._gate_state(cue, last, focus, hold)
                a = self._decide(s_idx, s_val, _GATE, explore)
                self.decisions[a] += 1
                self._path.append(a)
                focus, hold = self._gated(a, it, focus, hold, i)
                last = a
        else:
            focus, hold = items[0], given
        recall = self._recall(focus)
        clock = 0
        while True:
            s_idx, s_val = self._state(cue, last, clock, focus, recall, hold)
            a = self._decide(s_idx, s_val, self._allowed(clock), explore)
            self.decisions[a] += 1
            self._path.append(a)
            c_idx, c_val = self._content(focus)
            if a == FOLLOW:                                   # el recuerdo pasa al foco
                b = self._best(recall)
                focus = (b,) if b >= 0 else ()
            elif a == SPEAK:                                  # habla interna
                self._consults.append(((c_idx, c_val), self._know((c_idx, c_val))[1]))
                o = int(self.say.forward((c_idx, c_val)))
                self._observe_self("say", (c_idx, c_val), o)
                focus = (o,)
            else:
                break
            recall = self._recall(focus)
            clock += 1; last = a
        if a == SAY:
            self._consults.append(((c_idx, c_val), self._know((c_idx, c_val))[1]))
            out = int(self.say.forward((c_idx, c_val)))
            self._observe_self("say", (c_idx, c_val), out)
            self._cf = (c_idx, c_val)
        elif a == COPY:
            out = self._best(recall)
        elif a == EMIT:
            out = focus[0] if focus else -1
        else:                                                 # JUDGE
            js = self._base((s_idx, s_val))
            out = int(self.judge.forward(js))
            self._observe_self("judge", js, out)
            self._js = js
        self._write_registers(c_idx, c_val, s_idx, s_val, explore)
        self._key = (s_idx.tobytes(), a)
        self._out = out
        return out

    def _episode(self):
        """La pregunta actual como episodio repetible: (ítems, consigna, registros, sostén
        dado, memoria del episodio)."""
        items, cue, reg, given = self._q
        events = tuple(self.fast.events) if self.fast is not None else ()
        return (items, cue, reg, given, events)

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
            cue = self._q[1]
            ep = self._episode() if self.consolidate == "explicacion" or self.replay else None
            if self.reliability:
                r = self.rel.setdefault(self._key, [0, 0])
                r[0] += int(not wrong); r[1] += 1
            strategy_error = False
            if wrong and self.hindsight:
                paths = self._search(target)
                if paths is not None:                      # se podía con lo que ya sabe:
                    self._teach_control(paths)             # el error es de estrategia
                    strategy_error = True
            elif not wrong and self.hindsight and self.model != "perfecto" and not self._reliable():
                # acertó, pero con una respuesta que el modelo interno no da por fiable: el
                # acierto no prueba la estrategia (pudo ser suerte). Si había un camino fiable
                # al mismo objetivo, el crédito va a ese camino.
                paths = self._search(target)
                if paths is not None and [a for _, _, a in paths[0]] != list(self._path):
                    self._teach_control(paths)
            self.blame[0 if strategy_error else 1] += int(wrong)
            if strategy_error and self._path and self._path[-1] == JUDGE:
                # el juicio es sobre este estado: si falló, ese estado no basta para juzgar
                # (aunque otro camino sí habría servido)
                self._m_outcome("judge", self._js, False)
            conflict = None
            if not strategy_error:
                # Se corrige al módulo que respondió, sobre lo que tenía delante al responder.
                # Si respondió la memoria episódica, no hay conocimiento que corregir.
                last = self._path[-1] if self._path else -1
                refresh = False
                n_think = sum(1 for a in self._path if a not in (FOCUS, HOLD))
                if self.credit == "percepcion" and wrong and (n_think > 1 or last != SAY):
                    focus0 = self._q[0][0] if not self.gate else ()
                    self._cf = self._content(focus0, reg=self._q[2])
                    last = SAY; refresh = True
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
                            last = SAY; refresh = True
                if last == SAY:
                    if wrong:
                        conflict = self._correct("say", self._cf, target, cue, ep, refresh)
                    else:
                        self.say.teach(target)
                        self.kver += 1
                        self._confirm("say", self._cf, target, cue, ep)
                    if self.R:                              # si el conocimiento está protegido, la sospecha
                        self._hindsight(target)             # recae justamente en lo que leyó de los registros
                elif last == JUDGE:
                    if wrong:
                        self._m_outcome("judge", self._js, False)
                        conflict = self._correct("judge", self._js, target, cue, ep, False)
                    else:
                        self.judge.teach(target)
                        self.kver += 1
                        self._confirm("judge", self._js, target, cue, ep)
                if wrong:                                   # error sin explicar: queda para repetir
                    self._remember(ep, target)
            if not wrong:                                   # la cadena acertó: sus eslabones también
                for c, _ in self._consults:
                    self._m_outcome("say", c, True)
                if self._path and self._path[-1] == JUDGE:
                    self._m_outcome("judge", self._js, True)
            if conflict is not None:
                self._adjudicate(*conflict)
            if self.replay:
                self._replay()
        self._r += float(reward or 0.0)

    # ------------------------------------------------------------------ repetición (Dyna)
    def _version(self):
        return self.kver if self.model == "perfecto" else self.M.version

    def _remember(self, ep, target):
        """Guarda un error sin explicar para repetirlo cuando el conocimiento cambie."""
        if not self.replay:
            return
        q = self._pending.setdefault(ep[1], deque(maxlen=self.replay_size))
        q.append([ep, target, self._version(), self.replayed])   # episodio, objetivo, versión, última vez

    def _replay(self):
        """Repetición prioritaria (Dyna; ganancia según Mattar y Daw): vuelve a buscar, con
        el conocimiento de ahora, errores pasados que no se pudieron explicar, sólo si el
        conocimiento o el modelo cambiaron desde la última vez (si no, no hay nada que
        ganar). Si ahora hay camino, se enseña el control (planificación sin experiencia
        nueva) y se lo sigue repitiendo hasta que el control lo siga; entonces se olvida.
        Sólo se guardan errores que en su momento no tenían explicación: repetir también los
        explicados en el momento desestabilizó el control (transitiva en secuencia cayó a
        0.0-0.69 en 3 de 10 semillas)."""
        ver = self._version()
        todo = [(e[3], i, q, e) for i, q in enumerate(self._pending.values()) for e in q
                if e[2] is None or e[2] != ver]
        if not todo:
            return
        todo.sort(key=lambda x: (x[0], x[1]))               # el que hace más que no se repite, primero
        for _, _, q, e in todo[:self.replay]:
            ep, target = e[0], e[1]
            self.replayed += 1
            e[3] = self.replayed
            paths = self._search(target, start=ep)
            if paths is None:
                e[2] = ver
                continue
            greedy = all(int(a) == int(allowed[np.argmax(self._policy(*s, allowed))])
                         for s, allowed, a in paths[0])
            if greedy:                                       # nada que ganar: ya lo haría así
                for i, x in enumerate(q):
                    if x is e:
                        del q[i]
                        break
            else:                                            # lo nuevo todavía no llegó al control:
                self._teach_control(paths)                   # se repite hasta que lo siga
                e[2] = None

    # ------------------------------------------------------------------ simulación interna
    def _outcome(self, a, focus, recall, c, s, over=None):
        """Qué habría respondido la acción `a` y si sale de conocimiento establecido."""
        if a == SAY:
            return self._sim("say", c, over)
        if a == COPY:
            return self._best(recall), True
        if a == EMIT:
            return (focus[0] if focus else -1), True
        return self._sim("judge", self._base(s), over)     # JUDGE

    def _successors(self, a, focus, recall, c, over=None):
        """Focos posibles tras un paso interno (lista vacía si no lleva a nada establecido).
        HABLAR con conocimiento establecido pero inconsistente abre una rama por cada salida
        plausible (probabilidad >= `branch`, sólo con el modelo perfecto)."""
        if a == FOLLOW:
            b = self._best(recall)
            return [(b,)] if b >= 0 else []
        out, sure = self._sim("say", c, over)
        if sure:
            return [(out,)]
        if self.branch <= 0.0 or self.model != "perfecto":
            return []
        net = self.say
        v = net.layer.forward(*c)
        if v.novel or not len(v.winners) or net.layer.hits[int(v.winners[0])] < 1:
            return []
        p = net.readout.probs(net.readout.potentials(v.winners, net._votes(v)))
        return [(int(o),) for o in np.flatnonzero(p >= self.branch)]

    def _search(self, target: int, start=None, over=None):
        """Busca, con el conocimiento actual (según el modelo interno) y sin efectos, una
        secuencia de pasos (compuerta incluida) que habría llevado al objetivo. Devuelve
        una lista de caminos [(estado, permitidas, acción)], o None si no encuentra ninguno.

        "guiada": mejor primero según la propia política (costo = -log prob, con el piso
        de exploración), con a lo sumo `budget` nodos expandidos. Devuelve el camino más
        probable para el control actual: el que pide menos cambio. Los caminos largos
        cuestan más, así que también prefiere lo más corto (navaja de Occam).
        "amplitud": la búsqueda exhaustiva por niveles de la versión anterior.

        `start`: un episodio guardado (si no, la última pregunta). `over`: hipótesis sobre
        el conocimiento que se dan por ciertas (ver `_adjudicate`).

        Sólo cuenta conocimiento establecido: un acierto por casualidad no prueba que el
        error fue de estrategia."""
        items, cue, reg, given, events = self._episode() if start is None else start
        if self.fast is not None:
            rec = lambda f: _recall_events(events, f)  # noqa: E731
        else:
            rec = lambda f: _NONE  # noqa: E731
        if self.search == "amplitud":
            return self._search_bfs(target, items, cue, reg, given, rec, over)
        tie = itertools.count()
        # nodo: (costo, desempate, tipo, foco, sostén, recuerdo, última, reloj/ítem, camino)
        if self.gate:
            heap = [(0.0, next(tie), "g", (), frozenset(), None, -1, 0, [])]
        else:
            f0 = items[0]
            heap = [(0.0, next(tie), "t", f0, given, rec(f0), -1, 0, [])]
        n = 0
        while heap and n < self.budget:
            cost, _, kind, focus, hold, recall, last, clock, path = heapq.heappop(heap)
            if kind == "s":                                # se sacó una solución: es la más probable
                return [path]
            n += 1; self.expanded += 1
            if kind == "g":                                # compuerta del ítem `clock`
                s = self._gate_state(cue, last, focus, hold)
                q = (1.0 - self.epsilon) * self._policy(*s, _GATE) + self.epsilon / 2
                for k, a in enumerate(_GATE):
                    a = int(a)
                    f, h = self._gated(a, items[clock], focus, hold, clock)
                    step = path + [(s, _GATE, a)]
                    cst = cost - float(np.log(q[k]))
                    if clock + 1 < len(items):
                        heapq.heappush(heap, (cst, next(tie), "g", f, h, None, a, clock + 1, step))
                    else:
                        heapq.heappush(heap, (cst, next(tie), "t", f, h, rec(f), a, 0, step))
                continue
            s = self._state(cue, last, clock, focus, recall, hold, reg, over)
            allowed = self._allowed(clock)
            q = (1.0 - self.epsilon) * self._policy(*s, allowed) + self.epsilon / len(allowed)
            c = self._content(focus, reg=reg)
            forced = self._forced(s, path)
            for k, a in enumerate(allowed):
                a = int(a)
                if forced is not None and a != forced:
                    continue
                step = path + [(s, allowed, a)]
                cst = cost - float(np.log(q[k]))
                if a in _INTERNAL:
                    for f in self._successors(a, focus, recall, c, over):
                        heapq.heappush(heap, (cst, next(tie), "t", f, hold, rec(f), a, clock + 1, step))
                else:
                    out, sure = self._outcome(a, focus, recall, c, s, over)
                    if sure and out == target:
                        if self.reliability:
                            ok, n_try = self.rel.get((s[0].tobytes(), a), (0, 0))
                            cst -= self.reliability * float(np.log((ok + 1.0) / (n_try + 2.0)))
                        heapq.heappush(heap, (cst, next(tie), "s", focus, hold, None, a, clock, step))
        return None

    def _forced(self, s, path):
        """Si el camino ya pasó por este mismo estado, la acción tiene que ser la misma: el
        control es reactivo (una función del estado), así que una explicación que en el
        mismo estado hace dos cosas distintas no la puede aprender. Sin esta restricción,
        la composición en orden adverso encontraba HABLAR, HABLAR, EMITIR, que el control
        no distingue de HABLAR, EMITIR (mismo estado tras cada HABLAR) y no convergía."""
        if not self.reactive:
            return None
        key = s[0].tobytes()
        for (si, _), _, a in path:
            if si.tobytes() == key:
                return int(a)
        return None

    def _search_bfs(self, target, items, cue, reg, given, rec, over):
        if self.gate:                                       # la compuerta, enumerada
            roots = [((), frozenset(), -1, [])]
            for i, it in enumerate(items):
                nxt = []
                for focus, hold, last, path in roots:
                    s = self._gate_state(cue, last, focus, hold)
                    for a in _GATE:
                        f, h = self._gated(int(a), it, focus, hold, i)
                        nxt.append((f, h, int(a), path + [(s, _GATE, int(a))]))
                roots = nxt
        else:
            roots = [(items[0], given, -1, [])]
        frontier = [(f, h, rec(f), last, path) for f, h, last, path in roots]
        for clock in range(self.max_steps + 1):
            nxt = []; found = []
            for focus, hold, recall, last, path in frontier:
                self.expanded += 1
                s = self._state(cue, last, clock, focus, recall, hold, reg, over)
                allowed = self._allowed(clock)
                c = self._content(focus, reg=reg)
                forced = self._forced(s, path)
                for a in allowed:
                    a = int(a)
                    if forced is not None and a != forced:
                        continue
                    step = path + [(s, allowed, a)]
                    if a in _INTERNAL:
                        for f in self._successors(a, focus, recall, c, over):
                            nxt.append((f, hold, rec(f), a, step))
                    else:
                        out, sure = self._outcome(a, focus, recall, c, s, over)
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
        return dict(neurons=self.say.layer.n, expanded=int(self.expanded), replayed=int(self.replayed),
                    judge_neurons=self.judge.layer.n if self.judge is not None else 0,
                    revised=self.revised.tolist(),
                    **{name: int(n) for name, n in zip(ACTION_NAMES, self.decisions)})
