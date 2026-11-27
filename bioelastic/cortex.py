"""Cortex: jerarquía de áreas sensoriales + áreas de asociación que votan la decisión.

    entrada -> área 1 -> área 2 -> ... -> área L          (rasgos locales, no supervisado)
                  \\          \\               \\
                   +----------+---------------+--> áreas de asociación -> decisión

Profundidad y anchura son elásticas y configurables:
  * `areas`: lista de niveles sensoriales (`FeatureArea`); cada uno ve un campo
    receptivo mayor que el anterior.
  * `heads`: áreas de asociación (`BioElasticNet`). Cada una elige su fuente
    ("top", "all", índice de área o "input") y su memoria de entrada (retardos o traza
    lenta con su propia constante de tiempo). Todas votan sumando potenciales sobre las
    mismas neuronas de salida, y todas reciben el mismo error/recompensa (el
    neuromodulador es global). Es el análogo local de las conexiones residuales.

No hay backprop: las áreas sensoriales aprenden sin supervisión (o con un error local
propio, ver `FeatureArea`) y las de asociación con la regla de tres factores. Agregar un
área nueva no borra lo que saben las demás.

Decisión: por defecto (v0.2) se suman los potenciales de todos los votantes (áreas de
asociación, proyección directa y, si las hay, lectores locales de las áreas sensoriales).
Con `arbiter > 0` cada votante aprende con su propio error y un árbitro pondera sus
log-probabilidades, V = sum_i beta_i log p_i, con pesos que aprenden online del error
global: d beta_i = arbiter * (log p_i[objetivo] - E_p[log p_i]). Es un control de ganancia
por vía: un votante que no aporta pierde voz en vez de arrastrar la decisión.

Homeostasis de las salidas (`homeostasis` > 0, opcional; por defecto apagada): cada vez
que la red aprende de una decisión (`teach` o `reinforce`), el umbral de la salida elegida
sube `homeostasis` y el de las demás baja homeostasis/C (suma cero), como el `theta_plus`
de las capas ocultas. Una salida que acapara decisiones se vuelve más difícil de disparar
y una que nunca gana termina por ganar alguna. Contra el colapso de símbolos cuando la red
aprende sólo con recompensa (en la integración con el Thinker, desde cero, la pureza de la
percepción subió de 0.21 a 0.58 y de 0.62 a 0.85 en la identidad, 3 semillas). No cambia
cómo se reparte el crédito. Con etiquetas no hace falta y no se usa.

Metacognición y neuromodulación (opcionales; por defecto apagadas, sin efecto en nada):
  señales   `confidence()` (probabilidad de la decisión; calibrada si hay `metacog`),
            `conflict()` (desacuerdo entre votantes sobre la decisión: 1 - la probabilidad
            media que cada votante le da a la salida elegida, como el conflicto de
            respuesta del cingulado anterior: Yeung, Botvinick y Cohen, 2004),
            `surprise(correct)` (-log de la probabilidad del desenlace según la propia
            confianza), `familiarity()` (parecido de la entrada al prototipo más cercano de
            las áreas de asociación: novedad vs error en lo conocido) y `signals()`. Se calculan del último pase, a pedido.
  metacog   dict(lr=..., new=10): dos calibradores online (regresión logística). Uno aprende
            P(acierto) de cada desenlace conocido (etiqueta o recompensa) a partir de la
            confianza cruda, el conflicto, la familiaridad y la experiencia: `confidence()`.
            El otro aprende P(es algo que casi no se me enseñó) de la propia historia de
            enseñanza (clases con menos de `new` enseñanzas) a partir de familiaridad,
            experiencia y evidencia: `novelty()`. Sin `metacog`, `confidence()` es la
            probabilidad cruda y `novelty()` es 1 - familiaridad.
  hypercorrection   g > 0: un error se corrige con la tasa de las lecturas multiplicada
            por 1 + g * confianza en lo que se decidió (efecto de hipercorrección:
            Butterfield y Metcalfe, 2001). `hypercorrection_mode="plano"` es el control:
            la misma ganancia media en los errores, sin ponderar por confianza.
  neuromod  dict: incertidumbre esperada e inesperada (Yu y Dayan, 2005). La sorpresa de
            cada desenlace se promedia lento (ACh: incertidumbre esperada, el ruido de
            fondo) y rápido; NE = max(0, (rápida - lenta) / desvío lento) es la sorpresa
            que excede lo esperado (un cambio en el mundo). Efectos: la tasa de las lecturas
            se multiplica por (1 + ne_gain * NE) / (1 + ach_gain * ACh); si NE > ne_reset se
            libera parte de la consolidación de las lecturas (`release`) y las áreas de
            asociación crecen ante cualquier error mientras dure (neurogénesis).

Retroalimentación predictiva (opcional; por defecto la vía es sólo hacia adelante): si las
áreas tienen `predictive` (ver `FeatureArea`), cada una predice su entrada (el código del
área de abajo, o los píxeles) y aprende de su residuo. `feedback=dict(...)` agrega:
  iters    bucle de inferencia tipo Rao-Ballard de `iters` iteraciones (0 = un solo pase;
           la predicción sólo se usa para aprender y medir el error). En cada iteración, con
           las predicciones de la iteración anterior, las áreas se recalculan de abajo hacia
           arriba con
             representación del nivel  r = max(abajo * (1 + gain * Q) + context * Q, 0)
                                                                   (contexto, se lee)
             lo que sube al área       u = max(r + up * (r - Q), 0)      (código + residuo)
           donde Q es la predicción del área de arriba. up = 0: sube el código; up = 1:
           código + residuo; up grande: casi sólo el residuo (el área es invariante a la
           escala). context > 0 completa el código de abajo con lo que predice el de arriba;
           gain > 0 sólo amplifica la actividad de abajo que la de arriba predice (ganancia
           multiplicativa, no crea actividad donde no la hay).
  source   qué sorpresa modula la plasticidad de cada área (si su `predictive` lo pide):
           "input" su propio error (lo que no explica de su entrada), "code" el error con que
           el área de arriba predice su código (la de más arriba no se modula), "shuffle"
           control: un error propio pasado elegido al azar (misma distribución, sin relación
           con la imagen actual).
Los pesos generativos de cada área aprenden a predecir la entrada ascendente *antes* del
contexto (no su propia predicción).

Paralelismo (`workers` > 1): las áreas de asociación que leen la entrada cruda no
dependen de las áreas sensoriales, así que su pase y su aprendizaje corren en otro hilo
mientras la jerarquía codifica o aprende. Los resultados no cambian: cada hilo toca sólo
su propio estado y las sumas se hacen en el mismo orden. La ganancia está limitada por
el GIL (sólo las operaciones NumPy grandes lo liberan).
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from .area import FeatureArea
from .metacog import Calibrator, SurpriseTracker
from .network import BioElasticNet
from .readout import Readout


class Cortex:
    def __init__(self, input_shape, n_outputs: int, *, areas=(), heads=({},), projection=None,
                 replay: float = 0.0, rehearse: float = 0.0, replay_focus: str = "uniform",
                 rehearse_focus: str = "uniform", confusion_p: float = 0.05,
                 joint: bool = True, temperature: float = 0.25, arbiter: float = 0.0,
                 recall: str = "off", recall_k: int = 8, recall_margin: float = 0.2,
                 recall_gain: float = 1.0, recall_tau: float = 0.05, recall_source: str = "episodes",
                 stored_labels: bool = False, slow_error: str = "joint",
                 schema: str = "off", schema_fit: float = 0.5, schema_boost: float = 4.0,
                 schema_misfit_lr: float = 1.0, schema_beta: float = 0.02, schema_known: int = 20,
                 schema_frozen: int = 0,
                 replay_mode: str = "passive", spacing: str = "none", space_base: float = 200.0,
                 space_factor: float = 2.0, active_margin: float = 0.0, fail_weight: float = 4.0,
                 sleep_every: int = 0, sleep_trigger: str = "steps", sleep_load: int = 200,
                 sleep_replays: int = 0, sleep_micro: int = 0, sleep_downscale: float = 0.0,
                 sleep_scope: str = "proj", sleep_prune: float = 0.0, sleep_release: float = 0.0,
                 sleep_keep: int = 10, replay_contrast: bool = False, sleep_focus: str = "uniform",
                 sleep_release_check: str = "margin", cause_slow: bool = True,
                 seed: int = 0, workers: int = 1, homeostasis: float = 0.0,
                 metacog: dict | None = None, hypercorrection: float = 0.0,
                 hypercorrection_mode: str = "confianza", neuromod: dict | None = None,
                 feedback: dict | None = None):
        self._config = dict(input_shape=input_shape, n_outputs=int(n_outputs),
                            areas=[dict(a) for a in areas], heads=[],
                            projection=projection, replay=float(replay),
                            rehearse=float(rehearse), replay_focus=str(replay_focus),
                            rehearse_focus=str(rehearse_focus), confusion_p=float(confusion_p),
                            joint=bool(joint), temperature=float(temperature),
                            arbiter=float(arbiter),
                            recall=str(recall), recall_k=int(recall_k), recall_margin=float(recall_margin),
                            recall_gain=float(recall_gain), recall_tau=float(recall_tau),
                            recall_source=str(recall_source), stored_labels=bool(stored_labels),
                            slow_error=str(slow_error),
                            schema=str(schema), schema_fit=float(schema_fit),
                            schema_boost=float(schema_boost), schema_misfit_lr=float(schema_misfit_lr),
                            schema_beta=float(schema_beta), schema_known=int(schema_known),
                            schema_frozen=int(schema_frozen),
                            replay_mode=str(replay_mode), spacing=str(spacing), space_base=float(space_base),
                            space_factor=float(space_factor), active_margin=float(active_margin),
                            fail_weight=float(fail_weight),
                            sleep_every=int(sleep_every), sleep_trigger=str(sleep_trigger),
                            sleep_load=int(sleep_load), sleep_replays=int(sleep_replays),
                            sleep_micro=int(sleep_micro), sleep_downscale=float(sleep_downscale),
                            sleep_scope=str(sleep_scope), sleep_prune=float(sleep_prune),
                            sleep_release=float(sleep_release), sleep_keep=int(sleep_keep),
                            replay_contrast=bool(replay_contrast), sleep_focus=str(sleep_focus),
                            sleep_release_check=str(sleep_release_check), cause_slow=bool(cause_slow),
                            seed=int(seed))
        self.arbiter = float(arbiter)
        self.beta = None                                      # pesos del árbitro (uno por votante)
        self._L = None
        self._zs = []
        self._codes = []
        self.homeostasis = float(homeostasis)
        if self.homeostasis > 0.0:
            self._config["homeostasis"] = self.homeostasis
        # metacognición y neuromodulación (sólo se guardan en la configuración si se usan)
        self.metacog = dict(metacog) if metacog else None
        self.hypercorrection = float(hypercorrection)
        self.hypercorrection_mode = str(hypercorrection_mode)
        self.neuromod = None
        if neuromod:
            self.neuromod = dict(dict(slow=0.002, fast=0.05, ne_gain=0.5, ach_gain=0.0, ne_cap=4.0,
                                      ne_reset=3.0, release=0.5, grow=True, warmup=200, var0=1.0),
                                 **neuromod)
        for k, v in (("metacog", self.metacog), ("hypercorrection", self.hypercorrection or None),
                     ("neuromod", self.neuromod)):
            if v:
                self._config[k] = v
        if self.hypercorrection and self.hypercorrection_mode != "confianza":
            self._config["hypercorrection_mode"] = self.hypercorrection_mode
        self._cal = Calibrator(n_extra=3, lr=float(self.metacog.get("lr", 0.05))) if self.metacog else None
        self._nov = Calibrator(n_extra=3, lr=float(self.metacog.get("lr", 0.05))) if self.metacog else None
        if self._nov is not None:
            self._nov.w[:] = 0.0                              # sin opinión previa sobre la novedad
        self._taught = np.zeros(int(n_outputs), dtype=np.int64)   # enseñanzas por clase (historia propia)
        self._err_conf = [0.0, 0]                             # confianza media en los errores (control plano)
        nm = self.neuromod or {}
        self._surprise = SurpriseTracker(nm.get("slow", 0.002), nm.get("fast", 0.05),
                                         nm.get("warmup", 200), nm.get("var0", 1.0)) if nm else None
        self.lr_gain = 1.0
        self.releases = 0
        self._soft = None                                     # objetivo blando del `teach` en curso
        self._votes_last = []
        self.fb = None
        if feedback:
            spec = dict(iters=0, up=0.0, context=0.0, gain=0.0, source="input")
            unknown = set(feedback) - set(spec)
            if unknown:
                raise ValueError(f"opciones de `feedback` desconocidas: {sorted(unknown)}")
            spec.update(feedback)
            if spec["source"] not in ("input", "code", "shuffle"):
                raise ValueError(f"source desconocido: {spec['source']}")
            self.fb = spec
            self._config["feedback"] = dict(spec)
            self._fb_rng = np.random.default_rng(seed + 7919)
            self._pe_hist: list = []
        self.is_sheet = not np.isscalar(input_shape)
        self.input_shape = tuple(int(v) for v in input_shape) if self.is_sheet else int(input_shape)
        self.C = int(n_outputs)
        self.temperature = float(temperature)
        self.replay = float(replay); self._replay_debt = 0.0; self.replays = 0
        self.rehearse = float(rehearse); self._rehearse_debt = 0.0; self.rehearsals = 0
        if replay_focus not in ("uniform", "absent", "confusion"):
            raise ValueError(f"replay_focus desconocido: {replay_focus}")
        if rehearse_focus not in ("uniform", "confusion"):
            raise ValueError(f"rehearse_focus desconocido: {rehearse_focus}")
        self.replay_focus = replay_focus
        self.rehearse_focus = rehearse_focus
        self.confusion_p = float(confusion_p)
        self.directed = 0                                     # repasos y ensayos dirigidos a la confusión
        self.joint = bool(joint)
        # Dos sistemas explícitos (ver `forward` y `_recall_vote`).
        if recall not in ("off", "gate", "always", "fallback"):
            raise ValueError(f"recall desconocido: {recall}")
        if recall_source not in ("episodes", "readout"):
            raise ValueError(f"recall_source desconocido: {recall_source}")
        self.recall = recall; self.recall_k = int(recall_k); self.recall_margin = float(recall_margin)
        self.recall_gain = float(recall_gain); self.recall_tau = float(recall_tau)
        self.recall_source = recall_source
        self.stored_labels = bool(stored_labels)
        if slow_error not in ("joint", "own"):
            raise ValueError(f"slow_error desconocido: {slow_error}")
        # "joint": la proyección aprende con el error de la decisión conjunta (lo que la
        # memoria ya acierta no le enseña nada al lento); "own": con su propio error, como
        # el neocórtex de CLS, que no queda «explicado» por el hipocampo.
        self.slow_error = slow_error
        self.recalls = 0                                      # inferencias que consultaron el almacén
        self._slow = None; self._fast = None; self._recalled = False
        # Esquemas (ver `_schema_fit`).
        if schema not in ("off", "measure", "gate", "always"):
            raise ValueError(f"schema desconocido: {schema}")
        self.schema = schema; self.schema_fit = float(schema_fit); self.schema_boost = float(schema_boost)
        self.schema_misfit_lr = float(schema_misfit_lr); self.schema_beta = float(schema_beta)
        self.schema_known = int(schema_known)
        # > 0: el encaje se mide contra una foto congelada del sistema lento (se renueva cada
        # `schema_frozen` pasos y al terminar cada sueño), no contra el que se está modificando.
        self.schema_frozen = int(schema_frozen); self._Wf = None; self._Wf_t = 0
        self.reliability = 0.0                                # acierto reciente del lento en clases conocidas
        self.fit = float("nan"); self.fit_parts = (float("nan"),) * 3
        self.assimilated = 0
        self._taught = np.zeros(self.C, dtype=np.int64)       # enseñanzas por clase
        # Repaso como autoexamen espaciado (ver `_replay_scheduled`).
        if replay_mode not in ("passive", "active", "scheduled"):
            raise ValueError(f"replay_mode desconocido: {replay_mode}")
        if spacing not in ("none", "equal", "expanding"):
            raise ValueError(f"spacing desconocido: {spacing}")
        self.replay_mode = replay_mode; self.spacing = spacing
        self.space_base = float(space_base); self.space_factor = float(space_factor)
        self.active_margin = float(active_margin); self.fail_weight = float(fail_weight)
        self.tests = 0; self.test_errors = 0
        self.replay_contrast = bool(replay_contrast)          # repaso dirigido como examen de a dos
        self._steps = 0                                       # pasos de enseñanza (reloj del calendario)
        # Sueño (ver `_sleep_tick`).
        if sleep_trigger not in ("steps", "load"):
            raise ValueError(f"sleep_trigger desconocido: {sleep_trigger}")
        if sleep_scope not in ("proj", "all"):
            raise ValueError(f"sleep_scope desconocido: {sleep_scope}")
        self.sleep_every = int(sleep_every); self.sleep_trigger = sleep_trigger
        self.sleep_load = int(sleep_load); self.sleep_replays_n = int(sleep_replays)
        self.sleep_micro = int(sleep_micro); self.sleep_downscale = float(sleep_downscale)
        self.sleep_scope = sleep_scope; self.sleep_prune = float(sleep_prune)
        self.sleep_release = float(sleep_release); self.sleep_keep = int(sleep_keep)
        if sleep_focus not in ("uniform", "weak"):
            raise ValueError(f"sleep_focus desconocido: {sleep_focus}")
        if sleep_release_check not in ("margin", "redundant"):
            raise ValueError(f"sleep_release_check desconocido: {sleep_release_check}")
        # "weak": el sueño repasa primero lo que el lento resuelve peor (margen más bajo
        # medido; lo nunca medido, antes), entre 64 candidatos al azar (repaso priorizado).
        # "redundant": un episodio sólo deja el almacén si, además, los demás recuerdos del
        # almacén lo siguen reconociendo con su etiqueta (no se pierde lo único que lo cubría).
        self.sleep_focus = sleep_focus; self.sleep_release_check = sleep_release_check
        # Con un área que infiere causas latentes: lo que ella protege (causa nueva junto a
        # un recuerdo dormido) tampoco se deprime en el sistema lento.
        self.cause_slow = bool(cause_slow)
        self.sleeps = 0; self.sleep_replays = 0; self.released = 0; self.pruned_synapses = 0
        self.sleeping = False; self._sleep_debt = 0; self._sleep_start = 0
        self._last_sleep = 0; self._created_at_sleep = 0
        self.rng = np.random.default_rng(seed)
        self.areas: list[FeatureArea] = []
        shape = self.input_shape
        if areas and not self.is_sheet:
            raise ValueError("las áreas sensoriales necesitan una lámina (alto, ancho, canales)")
        for i, spec in enumerate(areas):
            extra = dict(n_classes=self.C, seed=seed + 100 + i) if spec.get("local") else {}
            a = FeatureArea(shape, **spec, **extra)
            a.on_reuse = lambda f, i=i: self._on_reuse(i, f)
            self.areas.append(a); shape = a.out_shape
        self._predictive = any(a.pc is not None for a in self.areas)
        if self.fb is not None:
            self._pe_hist = [[] for _ in self.areas]
        self._sizes = [int(np.prod(a.out_shape)) for a in self.areas]
        self.heads: list[BioElasticNet] = []
        self._src: list = []
        for i, spec in enumerate(heads):
            self.add_head(**spec, seed=seed + i)
        # Proyección directa (lámina 5) de las áreas sensoriales a la decisión.
        self.proj = None; self._proj_src = None; self._balance = False
        if projection is not None and self.areas:
            spec = dict(projection); self._proj_src = spec.pop("source", "top")
            self._balance = bool(spec.pop("balance", False))
            n_in = sum(self._sizes[i] for i, _ in self._offsets(self._proj_src))
            self.proj = Readout(self.C, capacity=n_in, **{"consolidation": 0.0, "lr": 0.5, **spec})
            self.proj.n = n_in
        self._V = None; self.pred = 0; self._pv = None
        self.umbral = np.zeros(self.C, dtype=np.float32)          # homeostasis de las salidas
        self.workers = int(workers)    # hilos (no es parte del modelo: no se guarda)
        self._pool = None

    # ------------------------------------------------------------------ estructura
    def _input_size(self) -> int:
        return int(np.prod(self.input_shape)) if self.is_sheet else self.input_shape

    def _offsets(self, src):
        """[(área, desplazamiento)] que forman la volea de una fuente."""
        if src == "input" or not self.areas:
            return []
        if src == "top":
            return [(len(self.areas) - 1, 0)]
        if src == "all":
            off = np.concatenate([[0], np.cumsum(self._sizes)[:-1]])
            return [(i, int(o)) for i, o in enumerate(off)]
        return [(int(src), 0)]

    def add_head(self, source="top", seed: int = 0, **spec) -> BioElasticNet:
        """Agrega un área de asociación (se puede hacer en caliente, con la red ya entrenada)."""
        parts = self._offsets(source)
        n_in = sum(self._sizes[i] for i, _ in parts) if parts else self._input_size()
        head = BioElasticNet(n_in, self.C, seed=seed, **spec)
        self.heads.append(head); self._src.append(source)
        self._config["heads"].append(dict(spec, source=source, seed=seed))
        return head

    def _on_reuse(self, area: int, f: int):
        a = self.areas[area]
        F = a.out_shape[2]                                    # canales de salida (con la entrada agrupada)
        if area + 1 < len(self.areas):
            self.areas[area + 1].forget_channel(f)
        ids = np.arange(a.ph * a.pw) * F + f
        for head, src in zip(self.heads, self._src):
            for i, off in self._offsets(src):
                if i == area:
                    head.layer.forget_afferents(ids + off)
        if self.proj is not None:                             # la proyección también olvida ese canal
            for i, off in self._offsets(self._proj_src):
                if i == area:
                    self.proj.W[:, ids + off] = 0.0

    # ------------------------------------------------------------------ inferencia
    def _volley(self, src, x, codes):
        parts = self._offsets(src)
        if not parts:
            return x
        idxs, vals = [], []
        for i, off in parts:
            c = codes[i].ravel()
            idx = np.flatnonzero(c != 0)                      # (sobre bool es ~10x más rápido)
            idxs.append(idx + off); vals.append(c[idx])
        if len(idxs) == 1:
            return idxs[0], vals[0]
        return np.concatenate(idxs), np.concatenate(vals)

    def _proj_input(self, codes):
        """Volea normalizada que recibe la proyección directa. Con `balance`, cada área se
        normaliza por separado (misma voz para todas, aunque una tenga muchas más unidades
        activas que otra); si no, se normaliza la concatenación."""
        if not self._balance:
            idx, val = self._volley(self._proj_src, None, codes)
            return idx, val / np.float32(max(float(np.sqrt(val @ val)), 1e-6))
        parts = self._offsets(self._proj_src)
        idxs, vals = [], []
        for i, off in parts:
            c = codes[i].ravel(); j = np.flatnonzero(c); v = c[j]
            nv = float(np.sqrt(v @ v))
            if nv > 0:
                idxs.append(j + off); vals.append(v / np.float32(nv * np.sqrt(len(parts))))
        if not idxs:
            return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.float32)
        return np.concatenate(idxs), np.concatenate(vals)

    def _parallel(self):
        """Ejecutor y áreas de asociación que leen la entrada cruda, si hay con qué
        superponer trabajo; si no, (None, [])."""
        if self.workers < 2 or not self.areas:
            return None, []
        indep = [i for i, src in enumerate(self._src) if not self._offsets(src)]
        if not indep:
            return None, []
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.workers - 1)
        return self._pool, indep

    def forward(self, x, explore: bool = False) -> int:
        pool, indep = self._parallel()
        job = None
        if pool is not None:                                  # en paralelo con la jerarquía
            job = pool.submit(lambda: [self.heads[i].forward(x) for i in indep])
        codes = []
        if self.areas:
            X = np.asarray(x, dtype=np.float32).reshape(self.input_shape)
            x0 = X
            for a in self.areas:
                X = a.encode(X); codes.append(X)
            if self._predictive:
                codes = self._infer(x0, codes)
        self._codes = codes                                   # códigos del último pase (diagnóstico)
        for i, (head, src) in enumerate(zip(self.heads, self._src)):
            if job is None or i not in indep:
                head.forward(self._volley(src, x, codes))
        if job is not None:
            job.result()
        votes = [(head._last.V, head.readout.temperature) for head in self.heads]
        Vp = None
        if self.proj is not None:
            idx, h = self._proj_input(codes)
            self._pv = (idx, h)
            Vp = self.proj.potentials(idx, h)
            votes.append((Vp, self.proj.temperature))
        self._zs = []
        for a in self.areas:                                  # lectores locales (si hay error local)
            z = a.potentials()
            self._zs.append(z)
            if z is not None:
                votes.append((z, a.clf.temperature))
        self._votes_last = votes
        if self.arbiter > 0.0 and votes:
            if self.beta is None or len(self.beta) != len(votes):
                self.beta = np.ones(len(votes), dtype=np.float32)
            L = np.stack([z / np.float32(T) for z, T in votes])
            L = L - L.max(axis=1, keepdims=True)
            L = L - np.log(np.exp(L).sum(axis=1, keepdims=True))   # log-probabilidades
            self._L = L.astype(np.float32)
            V = self.beta @ self._L
        else:
            V = np.zeros(self.C, dtype=np.float32)
            for z, _ in votes:                                # mismo orden de suma
                V = V + z
        self._slow = None; self._fast = None; self._recalled = False
        if self.recall != "off" and self.arbiter <= 0.0:      # (con árbitro, V son log-probabilidades)
            V = self._two_systems(V, Vp)
        if self.homeostasis > 0.0:
            V = V - self.umbral
        self._V = V
        if explore:
            self.pred = int(self.rng.choice(self.C, p=self._p()))
        else:
            self.pred = int(np.argmax(V))
        return self.pred

    predict = forward

    def _infer(self, x0, codes):
        """Bucle de inferencia predictiva (ver docstring del módulo) y error de predicción
        final de cada área predictiva contra su entrada ascendente sin contexto."""
        fb = self.fb or {}
        iters = int(fb.get("iters", 0))
        lam = float(fb.get("up", 0.0)); gam = float(fb.get("context", 0.0))
        gmul = float(fb.get("gain", 0.0))
        bottom = [x0] + codes[:-1]
        for _ in range(iters):
            Q = [a.predict() for a in self.areas]             # con los códigos de la iteración anterior
            X = x0; new = []
            for i, a in enumerate(self.areas):
                bottom[i] = X
                q = Q[i]
                rep = up = X
                if q is not None:
                    if gam or gmul:
                        rep = X * (1.0 + np.float32(gmul) * q) if gmul else X
                        rep = np.maximum(rep + np.float32(gam) * q, 0.0) if gam else np.maximum(rep, 0.0)
                        if i > 0:
                            new[i - 1] = rep                  # el nivel de abajo, con contexto
                    up = np.maximum(rep + np.float32(lam) * (rep - q), 0.0) if lam else rep
                X = a.encode(up); new.append(X)
            codes = new
        for i, a in enumerate(self.areas):
            if a.pc is not None:
                a.prediction_error(bottom[i])
        return codes

    def _surprises(self):
        """Sorpresa con que se modula la plasticidad de cada área (None: la propia)."""
        if self.fb is None or not self._predictive or self.fb["source"] == "input":
            return None
        pe = [a.pe if a.pc is not None else None for a in self.areas]
        if self.fb["source"] == "code":
            return [pe[i + 1] if i + 1 < len(pe) and pe[i + 1] is not None else float("nan")
                    for i in range(len(pe))]
        out = []                                              # "shuffle": un error propio pasado
        for i, e in enumerate(pe):
            h = self._pe_hist[i]
            if e is None:
                out.append(None); continue
            out.append(h[int(self._fb_rng.integers(len(h)))] if h else e)
            h.append(e)
            if len(h) > 1000:
                h.pop(int(self._fb_rng.integers(len(h))))
        return out

    # ------------------------------------------------------------------ dos sistemas
    def _store(self):
        """Almacén rápido: el área de asociación de la que se repasa, si guarda huellas."""
        mem, _ = self._replay_source()
        return mem if mem is not None and mem.episodes is not None else None

    def _slow_part(self, store, Vp):
        """Sistema lento: la proyección directa y las áreas de asociación que no son el almacén."""
        S = np.zeros(self.C, dtype=np.float32)
        for head in self.heads:
            if head is not store:
                S = S + head._last.V
        if Vp is not None:
            S = S + Vp
        for z in self._zs:                                    # lectores locales de las áreas
            if z is not None:
                S = S + z
        return S

    def _two_systems(self, V, Vp):
        """Complementary Learning Systems (McClelland et al. 1995; Kumaran, Hassabis y
        McClelland 2016): el sistema lento (proyección + áreas) responde solo cuando está
        seguro; si no, se recuperan los `recall_k` episodios más parecidos del almacén
        rápido y se combinan (razonamiento basado en memoria).

          recall="gate"    se consulta el almacén sólo si el margen p1 - p2 del sistema
                           lento (softmax a la temperatura de la red) es < `recall_margin`
          recall="always"  siempre lento + almacén (sin compuerta)
          recall="fallback" lento + lectura del almacén siempre (como sin `recall`); los
                           episodios recuperados se suman sólo si esa decisión conjunta
                           duda (margen < `recall_margin`). Corrige la compuerta, que
                           dejaba afuera a la memoria cuando el lento estaba seguro y
                           equivocado (ver results/consolidacion_f1.json).
          recall_source="episodes"  voto de los episodios recuperados con su etiqueta
                           guardada, pesos softmax(coseno / `recall_tau`)
          recall_source="readout"   voto de la lectura aprendida del almacén (sólo compuerta)
        """
        store = self._store()
        if store is None or store._last is None:
            return V
        S = self._slow_part(store, Vp)
        self._slow = S
        if self.recall == "fallback":                         # el almacén vota siempre (ya está calculado);
            F = S + store._last.V                             # los episodios, sólo si los dos dudan
            e = np.exp((F - F.max()) / self.temperature)
            q = np.sort(e / e.sum())
            if (q[-1] - (q[-2] if len(q) > 1 else 0.0)) >= self.recall_margin:
                return F
            self._fast = self._recall_vote(store)
            self._recalled = True
            self.recalls += 1
            return F + np.float32(self.recall_gain) * self._fast
        use = True
        if self.recall == "gate":
            e = np.exp((S - S.max()) / self.temperature)
            q = np.sort(e / e.sum())
            use = (q[-1] - (q[-2] if len(q) > 1 else 0.0)) < self.recall_margin
        if not use:
            return S
        self._fast = self._recall_vote(store)
        self._recalled = True
        self.recalls += 1
        return S + np.float32(self.recall_gain) * self._fast

    def _recall_vote(self, store) -> np.ndarray:
        if self.recall_source == "readout":
            return store._last.V.astype(np.float32)
        v = store._last.v
        ids, sim = store.recall(v.idx, v.val, self.recall_k)
        out = np.zeros(self.C, dtype=np.float32)
        if len(ids) == 0:
            return out
        y = store.ep["y"][ids].astype(np.int64)
        if (y < 0).any():                                     # sin etiqueta: la de su lectura
            y = np.where(y >= 0, y, store.readout.preferred(ids))
        w = np.exp((sim - sim.max()) / max(self.recall_tau, 1e-6))
        ok = y >= 0
        if ok.any():
            np.add.at(out, y[ok], (w[ok] / w[ok].sum()).astype(np.float32))
        return out

    def parts(self) -> dict:
        """Aporte de cada sistema en la última inferencia: «slow» (proyección + áreas que no
        son el almacén) y «fast» (el almacén: su lectura aprendida o, con `recall`, el voto
        de los episodios recuperados, calculado aunque la compuerta no lo haya pedido).
        «recalled»: si la respuesta consultó el almacén."""
        store = self._store() or (self.heads[0] if self.heads else None)
        Vp = self.proj.potentials(*self._pv) if self.proj is not None and self._pv is not None else None
        slow = self._slow if self._slow is not None else self._slow_part(store, Vp)
        if self.recall == "off" or store is None or store.episodes is None:
            fast = store._last.V if store is not None else np.zeros(self.C, dtype=np.float32)
        elif self.recall == "fallback":
            fast = store._last.V + (np.float32(self.recall_gain) * self._fast if self._recalled else 0.0)
        else:
            fast = self._fast if self._fast is not None else self._recall_vote(store)
        return dict(slow=slow, fast=fast, recalled=self._recalled)

    def _labels(self, mem) -> np.ndarray:
        """Clase de cada recuerdo del almacén para el repaso: la etiqueta guardada en el
        episodio (`stored_labels`) o la preferida de la lectura aprendida (puede derivar)."""
        return mem.labels(stored=self.stored_labels)

    # ------------------------------------------------------------------ aprendizaje
    def _p(self):
        T = 1.0 if self._arbitrated() else self.temperature     # con árbitro V ya es log-prob
        e = np.exp((self._V - self._V.max()) / T)
        return e / e.sum()

    def _arbitrated(self) -> bool:
        return self.arbiter > 0.0 and self._L is not None

    def _learn_areas(self, target=None):
        """Aprendizaje de las áreas sensoriales. Con `target`, las que tienen error local lo
        usan (con el error residual de las anteriores, si así se configuró)."""
        prior = None
        zs = self._zs if len(self._zs) == len(self.areas) else [None] * len(self.areas)
        soft = self._soft if target is not None else None
        sur = self._surprises()
        for i, (a, z) in enumerate(zip(self.areas, zs)):
            if a.con is not None and i > 0:                   # no cierra si abajo sigue abierto
                a.hold = self.areas[i - 1].con is not None and self.areas[i - 1].phi > 0.5
            kw = {}
            if soft is not None:
                kw["soft"] = soft
            if sur is not None:
                kw["surprise"] = sur[i]
            a.learn(target, prior, **kw)
            if a.con is not None and a.reopened and i + 1 < len(self.areas):
                self.areas[i + 1].reopen()                    # su entrada cambió: reabre también
            if z is not None:
                prior = z if prior is None else prior + z

    def _learn_areas_and(self, fn, target=None):
        """Aprendizaje de las áreas sensoriales (en otro hilo) mientras en este corre
        `fn(i)` para las áreas de asociación que leen la entrada cruda. Devuelve cuáles ya
        se hicieron; las demás deben esperar, porque el reuso de un rasgo puede cortar sus
        sinapsis (`_on_reuse`)."""
        pool, par = self._parallel()
        if pool is None:
            self._learn_areas(target)
            return []
        job = pool.submit(lambda: self._learn_areas(target))
        try:
            for i in par:
                fn(i)
        finally:
            job.result()
        return par

    def _adapt_thresholds(self):
        """Homeostasis de las salidas: sube el umbral de la elegida, baja el de las demás."""
        if self.homeostasis > 0.0:
            self.umbral -= np.float32(self.homeostasis / self.C)
            self.umbral[self.pred] += np.float32(self.homeostasis)

    # ------------------------------------------------------------------ metacognición
    def voter_probs(self) -> np.ndarray:
        """(votantes, C): la distribución propia de cada votante en el último pase (áreas
        de asociación, proyección directa y lectores locales, en ese orden)."""
        out = []
        for z, T in self._votes_last:
            e = np.exp((np.asarray(z, dtype=np.float64) - float(np.max(z))) / float(T))
            out.append(e / e.sum())
        return np.array(out) if out else np.ones((1, self.C)) / self.C

    def conflict(self) -> float:
        """Desacuerdo entre votantes sobre la última decisión: 1 - probabilidad media que
        cada votante le da a la salida elegida (0: todos la dan por segura)."""
        return float(1.0 - self.voter_probs()[:, self.pred].mean())

    def familiarity(self) -> float:
        """Cuán conocida le resulta la última entrada: el mayor coseno entre la entrada y
        los prototipos de las áreas de asociación (0 si no tienen neuronas). Separa «esto es
        nuevo o distinto» (familiaridad baja) de «me equivoqué en algo conocido»
        (familiaridad alta y confianza baja)."""
        best = 0.0
        for h in self.heads:
            st = h._last
            if st is not None and len(st.v.cos):
                best = max(best, float(st.v.cos[0]))
        return best

    def experience(self) -> float:
        """Cuánto se le enseñó a lo que responde a la última entrada: media del contador de
        consolidación de las ganadoras de las áreas de asociación, como log(1 + n). Lo nuevo
        cae en neuronas recién reclutadas y poco enseñadas (experiencia baja) aunque se
        parezca a algo; un error en lo conocido ocurre con experiencia alta."""
        vals = []
        for h in self.heads:
            st = h._last
            if st is not None and len(st.v.winners) and not st.v.novel:
                vals.append(float(np.mean(h.readout.count[st.v.winners])))
        return float(np.log1p(np.mean(vals))) if vals else 0.0

    def evidence(self) -> float:
        """Cuánta evidencia absoluta tuvo la última decisión: media, entre los votantes, del
        mayor potencial de salida (antes de la competencia). La confianza es relativa (una
        salida contra las otras); la evidencia es baja cuando nada de lo aprendido responde
        con fuerza, que es lo que pasa ante algo nuevo."""
        if not self._votes_last:
            return 0.0
        return float(np.mean([float(np.max(z)) for z, _ in self._votes_last]))

    def _novelty_features(self) -> np.ndarray:
        return self._nov.features(self.raw_confidence(), self.familiarity(), self.experience(), self.evidence())

    def novelty(self) -> float:
        """P(la última entrada es de algo que casi no se me enseñó): con `metacog`, un
        segundo calibrador online que aprende de la propia historia de enseñanza (después
        de cada `teach` el sistema sabe si la clase enseñada tenía menos de `metacog["new"]`
        enseñanzas previas) a partir de familiaridad, experiencia y evidencia. Sin `metacog`
        devuelve 1 - familiaridad. Separa «esto es nuevo» de «me equivoqué en algo conocido»,
        que la confianza sola no distingue."""
        if self._nov is None:
            return 1.0 - self.familiarity()
        return self._nov.predict(self._novelty_features())

    def raw_confidence(self) -> float:
        """Probabilidad de la última decisión en la competencia entre salidas."""
        return float(self._p()[self.pred])

    @property
    def expected_uncertainty(self) -> float:
        """ACh: sorpresa media esperada (0 sin `neuromod`)."""
        return self._surprise.expected if self._surprise is not None else 0.0

    @property
    def unexpected_uncertainty(self) -> float:
        """NE: exceso reciente de sorpresa sobre lo esperado (0 sin `neuromod`)."""
        return self._surprise.unexpected if self._surprise is not None else 0.0

    @property
    def last_surprise(self) -> float:
        return self._surprise.last if self._surprise is not None else 0.0

    def confidence(self) -> float:
        """P(acierto) de la última decisión: calibrada (si hay `metacog`) o la cruda."""
        raw = self.raw_confidence()
        if self._cal is None:
            return raw
        return self._cal.predict(self._cal.features(raw, self.conflict(), self.familiarity(), self.experience()))

    def surprise(self, correct: bool) -> float:
        """Sorpresa del desenlace según la propia confianza: -log P(desenlace)."""
        return SurpriseTracker.surprise(self.confidence(), correct)

    def signals(self) -> dict:
        """Señales internas del último pase, leídas antes de saber si acertó (API estable,
        la consumen otras partes del proyecto):

          confidence              P(acierto) de la decisión: calibrada si hay `metacog`
                                  (regresión logística online sobre logit de la confianza
                                  cruda, conflicto, familiaridad y experiencia), si no la cruda
          raw_confidence          probabilidad de la decisión en la competencia de salidas
          conflict                desacuerdo entre votantes sobre la decisión (0 = todos la
                                  dan por segura): predice el error dentro de lo conocido
          margin                  diferencia entre las dos salidas más probables
          familiarity             coseno con el prototipo más cercano de las áreas de
                                  asociación: baja ante lo nuevo o distinto (novedad), alta
                                  ante un error en algo conocido
          experience              log(1 + consolidación media de las neuronas que
                                  responden): baja si lo que responde casi no fue enseñado
                                  (novedad), alta en lo conocido aunque se equivoque
          evidence                evidencia absoluta de la decisión (mayor potencial de
                                  salida, media entre votantes): baja ante lo nuevo
          novelty                 P(es algo que casi no se me enseñó): calibrador online
                                  sobre familiaridad, experiencia y evidencia, entrenado con
                                  la propia historia de enseñanza (con `metacog`; si no,
                                  1 - familiaridad)
          expected_uncertainty    ACh: sorpresa media esperada (0 sin `neuromod`)
          unexpected_uncertainty  NE: exceso estandarizado de la sorpresa reciente (0 sin
                                  `neuromod`): un cambio en el mundo
        """
        p = self._p(); srt = np.sort(p)
        return dict(confidence=self.confidence(), raw_confidence=float(p[self.pred]),
                    conflict=self.conflict(), margin=float(srt[-1] - srt[-2]) if len(p) > 1 else 1.0,
                    familiarity=self.familiarity(), experience=self.experience(),
                    evidence=self.evidence(), novelty=self.novelty(),
                    expected_uncertainty=self.expected_uncertainty,
                    unexpected_uncertainty=self.unexpected_uncertainty)

    def _outcome(self, correct: bool) -> float:
        """Desenlace conocido de la última decisión: calibra la confianza, actualiza la
        neuromodulación y devuelve la ganancia de la tasa de aprendizaje para esta vez."""
        raw = self.raw_confidence()
        g = 1.0
        s = self.surprise(correct) if self._surprise is not None else 0.0   # con la confianza de antes
        if self._cal is not None:
            self._cal.update(self._cal.features(raw, self.conflict(), self.familiarity(), self.experience()), correct)
        if self.hypercorrection > 0.0 and not correct:
            m = self._err_conf
            m[0] += raw; m[1] += 1
            c = raw if self.hypercorrection_mode == "confianza" else m[0] / m[1]
            g *= 1.0 + self.hypercorrection * c
        nm = self.neuromod
        if nm:
            ne = self._surprise.update(s)
            g *= (1.0 + nm["ne_gain"] * min(ne, nm["ne_cap"])) / (1.0 + nm["ach_gain"] * self.expected_uncertainty)
            if ne > nm["ne_reset"] and nm["release"] > 0.0:
                for h in self.heads:                          # se reabre lo consolidado
                    h.readout.count[:h.readout.n] *= np.float32(1.0 - nm["release"])
                self.releases += 1
        self.lr_gain = g
        return g

    def _readouts(self):
        return [h.readout for h in self.heads] + ([self.proj] if self.proj is not None else [])

    def _gained(self, g: float, fn, grow: bool = False):
        """Corre `fn` con la tasa de las lecturas multiplicada por `g` (y, con `grow`, con
        las áreas de asociación creciendo ante cualquier error)."""
        if g == 1.0 and not grow:
            return fn()
        saved = [(r, r.lr) for r in self._readouts()]
        rho = [(h, h.rho_sup) for h in self.heads]
        for r, lr in saved:
            r.lr = lr * g
        if grow:
            for h, _ in rho:
                h.rho_sup = 2.0
        try:
            return fn()
        finally:
            for r, lr in saved:
                r.lr = lr
            for h, v in rho:
                h.rho_sup = v

    def _modulated(self) -> bool:
        return bool(self.metacog or self.hypercorrection > 0.0 or self.neuromod)

    def labilize(self, fraction: float):
        """Reconsolidación: las lecturas de las áreas de asociación pierden una fracción de
        su consolidación acumulada (el contador que frena sus cambios) y vuelven a ser
        plásticas. Para cuando el sistema sabe, por un desenlace que sí observa, que su
        código actual no es fiable: lo consolidado en esas condiciones no debe protegerse."""
        f = np.float32(1.0 - min(max(float(fraction), 0.0), 1.0))
        for h in self.heads:
            h.readout.count[:h.readout.n] *= f

    def teach(self, target: int | None = None, soft=None, gain: float = 1.0):
        """Aprende de la última decisión con el objetivo `target`, o con un objetivo
        blando `soft` (distribución sobre las salidas, p. ej. el posterior que calcula el
        razonamiento): cada regla local usa soft - p como tercer factor donde usaba
        1[objetivo] - p (lecturas de las áreas de asociación, proyección, lectores locales
        y árbitro). La estructura (neuronas nuevas, STDP) sigue a la clase más probable.
        `gain`: ganancia escalar de la tasa de las lecturas para esta vez (neuromodulador
        externo: p. ej. más plasticidad mientras el código no es fiable)."""
        if soft is not None:
            soft = np.asarray(soft, dtype=np.float64)
            soft = soft / soft.sum()
            if target is None:
                target = int(np.argmax(soft))
        self._soft = soft
        try:
            if self._nov is not None:                         # ¿era algo que casi no se me enseñó?
                self._nov.update(self._novelty_features(),
                                 bool(self._taught[int(target)] < int(self.metacog.get("new", 10))))
                # el conteo de enseñanzas por clase lo suma `_teach` (una vez por enseñanza)
            if self._modulated():
                g = self._outcome(self.pred == int(target)) * float(gain)
                grow = bool(self.neuromod and self.neuromod["grow"]
                            and self.unexpected_uncertainty > self.neuromod["ne_reset"])
                return self._gained(g, lambda: self._teach(target), grow)
            if gain != 1.0:
                return self._gained(float(gain), lambda: self._teach(target))
            return self._teach(target)
        finally:
            self._soft = None

    def _teach(self, target: int):
        self._adapt_thresholds()
        p = self._p()                                         # no depende del aprendizaje de las áreas
        arb = self._arbitrated()
        soft = self._soft
        if arb:                                               # el árbitro aprende del error global
            g = (self._L @ soft if soft is not None else self._L[:, target]) - self._L @ p
            self.beta = np.maximum(self.beta + np.float32(self.arbiter) * g.astype(np.float32), 0.0)
        conf = None
        if (self.replay_focus == "confusion" and self.replay > 0.0 and self.proj is not None) or \
                (self.rehearse_focus == "confusion" and self.rehearse > 0.0):
            conf = self._confusion(target, p)                 # antes de que la memoria cambie
        grow, slow_lr = None, 1.0
        if self.schema != "off":                              # antes de que las áreas aprendan
            grow, slow_lr = self._schema_route(target)
        store = self._store() if grow is not None else None

        def one(i):
            kw = {} if soft is None else dict(soft=soft)
            if grow is not None and self.heads[i] is store:
                kw["grow"] = grow
            if self.joint and not arb:
                self.heads[i].teach(target, pred=self.pred, p=p, **kw)
            else:                                             # cada área aprende de su propio error
                self.heads[i].teach(target, **kw)
        done = self._learn_areas_and(one, target)
        for i in range(len(self.heads)):
            if i not in done:
                one(i)
        self._taught[target] += 1
        self._steps += 1
        if self.proj is not None:
            pp = None if arb else p
            prot = [c for h in self.heads if self.cause_slow and getattr(h, "causes", False)
                    for c in h.protect_classes.tolist() if c != target]
            if self.slow_error == "own" and not arb:
                S = self._slow if self._slow is not None else self._slow_part(
                    self._store(), self.proj.potentials(*self._pv))
                e = np.exp((S - S.max()) / self.temperature); pp = e / e.sum()
            if prot and pp is not None:                       # causa nueva: el lento no deprime lo viejo
                pp = np.array(pp, dtype=np.float64); pp[prot] = 0.0
            kw = {} if soft is None else dict(soft=soft)
            if slow_lr != 1.0:
                kw["lr_scale"] = slow_lr
            self.proj.learn(*self._pv, target, p=pp, **kw)
            if self.replay > 0.0:
                self._replay_debt += self.replay
                aimed = conf is not None and self.replay_focus == "confusion"
                while self._replay_debt >= 1.0:
                    self._replay_debt -= 1.0
                    if aimed:                                 # un repaso por paso va a la confusión
                        if self.replay_contrast:
                            self._contrast(conf[0], conf[1], conf[3], target)
                        else:
                            self._replay_pattern(conf[0], conf[1], conf[3])
                        aimed = False
                        self.directed += 1
                    else:
                        self._replay_one()
        if self.rehearse > 0.0:
            self._rehearse_debt += self.rehearse
            aimed = conf is not None and self.rehearse_focus == "confusion"
            while self._rehearse_debt >= 1.0:
                self._rehearse_debt -= 1.0
                if aimed:
                    self._rehearse_one(conf); aimed = False
                    self.directed += 1
                else:
                    self._rehearse_one()
        if conf is not None:
            conf[2].watch = -1
        if self.sleep_replays_n > 0 and (self.sleep_every > 0 or self.sleep_trigger == "load"):
            self._sleep_tick(self.sleep_micro if self.sleep_micro > 0 else None)

    # ------------------------------------------------------------------ sueño
    def _sleep_due(self) -> bool:
        if self.sleep_trigger == "load":                      # carga: episodios nuevos sin dormir
            store = self._store()
            return store is not None and store.layer.created - self._created_at_sleep >= self.sleep_load
        return self._steps - self._last_sleep >= self.sleep_every

    def _start_sleep(self, replays: int):
        self.sleeping = True; self._sleep_debt = int(replays)
        self._sleep_start = self._steps; self._last_sleep = self._steps

    def _sleep_tick(self, budget: int | None):
        """Sueño (Klinzing, Niethard y Born 2019): consolidación activa de sistemas
        (repaso intercalado del almacén rápido al lento, sin entrada nueva) seguida de
        renormalización sináptica global (Tononi y Cirelli 2014: los pesos del sistema
        lento se multiplican por 1 - `sleep_downscale`, lo que conserva sus relaciones) y
        poda (sinapsis del lento por debajo de `sleep_prune` x el máximo; y, con
        `sleep_release`, los episodios que el lento ya predice con margen >= ese valor
        dejan el almacén rápido, conservando `sleep_keep` por clase: el lugar queda para lo
        nuevo).

        Empieza cada `sleep_every` pasos (sleep_trigger="steps") o cuando el almacén
        escribió `sleep_load` episodios desde el último sueño ("load") y dura
        `sleep_replays` repasos. Con `sleep_micro` > 0 corre en micro-fases de a lo sumo
        ese número de repasos por paso (o por llamada a `idle`): la red nunca deja de
        responder ni de aprender; con 0, el sueño entero ocurre en el paso que lo dispara."""
        if not self.sleeping:
            if not self._sleep_due():
                return 0
            self._start_sleep(self.sleep_replays_n)
        k = self._sleep_debt if budget is None else min(int(budget), self._sleep_debt)
        for _ in range(k):
            if self.sleep_focus == "weak":
                self._replay_weak()
            else:
                self._replay_one()
        self.sleep_replays += k; self._sleep_debt -= k
        if self._sleep_debt <= 0:
            self._end_sleep()
        return k

    def _end_sleep(self):
        self.sleeping = False; self.sleeps += 1
        if self.schema_frozen > 0 and self.proj is not None:
            self._freeze()                                    # lo consolidado en el sueño
        d = self.sleep_downscale
        if d > 0.0:
            f = np.float32(1.0 - d)
            if self.proj is not None:
                self.proj.W[:, :self.proj.n] *= f
            if self.sleep_scope == "all":
                for h in self.heads:
                    h.readout.W[:, :h.readout.n] *= f
        if self.sleep_prune > 0.0 and self.proj is not None:
            W = self.proj.W[:, :self.proj.n]
            thr = self.sleep_prune * float(np.abs(W).max()) if W.size else 0.0
            small = (np.abs(W) < thr) & (W != 0.0)
            self.pruned_synapses += int(small.sum())
            W[small] = 0.0
        store = self._store()
        if self.sleep_release > 0.0 and store is not None and store.ep is not None:
            self._release(store)
        if store is not None:
            self._created_at_sleep = store.layer.created

    def _replay_weak(self):
        """Repaso priorizado del sueño: entre 64 episodios al azar, el de menor margen del
        lento medido (los nunca medidos tienen -1: primero)."""
        mem, space = self._replay_source()
        if mem is None or mem.ep is None:
            self._replay_one(passive=True)
            return
        lab = self._labels(mem)
        ok = np.flatnonzero(lab >= 0)
        if len(ok) == 0:
            return
        cand = ok if len(ok) <= 64 else self.rng.choice(ok, 64, replace=False)
        j = int(cand[int(np.argmin(mem.ep["m"][cand]))])
        self._replay_pattern(mem.episode(j), int(lab[j]), space, mem, j)

    def _redundant(self, store, j: int, y: int) -> bool:
        """¿Los demás recuerdos del almacén reconocen el episodio j con su etiqueta?"""
        x = store.episode(j)
        idx = np.flatnonzero(x > 0)
        v = store.layer.forward(idx, x[idx])
        if v.novel:
            return False
        others = v.winners[v.winners != j]
        if len(others) == 0:
            return False
        lab = self._labels(store)
        return int(lab[int(others[0])]) == y

    def _release(self, store):
        """Los episodios que el sistema lento ya resuelve (margen medido en este sueño >=
        `sleep_release`) dejan el almacén rápido; quedan al menos `sleep_keep` por clase."""
        n = store.layer.n; ep = store.ep
        lab = self._labels(store)
        cand = np.flatnonzero((ep["m_t"][:n] >= self._sleep_start) & (ep["m"][:n] >= self.sleep_release))
        if len(cand) == 0:
            return
        cnt = np.bincount(lab[lab >= 0], minlength=self.C)
        out = []
        checked = 0
        for j in cand[np.argsort(-ep["m"][cand], kind="stable")]:   # primero lo mejor consolidado
            c = int(lab[j])
            if c >= 0 and cnt[c] > self.sleep_keep:
                if self.sleep_release_check == "redundant":
                    if checked >= 64:                         # costo acotado por sueño
                        break
                    checked += 1
                    if not self._redundant(store, int(j), c):
                        continue
                cnt[c] -= 1; out.append(int(j))
        for j in sorted(out, reverse=True):                   # descendente: seguro con swap-remove
            store.layer.remove(j)
        self.released += len(out)

    def sleep(self, replays: int | None = None) -> int:
        """Duerme ahora, de un bloque (p. ej. entre sesiones): `replays` repasos (por
        defecto `sleep_replays`) y después renormalización, poda y liberación."""
        self._start_sleep(self.sleep_replays_n if replays is None else replays)
        return self._idle_run(None)

    def idle(self, budget: int = 1) -> int:
        """Entre consultas: avanza a lo sumo `budget` repasos del sueño pendiente (o lo
        empieza si corresponde). No toca lo necesario para aprender la última entrada."""
        if not self.sleeping and not (self.sleep_replays_n > 0 and self._sleep_due()):
            return 0
        if not self.sleeping:
            self._start_sleep(self.sleep_replays_n)
        return self._idle_run(budget)

    def _idle_run(self, budget):
        caches = [a._cache for a in self.areas]               # el repaso pisa la última codificación
        k = self._sleep_tick(budget)
        for a, c in zip(self.areas, caches):
            a._cache = c
        return k

    # ------------------------------------------------------------------ esquemas
    def _coverage(self) -> float:
        """Qué parte de la entrada explican las áreas sensoriales: fracción de la energía de
        los parches válidos cuyo mejor rasgo supera el umbral del área (promedio entre
        áreas). Usa la última codificación (antes de que las áreas aprendan)."""
        covs = []
        for a in self.areas:
            if a._cache is None:
                continue
            Pm, norms, vr, act = a._cache[:4]
            if act is None or len(vr) == 0:
                covs.append(0.0); continue
            w = norms[vr]
            covs.append(float((w * (act.max(axis=1) >= a.theta)).sum() / max(float(w.sum()), 1e-12)))
        return float(np.mean(covs)) if covs else 1.0

    def _schema_fit(self, target: int) -> float:
        """Encaje de (entrada, etiqueta) con el conocimiento del sistema lento (esquema, Tse
        et al. 2007; su lectura computacional, McClelland 2013):

            encaje = cobertura x fiabilidad x (1 - conflicto)

          cobertura   cuánto de la entrada explican los diccionarios de las áreas
          fiabilidad  acierto reciente (media móvil, tasa `schema_beta`) del sistema lento
                      en clases ya enseñadas >= `schema_known` veces: si lo conocido no se
                      predice bien, no hay esquema
          conflicto   probabilidad de la clase equivocada más fuerte según el lento: un
                      ítem que contradice lo que el lento cree con seguridad exigiría
                      reestructurarlo (interferencia)
        """
        store = self._store()
        if self.schema_frozen > 0 and self.proj is not None:  # lo consolidado, no lo que está cambiando
            if self._Wf is None or self._steps - self._Wf_t >= self.schema_frozen:
                self._freeze()
            idx, h = self._pv
            S = self._Wf[:, idx] @ h
        else:
            S = self._slow if self._slow is not None else self._slow_part(
                store, self.proj.potentials(*self._pv) if self.proj is not None else None)
        e = np.exp((S - S.max()) / self.temperature); q = e / e.sum()
        conflict = float(np.max(np.delete(q, target))) if self.C > 1 else 0.0
        cov = self._coverage()
        rel = self.reliability
        fit = cov * rel * (1.0 - conflict)
        if self._taught[target] >= self.schema_known:         # la fiabilidad se mide en lo conocido
            ok = float(int(np.argmax(S)) == target)
            self.reliability += self.schema_beta * (ok - self.reliability)
        self.fit = fit; self.fit_parts = (cov, rel, conflict)
        return fit

    def _freeze(self):
        self._Wf = self.proj.W[:, :self.proj.n].copy(); self._Wf_t = self._steps

    def _schema_route(self, target: int):
        """(neurogénesis permitida en el almacén, factor de tasa del sistema lento).

          schema="measure"  sólo mide el encaje (diagnóstico, sin efecto)
          schema="gate"     encaja (>= `schema_fit`): asimilación rápida en el lento (tasa
                            x `schema_boost`) y sin neurona nueva en el almacén; no encaja:
                            queda en el almacén (neurogénesis habilitada) y el lento aprende
                            con tasa x `schema_misfit_lr` (0 = sólo desde el repaso)
          schema="always"   asimilación rápida de todo (ablación: sin medir el encaje)
        """
        fit = self._schema_fit(target)
        if self.schema == "gate":
            if fit >= self.schema_fit:
                self.assimilated += 1
                return False, self.schema_boost
            return None, self.schema_misfit_lr
        if self.schema == "always":
            self.assimilated += 1
            return None, self.schema_boost
        return None, 1.0

    def _confusion(self, target: int, p):
        """Repaso dirigido a la confusión (cf. MIR, Aljundi et al. 2019; ASER, Shim et al.
        2021): la clase que más compite con el objetivo en esta entrada (la de mayor
        probabilidad después de él, si pasa de `confusion_p`) y, de esa clase, el recuerdo
        de la memoria más parecido a lo que entra: la ganadora de esa clase si disparó, si
        no la de mayor coincidencia entre sus neuronas. Es el ejemplo viejo que el
        aprendizaje de ahora más va a perturbar, justo en la frontera.

        Devuelve (recuerdo denso, clase, memoria, espacio) o None. El recuerdo se copia
        antes de que la memoria aprenda (puede desalojar o mover neuronas); la neurona
        queda marcada en `memoria.watch` para excluirla del ensayo."""
        mem, space = self._replay_source()
        if mem is None or mem._last is None:
            return None
        q = np.asarray(p, dtype=np.float64).copy()
        q[target] = -1.0
        c2 = int(np.argmax(q))
        if q[c2] < self.confusion_p:
            return None
        L = mem.layer
        v = mem._last.v
        if len(v.winners) and int(v.winners.max()) >= L.n:
            return None
        lab = self._labels(mem)
        pw = lab[v.winners]
        hit = np.flatnonzero(pw == c2)
        if len(hit):
            j = int(v.winners[hit[0]])
        else:
            if v.norm == 0.0:
                return None
            rows = np.flatnonzero(lab == c2)
            if len(rows) == 0:
                return None
            cos = L._cos_at(rows, v.idx, v.val, v.norm, L._fan_max())
            j = int(rows[int(np.argmax(cos))])
        mem.watch = j
        return mem.episode(j).copy(), c2, mem, space

    def _replay_source(self):
        """Área de asociación de la que se puede repasar y en qué espacio guarda sus recuerdos.

        "input": sus neuronas son patrones de entrada; el recuerdo se reactiva y recorre
                 las áreas sensoriales (como en el sueño, sin entrenar los rasgos).
        "code" : sus neuronas ya están en el espacio de la proyección (lee de la misma
                 fuente); el recuerdo se usa directamente.
        Sin ninguna de las dos, no hay de dónde repasar."""
        usable = [(h, src) for h, src in zip(self.heads, self._src)
                  if h.layer.n and h.context == 1 and h.tau <= 0]
        for h, src in usable:
            if not self._offsets(src):
                return h, "input"
        for h, src in usable:
            if self._offsets(src) == self._offsets(self._proj_src):
                return h, "code"
        return None, None

    def _replay_one(self, passive: bool = False):
        """Consolidación por repetición: un recuerdo del área de asociación (el patrón que
        guarda una de sus neuronas) se reactiva y entrena la proyección directa con la clase
        de esa neurona. Las clases se eligen de forma pareja, así lo viejo se repasa aunque
        el flujo ya no lo muestre. Con replay_focus="absent" (y una proyección con
        `active_window`) se repasan sólo las clases ausentes del flujo: las presentes ya
        reciben ejemplos reales.
        Con replay_mode="active" o "scheduled", ver `_replay_scheduled`."""
        if self.replay_mode != "passive" and not passive:
            self._replay_scheduled()
            return
        mem, space = self._replay_source()
        if mem is None:
            return
        L = mem.layer
        pref = self._labels(mem)
        classes = np.unique(pref[pref >= 0])
        if self.replay_focus == "absent" and self.proj.active_window > 0 and len(classes):
            absent = classes[~self.proj.active()[classes]]
            if len(absent):
                classes = absent
        if len(classes) == 0:
            return
        c = int(self.rng.choice(classes))
        j = int(self.rng.choice(np.flatnonzero(pref == c)))
        self._replay_pattern(mem.episode(j), c, space, mem, j)

    def _reactivate(self, x, space: str):
        """Recuerdo denso `x` (espacio de la memoria) -> (volea normalizada de la proyección,
        códigos de las áreas). El repaso no entrena los rasgos de las áreas."""
        codes = []
        if space == "input":
            X = (x / max(float(x.max()), 1e-8)).reshape(self.input_shape)
            for a in self.areas:
                X = a.encode(X); codes.append(X)
                a._cache = None                               # el repaso no entrena los rasgos
            idx, h = self._proj_input(codes)
        else:
            idx = np.flatnonzero(x > 0); val = x[idx]
            h = val / np.float32(max(float(np.sqrt(val @ val)), 1e-6))
        if len(idx) == 0:
            return None
        return idx, h, codes

    def _replay_local(self, c: int, codes):
        """Los lectores locales de las áreas (si hay error local) también aprenden del repaso."""
        for a, X in zip(self.areas, codes):
            a.learn_readout(c, X)

    def _replay_pattern(self, x, c: int, space: str, mem=None, j: int = -1):
        """Reactiva el recuerdo denso `x` (espacio de la memoria) y entrena la proyección
        directa con la clase c, con el error completo. Con `mem` y `j`, anota en el
        episodio el margen del sistema lento antes de aprender (lo usa el sueño)."""
        r = self._reactivate(x, space)
        if r is None:
            return
        idx, h, codes = r
        self._replay_local(c, codes)
        V = self.proj.potentials(idx, h)
        e = np.exp((V - V.max()) / self.temperature)
        q = e / e.sum()
        if mem is not None and j >= 0 and mem.ep is not None:
            self._note_margin(mem, j, q, c)
        self.proj.learn(idx, h, c, p=q, replay=True)
        self.replays += 1

    def _contrast(self, x, c: int, space: str, rival: int):
        """Autoexamen dirigido contra la clase que compite (propuesta de Continuo 2): el
        recuerdo de la clase c más cercano a lo que entra (ver `_confusion`) se presenta a
        la proyección y se compara su puntaje para c con el de `rival`, la clase que se
        está enseñando, aunque sea la única presente en el flujo. Si c no le gana por más
        de `active_margin` (en la competencia de a dos), aprende con el error de esa
        competencia: sólo cambian las filas c y rival."""
        r = self._reactivate(x, space)
        if r is None:
            return
        idx, h, codes = r
        V = self.proj.potentials(idx, h)
        pair = np.array([c, rival])
        z = V[pair] / self.temperature
        e = np.exp(z - z.max()); e /= e.sum()
        self.tests += 1
        self.replays += 1
        if e[0] - e[1] > self.active_margin:
            return
        self.test_errors += 1
        q = np.zeros(self.C); q[pair] = e
        self._replay_local(c, codes)
        self.proj.learn(idx, h, c, p=q, replay=True)

    def _note_margin(self, mem, j: int, q, c: int):
        other = np.delete(q, c)
        mem.ep["m"][j] = np.float32(q[c] - (other.max() if len(other) else 0.0))
        mem.ep["m_t"][j] = self._steps

    # ------------------------------------------------------------------ autoexamen espaciado
    def _pick_scheduled(self, mem, lab):
        """Recuerdo a examinar. Sin espaciado: muestreo proporcional a 1 + `fail_weight` x
        (media móvil de sus fallos). Con espaciado: entre los vencidos, el de mayor
        prioridad = `fail_weight` x fallo + atraso / intervalo; si no hay vencidos, el que
        vence antes (el presupuesto de repasos se gasta igual)."""
        ep = mem.ep; n = mem.layer.n
        ok = np.flatnonzero(lab >= 0)
        if len(ok) == 0:
            return -1
        fail = ep["fail"][ok]
        if self.spacing == "none":
            w = 1.0 + self.fail_weight * fail.astype(np.float64)
            return int(self.rng.choice(ok, p=w / w.sum()))
        due = ep["due"][ok]
        late = self._steps - due
        ready = late >= 0
        if not ready.any():
            return int(ok[int(np.argmin(due))])
        cand = ok[ready]
        pri = self.fail_weight * fail[ready] + late[ready] / np.maximum(ep["gap"][cand], 1.0)
        return int(cand[int(np.argmax(pri))])

    def _replay_scheduled(self):
        """Repaso como autoexamen (efecto de testeo, Roediger y Karpicke 2006): se presenta
        la clave (el episodio guardado), el sistema lento predice y se compara con la
        etiqueta guardada. replay_mode="active": aprende sólo si falla (o si el margen
        p(etiqueta) - p(otra) no supera `active_margin`); "scheduled": misma elección y
        calendario, pero aprende siempre (ablación del examen). El resultado ajusta el
        calendario: acierto -> el intervalo se mantiene (spacing="equal") o se multiplica
        por `space_factor` ("expanding"); fallo -> vuelve a `space_base`. Un recuerdo que
        se sabía y ahora falla (se está olvidando) queda con prioridad máxima. Un examen
        adelantado (no había ninguno vencido) que acierta no estira el intervalo."""
        mem, space = self._replay_source()
        if mem is None or mem.ep is None:
            self._replay_one(passive=True)
            return
        lab = self._labels(mem)
        j = self._pick_scheduled(mem, lab)
        if j < 0:
            return
        c = int(lab[j])
        r = self._reactivate(mem.episode(j), space)
        if r is None:
            return
        idx, h, codes = r
        V = self.proj.potentials(idx, h)
        e = np.exp((V - V.max()) / self.temperature); q = e / e.sum()
        other = np.delete(q, c)
        passed = int(np.argmax(q)) == c and q[c] - (other.max() if len(other) else 0.0) > self.active_margin
        self._note_margin(mem, j, q, c)
        self.tests += 1
        if not passed:
            self.test_errors += 1
        if self.replay_mode == "scheduled" or not passed:
            self._replay_local(c, codes)
            self.proj.learn(idx, h, c, p=q, replay=True)
        self.replays += 1
        ep = mem.ep
        if passed:
            g = float(ep["gap"][j])
            early = self._steps < int(ep["due"][j])             # examen adelantado (no había vencidos)
            if g <= 0.0:
                g = self.space_base
            elif self.spacing == "expanding" and not early:
                g = min(g * self.space_factor, 1e7)
            ep["gap"][j] = g
            ep["fail"][j] *= np.float32(0.5)
        else:
            forgetting = ep["ok"][j] == 1
            ep["gap"][j] = self.space_base
            ep["fail"][j] = np.float32(1.0) if forgetting else np.float32(0.5) * ep["fail"][j] + np.float32(0.5)
        ep["due"][j] = self._steps + int(round(float(ep["gap"][j])))
        ep["ok"][j] = 1 if passed else 0
        ep["tests"][j] += 1

    def _rehearse_one(self, conf=None):
        """Ensayo en la propia memoria: el recuerdo de una neurona (clase elegida de forma
        pareja) vuelve a entrar como si fuera un ejemplo, sin la neurona que lo guarda, y
        las demás ganadoras corrigen su lectura con el error completo. Mantiene la frontera
        con clases ausentes: las neuronas nuevas que invaden el territorio de una clase
        vieja aprenden que ahí también vive esa clase. No crea ni mueve neuronas.
        Con `conf` (ver `_confusion`) el recuerdo es el de la clase que compite con lo que
        acaba de entrar, el más cercano a la entrada."""
        if conf is not None:
            x, c, mem, _ = conf
            j = mem.watch
        else:
            mem = next((h for h, src in zip(self.heads, self._src)
                        if not self._offsets(src) and h.context == 1 and h.tau <= 0 and h.layer.n), None)
            if mem is None:
                return
            pref = self._labels(mem)
            classes = np.unique(pref[pref >= 0])
            if len(classes) == 0:
                return
            c = int(self.rng.choice(classes))
            j = int(self.rng.choice(np.flatnonzero(pref == c)))
            x = mem.episode(j)
        L, R = mem.layer, mem.readout
        idx = np.flatnonzero(x > 0); val = x[idx] / max(float(x.max()), 1e-8)
        v = L.forward(idx, val)
        keep = v.winners != j
        if v.novel or not keep.any():
            return
        wn = v.winners[keep]
        h = mem._votes(v)[keep]
        h = h / h.sum()
        R.learn(wn, h, c, replay=True)
        self.rehearsals += 1

    def reinforce(self, reward: float):
        if self._modulated():
            g = self._outcome(reward > 0)
            return self._gained(g, lambda: self._reinforce(reward))
        return self._reinforce(reward)

    def _reinforce(self, reward: float):
        self._adapt_thresholds()
        p = self._p()
        one = lambda i: self.heads[i].reinforce(reward, pred=self.pred, p=p)
        done = self._learn_areas_and(one)
        for i in range(len(self.heads)):
            if i not in done:
                one(i)
        if self.proj is not None:
            self.proj.learn_reward(*self._pv, self.pred, reward, p=p)

    def observe(self, x):
        self.forward(x)
        one = lambda i: self.heads[i].observe()
        done = self._learn_areas_and(one)
        for i in range(len(self.heads)):
            if i not in done:
                one(i)

    def step(self, x, target: int) -> int:
        pred = self.forward(x)
        self.teach(target)
        return pred

    def reset_context(self):
        for head in self.heads:
            head.reset_context()

    def new_sequence(self):
        """Empieza otra secuencia de vistas (otra imagen): se borran las trazas de las áreas."""
        for a in self.areas:
            a.reset_trace()

    # ------------------------------------------------------------------ estado
    def stats(self) -> dict:
        areas = [a.stats() for a in self.areas]
        heads = [dict(h.stats(), synapses=h.layer.synapses()) for h in self.heads]
        return dict(
            depth=len(self.areas) + 1,
            areas=areas, heads=heads,
            features=sum(a["features"] for a in areas),
            neurons=sum(h["neurons"] for h in heads),
            units=sum(a["units"] for a in areas) + sum(h["neurons"] for h in heads) + self.C,
            synapses=sum(a["synapses"] for a in areas) + sum(h["synapses"] for h in heads)
                     + sum(h["neurons"] for h in heads) * self.C
                     + (self.proj.n * self.C if self.proj is not None else 0),
            replays=self.replays, recalls=self.recalls, assimilated=self.assimilated,
            tests=self.tests, sleeps=self.sleeps, sleep_replays=self.sleep_replays,
            released=self.released,
        )

    def cost(self) -> dict:
        """Operaciones sinápticas (multiplicación-suma) por inferencia y memoria, por parte.

        Cuenta lo que hace el código, no lo que guarda: un área sensorial evalúa su
        diccionario en cada posición, y un área de asociación con el pase denso evalúa
        todas sus neuronas (n x fan-in máximo). Es una cota superior para las áreas que
        usan el índice invertido. El repaso agrega, por paso de entrenamiento, `replay`
        codificaciones más de la jerarquía."""
        parts = {}
        passes = 1 + (int(self.fb["iters"]) if self.fb is not None else 0)
        for i, a in enumerate(self.areas):
            c = a.cost()
            parts[f"area{i}"] = dict(macs=c["macs"] * passes, memory_bytes=c["memory_bytes"])
            if "pred_macs" in c:                              # predicciones descendentes por inferencia
                parts[f"pred{i}"] = dict(macs=c["pred_macs"] * passes, memory_bytes=0)
        for i, h in enumerate(self.heads):
            L = h.layer
            kf = int(L.fill[:L.n].max()) if L.n else 0
            parts[f"head{i}"] = dict(macs=L.n * kf,
                                     memory_bytes=L.memory()["total"] + h.readout.W.nbytes + h.readout.count.nbytes)
        if self.proj is not None:
            parts["proj"] = dict(macs=self.C * self.proj.n,          # cota: todas las entradas activas
                                 memory_bytes=self.proj.W.nbytes + self.proj.count.nbytes)
        infer = sum(p["macs"] for p in parts.values())
        area_macs = sum(parts[f"area{i}"]["macs"] for i in range(len(self.areas)))
        return dict(parts=parts, macs_per_inference=infer,
                    macs_per_training_step=infer + self.replay * (area_macs + parts.get("proj", {}).get("macs", 0)),
                    memory_bytes=sum(p["memory_bytes"] for p in parts.values()))

    def save(self, path):
        """Guarda estructura y pesos. El contexto en curso (retardos, trazas) no se guarda."""
        out = {"config": np.array(json.dumps(self._config)),
               "meta": np.array([self.replays], dtype=np.int64)}
        for i, a in enumerate(self.areas):
            out.update({f"a{i}_{k}": v for k, v in a.state().items()})
        for i, h in enumerate(self.heads):
            out.update({f"h{i}_{k}": v for k, v in h.state().items()})
        if self.proj is not None:
            out.update(proj_W=self.proj.W, proj_count=self.proj.count,
                       proj_baseline=np.float64(self.proj.baseline))
        if self.beta is not None:
            out["beta"] = self.beta
        if self.homeostasis > 0.0:
            out["umbral"] = self.umbral
        if self._cal is not None:
            out["cal"] = self._cal.w; out["nov"] = self._nov.w; out["taught"] = self._taught
        np.savez_compressed(path, **out)

    @classmethod
    def load(cls, path) -> "Cortex":
        with np.load(path, allow_pickle=False) as z:
            cfg = json.loads(str(z["config"]))
            shape = cfg.pop("input_shape")
            head_specs = cfg.pop("heads")
            net = cls(tuple(shape) if isinstance(shape, list) else shape, cfg.pop("n_outputs"),
                      heads=[], **cfg)
            for i, a in enumerate(net.areas):
                a.restore(z, f"a{i}_")
            for i, spec in enumerate(head_specs):
                sub = {k[len(f"h{i}_"):]: z[k] for k in z.files if k.startswith(f"h{i}_")}
                net.heads.append(BioElasticNet.from_state(sub))
                net._src.append(spec.get("source", "top"))
            net._config["heads"] = head_specs
            if net.proj is not None:
                net.proj.W[:] = z["proj_W"]; net.proj.count[:] = z["proj_count"]
                net.proj.baseline = float(z["proj_baseline"])
            net.replays = int(z["meta"][0])
            if "beta" in z.files:
                net.beta = z["beta"].copy()
            if "umbral" in z.files:
                net.umbral[:] = z["umbral"]
            if "cal" in z.files and net._cal is not None:
                net._cal.w = z["cal"].astype(np.float64).copy()
                if "nov" in z.files:
                    net._nov.w = z["nov"].astype(np.float64).copy(); net._taught[:] = z["taught"]
        return net
