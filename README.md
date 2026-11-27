# bioelastic_snn

Red neuronal de spikes **bio-elástica**: aprende online, muestra por muestra, sin backprop ni
gradientes globales. **Crea y elimina neuronas según lo que recibe**, nunca deja de aprender y el
mismo núcleo procesa imagen, voz, texto o sensores (sólo cambia el encoder). Sobre la percepción
corre un **Thinker**: un lazo de razonamiento sobre símbolos con memoria rápida, registros y
aprendizaje por refuerzo.

NumPy (y SciPy para el encoder de imagen). **307 tests**, incluidos tests de regresión
bit a bit. Todo lo que figura abajo está medido con los scripts de `experiments/`, y los CSV de
cada corrida están en `results/`.

> **Estado (v0.3-dev, noviembre 2026).** Este README separa lo **medido** de lo que sigue
> siendo **hipótesis**. El objetivo del proyecto es encontrar dónde falla: la sección
> [Dónde falla](#dónde-falla) importa tanto como los resultados.
> Hay rondas en curso que implementan mecanismos del aprendizaje humano
> (ver [En curso](#en-curso-mecanismos-del-aprendizaje-humano)) y una
> [hoja de ruta](#hoja-de-ruta-hacia-el-aprendizaje-humano) con criterios de abandono.

## Arquitectura

```mermaid
flowchart LR
    X["Dato crudo<br/>imagen · audio · texto · sensores"] --> E["Encoder<br/>un spike por aferente<br/>(más fuerte = más temprano)"]
    E --> A1["Área sensorial 1<br/>L4: diccionario elástico<br/>L2/3: k-WTA + pooling<br/>readout local de error"]
    A1 --> A2["Área sensorial 2 … n"]
    E --> M["Memoria asociativa<br/>neuronas TTFS de fan-in acotado<br/>neurogénesis · poda · consolidación<br/>huella episódica"]
    A1 --> P["Proyección directa (L5)<br/>regla de 3 factores"]
    A2 --> P
    M --> AR(("Árbitro"))
    P --> AR
    A1 --> AR
    AR --> T["Thinker<br/>espacio de trabajo · memoria rápida<br/>registros · actor-crítico TD(λ)"]
    T --> O(("Respuesta"))
    M -. "repaso dirigido<br/>a la confusión" .-> P
    O -. "error o recompensa" .-> T
    T -. "crédito simbólico<br/>(qué símbolo explicaría el error)" .-> AR
```

| Cerebro | En el código | Estado |
|---|---|---|
| Retina, cóclea, tálamo | `encoders.py`: código de latencia | hecho |
| Corteza sensorial, lámina 4 | `FeatureArea`: diccionario elástico de rasgos locales | hecho |
| Láminas 2/3 (inhibición lateral, salida al área siguiente) | k-WTA + pooling en `FeatureArea` | hecho |
| Error local por área | readout de 3 factores por área (estilo Nøkland) | hecho |
| Lámina 5 (salida hacia la decisión) | `Cortex.proj` | hecho |
| Hipocampo (memoria rápida) y repaso | `BioElasticNet` + huella episódica + repaso dirigido | hecho |
| Dopamina: error de predicción de recompensa | TD(λ) con trazas en el Thinker; tercer factor en la percepción | hecho |
| Corteza prefrontal: memoria de trabajo | registros del Thinker, aprendidos en retrospectiva | hecho |
| Ligadura rápida tipo cabeza de inducción | `FastMemory` (por eventos, conjuntiva) | hecho |
| Láminas 6 y 1 (predicción descendente) | `predictive` de `FeatureArea`, `feedback` de `Cortex` | opción; no hace que la profundidad sume |
| Cingulado anterior (detección del propio error), ACh/NE | `Cortex.signals()`, `metacog` (`metacog.py`), compuerta de fiabilidad (`posterior.py`) | opción; medido |
| Consolidación en sueño, reconsolidación, esquemas, causas latentes | opciones de `Cortex`, `BioElasticNet` y `Readout` | opción; medido |
| Aprender a aprender (prefrontal como meta-RL) | `prev_io` del Thinker, `MetaThinker` | hecho (ver resultados) |
| Mapa cognitivo recorrible en ambos sentidos | `back` del Thinker | hecho (opción); ya no daña la transitiva |
| Prefrontal: conjuntos de tareas y causa latente de cada regla | `task_sets` del Thinker (`tasksets.py`) | opción; falla con todo encendido |
| Estructura del mapa separada de lo sensorial (entorrinal) | `cogmap` del Thinker (`cogmap.py`) | opción |
| Dopamina, serotonina, acetilcolina y noradrenalina como meta-parámetros | `modulators` del Thinker (`modulators.py`) | opción |
| Cerebelo (69 de los 86 mil millones de neuronas) | — | no modelado |

## Resultados medidos

![resultados v0.3](results/v03.png)

### Percepción (3 semillas, una sola pasada online)

| Test 10k | 1 nivel | 2 niveles | 3 niveles | CNN con backprop, misma pasada |
|---|---|---|---|---|
| MNIST | 96.9 ± 0.1 % | 98.9 ± 0.1 % | **99.0 ± 0.1 %** | 98.5 % (la mejor de 2) |
| Fashion-MNIST | 82.4 ± 0.3 % | **89.5 ± 0.4 %** | 89.4 ± 0.4 % | 88.3 % (la mejor de 2) |

Robustez con perturbaciones que la red nunca vio. En Fashion-MNIST, 2 niveles contra las dos CNN
(1 y 2 capas convolucionales):

| Perturbación | BioElastic | CNN de 1 capa | CNN de 2 capas |
|---|---|---|---|
| Ruido | **84.9 %** | 52.6 % | 79.7 % |
| Oclusión | **84.0 %** | 77.4 % | 72.8 % |
| Desplazamiento de 3 px | 51.1 % | 46.9 % | **56.0 %** |

En MNIST, con 3 niveles y contra la mejor CNN, la red gana en ruido (98.4 % contra 97.1 %) y en
oclusión (84.9 % contra 81.1 %). El desplazamiento la deja pareja con las CNN.

### Razonamiento sobre símbolos (Thinker, símbolos perfectos, 3–5 semillas)

| Prueba | Resultado | Azar |
|---|---|---|
| Ligadura de variables dentro del episodio | 100 % (red sin Thinker: 15.6 %) | 3.1 % |
| Cadenas de 3 y 4 saltos, nunca entrenadas | 100 % | 1.6 % |
| Composición f(f(x)) con x nunca vistos | 100 % (sin habla interna: 0 %) | 6.3 % |
| Composición de 5 a 8 saltos, con 16 distractores y 8 ruidos, vocabulario de 256 | 100 % | — |
| Suma decimal con acarreo aprendida desde cero (12 000 episodios), 8 dígitos exacta | 94.2 % (74.2–100 %) | ~0 % |

### Aprendizaje continuo (flujo no estacionario de 200 000 pasos, 3 semillas)

| Modelo | Exactitud final | Olvido |
|---|---|---|
| **BioElastic** | **89.9 ± 0.8 %** | 13.1 ± 1.3 % |
| MLP + replay | 85.7 ± 0.6 % | 22.1 ± 2.0 % |
| SLDA | 78.8 ± 1.7 % | 7.8 ± 1.2 % |
| kNN + buffer | 61.3 ± 1.9 % | 49.4 ± 0.9 % |

SLDA olvida menos pero aprende menos. Sin neurogénesis, la exactitud final cae a entre 31 % y 57 %
(flujo a escala 0.1): la plasticidad depende de crear neuronas.

**Clases parecidas (ronda continuo 2, 3 semillas).** Se entrenan primero unas prendas (C) y
después otras (D). Con la huella episódica, el repaso dirigido a la clase que más compite y la
separación al reclutar, el resultado sobre todas las clases sube de 0.651 a **0.776**, a la par de
ER-ACE con los mismos bytes (0.784). El pulóver, que antes se olvidaba por completo (~0.01), queda
en 0.42.

**Split por clases, una pasada** (test, semillas 3–5; ER y ER-ACE corridos acá con los mismos bytes):

| | c2 | c2 + `ace` | ER-ACE | ER |
|---|---|---|---|---|
| Split-MNIST, exactitud | 0.966 ± 0.001 | 0.970 ± 0.001 | 0.950 ± 0.002 | 0.954 ± 0.008 |
| Split-MNIST, olvido | 0.034 | 0.028 | 0.024 | 0.040 |
| Split-Fashion, exactitud | 0.809 ± 0.010 | 0.833 ± 0.003 | 0.843 ± 0.010 | 0.852 ± 0.006 |
| Split-Fashion, olvido | 0.187 | 0.152 | 0.092 | 0.137 |

Por qué perdía en Split-Fashion:

- **No es capacidad.** Con 1 000 a 8 000 neuronas da 0.805, 0.809, 0.805 y 0.799, contra 0.84 de
  ER-ACE en todas. Los desalojos caen 5 veces sin que se mueva la exactitud.
- **No es el aprendiz.** Con las mismas imágenes mezcladas llega a 0.883, por encima de ER-ACE
  (0.855). La secuencia le cuesta 7.4 puntos; a ER-ACE, 1.2.
- **Es la lectura lenta, que se inclina hacia lo reciente.** Los rasgos de V1 no derivan: la sonda
  da 0.852 al aprender y 0.847 al final. La lectura sobre esos rasgos cae de 0.967 a 0.717, y el
  82 % de sus errores va a clases más nuevas.

**El arreglo, `ace`.** Para lo que llega, la competencia de la lectura se resuelve solo entre las
clases presentes. Es la versión local de la entropía cruzada enmascarada de ER-ACE. Se eligió en
validación y se midió una sola vez en test.

### Aprender a aprender (Thinker, 3 semillas)

Problemas al estilo Harlow: en cada uno aparecen 2 objetos nunca vistos, uno premiado, 6 ensayos.
El Thinker no ve el resultado como percepción: sólo recibe, como parte de su estado, cuál fue su
última respuesta y si fue premiada (`prev_io`). Después de entrenar, **se congelan los pesos** y se
mide en problemas nuevos.

| Problemas de entrenamiento | 10 | 25 | 50 | 200 |
|---|---|---|---|---|
| Acierto en el ensayo 2, pesos congelados | 0.52 ± 0.03 | 0.90 ± 0.07 | 1.00 ± 0.00 | 0.82 ± 0.13 |

Con los pesos fijos resuelve problemas nuevos en un ensayo: el aprendizaje rápido ocurre en el
estado, no en los pesos (como propone Wang et al. 2018 para la corteza prefrontal). Con 3 objetos
acierta 0.35, 0.66 y 0.82 en los ensayos 1 a 3, el óptimo de una memoria de un ensayo (0.33, 0.67, 0.83). Una reversión nunca vista se
corrige en un ensayo (0.83 en el ensayo siguiente). Control: si las dos señales entran por separado y no conjugadas,
queda en azar (0.50).

Mapa relacional: con la acción RETROCEDER (`back`) la relación inversa pasa de 0 % a 100 %
(pesos congelados, también con 8 elementos y cada relación vista una sola vez).

### Reglas que no se pisan, mapa del episodio y neuromoduladores (Thinker, 3 semillas)

Cada mecanismo es una opción apagada por defecto. Pesos congelados y problemas nuevos, salvo
aviso; configuración elegida en validación y medida una vez con otras semillas.

| Prueba | Sin el mecanismo | Con el mecanismo |
|---|---|---|
| Volver a una regla sin consigna después de 400 problemas de otra (`task_sets`, como en Collins & Frank 2013) | 0.08 ± 0.11 | 1.00 ± 0.00 |
| Batería C0–C6 mezclada, 12 medidas (`back`: los predecesores se ven sólo al retroceder) | `back` anterior: peor medida 0.72 (orden de 8, distancia ≥ 4) | las 12 en 1.00 |
| Harlow sin ver el resultado, peor instantánea por semilla desde el problema 50 (neuromoduladores) | 0.75 ± 0.05, 4 instantáneas bajo 0.95 | 1.00 ± 0.00, ninguna |
| Cierre de un anillo con relaciones mezcladas con preguntas (`cogmap` + neuromoduladores) | 0.67 ± 0.47 | 1.00 ± 0.00 |
| Dos regímenes, estable y volátil: elige el premiado | valores a mano 0.773 ± 0.005; los 7 ajustados por perturbación 0.50 ± 0.36 | 0.828 ± 0.010 (oráculo por régimen: 0.831 ± 0.010) |

Qué explica cada mejora, según las ablaciones:
- En los dos regímenes, toda la mejora viene de la memoria bayesiana: con ella sola y el control
  con valores fijos da el mismo 0.828. Los neuromoduladores no suman ahí.
- En Harlow, la mejora viene del horizonte de un ensayo que fija la serotonina: sin él, 0.49 ± 0.41.
  La causa de las caídas era el error de predicción de fin de episodio.
- En el anillo, los neuromoduladores bajan la relación inversa de 1.00 a 0.82.
- Mientras aprende (en línea, bloques 750–799), el cierre queda en 0.55 ± 0.02 en el anillo y en
  0.76 ± 0.03 en la línea con final.

### Percepción + razonamiento

| Prueba | Resultado |
|---|---|
| Percepción entrenada aparte, Thinker aprende a sumar sobre lo percibido (3 dígitos) | 71.9 % exacta (76.1 % con crédito simbólico); con símbolos perfectos, 97.4 % |
| L0: identidad aprendida sólo con el crédito del Thinker | pureza de la percepción 88–90 % |
| **L3: suma completa aprendida desde cero, todo junto** | **1.5 % exacta (0–4 %), 24 % por columna (azar: 10 %)** |
| **Suma con imágenes, regla conocida (MNIST-addition): crédito conjunto entre lo que ve y la suma; 27 500 pares, una pasada, test, 3 semillas** | **97.6 ± 0.2 % de sumas, 98.8 ± 0.1 % por dígito.** Error del dígito: AUROC 0.984, ECE 0.002. Publicado, con 30 000 pares y backprop: DeepProbLog 97.2 ± 0.5, A-NeSI 97.66, DeepStochLog 97.9 (citado en la tabla 1 de A-NeSI). Queda por encima de DeepProbLog, a la par de A-NeSI y por debajo de DeepStochLog |
| Atajo de razonamiento con 3 000 pares (8 semillas, test) | sin la compuerta de fiabilidad caen 4 de 8 (dígito ≈ 0.01); con ella, 0 de 8: suma 93.8 ± 0.6 %, dígito 96.8 ± 0.3 %. DeepProbLog con 3 000 pares, otro protocolo: 92.2 ± 1.6 |
| L3 por etapas: categorías sin etiquetas y después ligadura, regla desconocida (validación, 1 semilla) | las categorías llegan a una pureza de 0.79 (hace falta ≥ 0.965); después de ligar, símbolos 0.33 y 0.2 % exacta |
| **L3 con 10 imágenes nombradas por dígito y la regla desconocida** (test, 3 semillas, 8 000 episodios) | **75.7 ± 20.9 % exacta** (87.6, 88.0 y 51.6 %); pureza de los símbolos 0.950; con 4 000 episodios, 67.6 ± 20.0 %. Con 0, 1, 3 y 5 nombradas no llega (validación: 0.2, 0.0, 6.4 y 0.4 %). En la semilla floja, la regla aprendida da 73.8 % aun con los dígitos correctos |
| El Thinker sabe cuándo se equivoca al razonar (3 semillas) | AUROC de sus errores de cálculo 0.923 con percepción perfecta y 0.831 con percepción de 80 %; de los símbolos mal percibidos que recibe, solo 0.627. ECE calibrada 0.04 |

### Saber cuándo se equivoca (`Cortex` con `metacog`, flujo continuo, 3 semillas)

| Medida | Escala 0.05 | Escala 0.1 |
|---|---|---|
| AUROC del propio error, confianza calibrada | 0.855 ± 0.003 | 0.885 ± 0.001 |
| ECE calibrada | 0.014 ± 0.001 | 0.012 ± 0.001 |
| Apuesta por confianza (Kepecs 2008): AUROC acierto contra error | 0.845 ± 0.017 | 0.870 ± 0.018 |
| Pendiente de la apuesta contra la facilidad: aciertos / errores | +0.26 / −0.17 | +0.23 / −0.11 |
| Entre los errores, nuevo contra conocido: señal NE | 0.851 ± 0.018 | 0.890 ± 0.020 |

- **Firma de Kepecs:** reproduce la que se midió en ratas. En los aciertos, la apuesta sube con la
  facilidad; en los errores, baja.
- **Error contra novedad:** la confianza no distingue un error en algo conocido de uno ante algo
  nuevo (0.52). La señal NE sí los distingue.
- **Hipercorrección:** no tiene efecto propio. Lo que acelera la corrección (de 102.5 a 82.7 pasos)
  es aplicar más plasticidad en los errores. El control con la misma ganancia, sin ponderar por
  confianza, da lo mismo (82.0).

### Rendimiento

El índice invertido exacto de la capa dispersa es idéntico bit a bit a la pasada densa y da entre
1.2 y 3 veces más velocidad. Una imagen con 2 niveles tarda ~4 ms en un núcleo de Python/NumPy.

## Comparación con referencias publicadas

La batería (`experiments/battery/`, `python3 experiments/exp_battery.py correr`) mide cada
prueba contra una referencia y marca si la comparación es **equivalente**: misma entrada y mismo
protocolo que el estudio. El índice es (red − azar) / (referencia − azar): 1 iguala la referencia
y 0 es azar. Las referencias son cifras publicadas, salvo donde dice «línea base local», que es
otro modelo corrido acá con los mismos datos. «R3» son las opciones de razonamiento 3 encendidas:
`prev_io`, `back`, `cogmap` y `modulators`.

Hoy solo 3 de 10 dimensiones tienen una comparación equivalente con personas o animales. Por eso
no hay un «porcentaje humano» global: un promedio no sería informativo. El perfil por dimensión,
solo con las equivalentes y las opciones por defecto, da:

- percepción 0.84;
- aprender de pocos ejemplos 0.34 (0.55 con R3);
- metacognición dentro del rango humano.

Los archivos son `results/bateria_dbc5d4bd76bf.*`.

**Equivalentes** (3 semillas, salvo aviso):

| Prueba | Red | Referencia | Índice |
|---|---|---|---|
| MNIST, 2 niveles | 98.9 % | personas 99.8 % (Keysers 2007) | 0.99 |
| Fashion-MNIST, 2 niveles | 89.5 % | personas 83.5 % (Zalando) | 1.08 |
| CIFAR-10, 2 niveles; 5 000 imágenes de entrenamiento de 50 000 y las primeras 5 000 del test | 48.1 ± 0.4 % | personas 95.2 % (CIFAR-10H) | 0.45 |
| Omniglot: un solo ejemplo, 20 opciones; imágenes de 28 × 28 en lugar de 105 × 105, lector de la batería sobre V1 | 28.8 ± 0.9 % | personas ≥ 95.5 % (Lake, Salakhutdinov & Tenenbaum 2015) | ≤ 0.26 |
| Harlow con objetos vistos, misma cantidad de problemas que los monos | 0.72 ± 0.04; con R3, 0.94 ± 0.02 (el techo es ~0.94 por colisiones de la percepción) | monos rhesus: 0.52 en los primeros 8 problemas (Harlow 1949) y 0.82 en los problemas 33–132 (Harlow 1950). En esos tramos la red da 0.50 y 0.70 por defecto, y 0.79 y 0.94 con R3 | — |
| Saber cuándo se equivoca: AUROC de tipo 2 en una decisión visual con dificultad igualada | 0.748 ± 0.007 | personas 0.55–0.75 (Fleming et al. 2010) | dentro del rango, en el borde superior |
| Igual/distinto con figuras nuevas; «igual» es una copia idéntica, como con los animales | 0.85 ± 0.03 (azar 0.5); con la copia corrida 2 px, 0.68; con dos dibujos del mismo carácter, 0.62 | con 8 figuras, rhesus, capuchinos y palomas no transfieren (Katz, Sturz & Wright 2010) | — |
| MNIST-addition: 27 500 pares, una pasada | 97.6 % | DeepProbLog 97.2 ± 0.5; A-NeSI 97.66; DeepStochLog 97.9 (citado en la tabla 1 de A-NeSI); todos con backprop | — |
| Split-MNIST por clases, una pasada | 96.6 %; 97.0 % con `ace` | publicado, con otros presupuestos de memoria y ~21 pasadas por tarea (van de Ven, Tuytelaars & Tolias 2022): entrenamiento conjunto 98.2, BI-R 94.4, iCaRL 92.5, ER 88.8, EWC 20.6. Línea base local con los mismos bytes: ER-ACE 95.0 | — |
| Split-Fashion por clases, una pasada | 80.9 %; 83.3 % con `ace` | sin cifra publicada con un protocolo comparable. Línea base local con los mismos bytes: ER-ACE 84.3, ER 85.2 | — |
| Texto no visto, 600 KB de habla dirigida a niños; 1 corrida, porque la red es determinista dado el corpus | 2.96 bits/byte (2.54 con la temperatura elegida en validación) | línea base local: n-grama Kneser-Ney de orden 8 con los mismos datos, 1.43 | — |
| Curva de escalado del texto: 0.2, 0.6 y 1.8 MB, últimos 60 000 bytes del corpus | 3.97, 3.68 y 3.75 bits/byte | línea base local, Kneser-Ney con los mismos datos: 2.13, 1.97 y 1.88. Brecha pareada 1.84 [1.82; 1.87], 1.70 y 1.86 (IC 95 % por bootstrap de bloques) | — |

En igual/distinto, la red transfiere la identidad de los píxeles, no un concepto robusto. En Harlow,
el salto con R3 lo aporta la entrada explícita del ensayo anterior (`prev_io`), no un aprendizaje
lento como el del mono.

**No equivalentes:** otra entrada u otro procedimiento. No se promedian con las de arriba.

| Prueba | Red | Referencia | Por qué no es equivalente |
|---|---|---|---|
| Harlow con símbolos | 0.76 ± 0.05; con R3, 1.00 | — | símbolos en lugar de objetos vistos |
| Inferencia transitiva, serie de 5 (par BD) | 0.50 ± 0.50 | palomas 0.875 (von Fersen et al. 1991) | ídem |
| Inferencia transitiva, serie de 6 | 0.25 ± 0.22 | personas 0.743 (Siemann & Delius 1996) | ídem |
| Inferencia transitiva, serie de 7, fase 3 (los 21 pares, con premio) | 0.961 ± 0.011 | sin cifra verificada (Jensen et al. 2017) | ídem |
| Más tasa de aprendizaje cuando el ambiente es volátil | 0.013 ± 0.079: sin efecto | personas: mayor en la fase volátil (Behrens et al. 2007) | ídem |
| Detección de cambio, capacidad K | 5.7 ± 4.3, bimodal; con R3, 8.5 ± 3.2 | personas 2.46; rhesus 0.71 | la memoria es exacta por diseño |
| Reversión serial | 4.0 errores por reversión (azar 70.4), sin mejora entre reversiones | sin cifra verificada | símbolos |
| Igual/distinto con símbolos | 0.50 | — | símbolos |
| MLC: instrucciones con pocos ejemplos | 0.00 | personas 0.807 (Lake & Baroni 2023) | símbolos en lugar de palabras y colores |
| SCAN, experimento 1: 1, 2, 4 y 8 % de los comandos | 0.00 en las cuatro | mejor seq2seq: ~0.05, ~0.54 y ~0.93 con 1, 2 y 4 % (Lake & Baroni 2018) | 1 semilla; se compara con un sistema |
| bAbI 1k, 20 tareas | error medio 52.8 % (pasa 2) | LSTM 51.3 % y MemN2N 13.9 % (Sukhbaatar et al. 2015); MemNN con hechos de apoyo, ~7 % (Weston et al. 2015) | 1 semilla; se compara con sistemas |
| BLiMP con 600 KB | 0.482, por debajo del azar (p = 0.04) y del n-grama de bytes (p = 0.014); la causa principal es la longitud | línea base local con los mismos datos: n-gramas 0.505–0.511. Personas 0.886 (Warstadt et al. 2020) | a esta escala ningún modelo supera al azar |

## Dónde falla

- **Aprender a ver y a razonar a la vez desde cero (L3) no funciona.** Por separado anda: la
  percepción supervisada y el Thinker con símbolos perfectos. Juntos, con el resultado final como
  única señal, el crédito no se reparte entre las imágenes y la regla.
  - **Con la regla conocida, el crédito conjunto sí funciona** (97.6 % en MNIST-addition).
  - **Con la regla desconocida y sin ningún ejemplo nombrado, tampoco funciona por etapas.** Las
    categorías sin etiquetas no son lo bastante puras (0.79 contra 0.965), y ligarlas a símbolos
    no llega a la cota.
  - **El mecanismo de ligadura se aplana:** con 3 prototipos por dígito da 0.60, 0.73 y 0.71 con
    10 000, 40 000 y 160 000 columnas. Es un límite de diseño, no de datos.
  - **Salir de un atajo ya consolidado es lento:** la identidad llega a 0.47 en 7 500 pares.
  - **Con 10 imágenes nombradas por dígito sí funciona, pero no siempre:** 75.7 ± 20.9 % en test
    (2 de 3 semillas pasan el 80 %). Lo que decide es cuán buena es la percepción cuando empieza
    el aprendizaje conjunto. Medido en el Cortex: desde cero, enseñar una imagen generaliza igual
    a su dígito que a los demás (g = h = 1); con 10 anclas, g = 0.335 contra h = 0.153.
- **La profundidad no suma: la pila de áreas hacia adelante es el diseño equivocado.**
  - Con una pasada y 5–10k muestras, tampoco suma en una CNN con backprop: en ese régimen la
    prueba no discrimina.
  - Con el presupuesto completo, la segunda capa de la CNN sí gana: +1.1 puntos en Fashion y +4.5
    en CIFAR-10. Ahí nuestro V2 queda 2.7 puntos debajo de V1 en Fashion (1 semilla), y una sonda
    ridge fuera de línea da lo mismo. La información no está en V2; no es la lectura.
  - Probados sin éxito: ancho 256–2048, decorrelación, predicción descendente con residuo, regla
    de traza con micro-sacadas, CLAPP y WTA blando tipo SoftHebb (este, además, 17 veces más
    lento con 1024 neuronas).
  - En CIFAR-10 con 45k imágenes, el V2 por defecto terminó con código nulo: es un fallo numérico
    sin diagnosticar.
  - Nuestro V1 queda por encima de la capa 2 de la CNN en la sonda lineal: 0.906 contra 0.893 en
    Fashion.
  - Con el diccionario lleno, la poda solo libera rasgos jóvenes: la neurogénesis queda detenida
    para siempre, y la elasticidad muere.
- **Olvido de clases parecidas.** En Split-Fashion, aun con `ace`, la red queda 1 punto debajo de
  ER-ACE y olvida más: 0.152 contra 0.092. La causa medida es el sesgo de la lectura lenta hacia
  lo reciente, no la capacidad ni los rasgos. Falta medir `ace` en el flujo de 200 000 pasos.
  Probados sin ganancia:
  - compuertas de recuerdo;
  - reconsolidación por error de predicción (0.764 contra 0.775 con la consolidación fija);
  - autoexamen y espaciado;
  - sueño en microfases: baja la latencia máxima de 2 107 a 74 ms, pero no mejora la exactitud, y
    liberar neuronas es catastrófico (0.554);
  - todo junto: 0.723, peor que lo mejor de cada fase.
- **Costo.** Con 2 niveles, entrenar tarda entre 16 (MNIST) y 30 (Fashion) veces más que la CNN de
  referencia, y usa más sinapsis que un MLP.
- **Desplazamientos.** La CNN de 2 capas tolera mejor un desplazamiento de 3 px (56.0 % contra
  51.1 % en Fashion).
- **Texto.** Predice peor que un n-grama de Kneser-Ney con los mismos datos (línea base local)
  en las tres escalas medidas: la brecha es de 1.84, 1.70 y 1.86 bits por byte con 0.2, 0.6 y
  1.8 MB. Ya en 0.2 MB, sin llegar al tope de neuronas, la brecha es la misma, y de 0.6 a 1.8 MB
  no mejora (con 30 000 neuronas, el tope, y 50 628 desalojadas). Es evidencia contra el
  mecanismo de secuencias (criterio F). Falta la serie con el tope triplicado para descartar la
  capacidad.
- **Razonamiento composicional con benchmarks externos: cero.** SCAN, MLC y la mitad de bAbI
  fallan (ver la tabla de comparación). Las causas medidas son de diseño:
  - no ajusta ni los comandos que vio;
  - casi nunca decide terminar la respuesta;
  - guarda prototipos de comandos enteros, no significados por palabra;
  - pierde el orden de las palabras.

  La curva es plana entre 1 y 8 % de los datos. El Thinker razona sobre tareas formales chicas;
  no hay evidencia de razonamiento en lenguaje natural ni en dominios abiertos.
- **Aprender a aprender tiene límites medidos.** Ajustar por recompensa los meta-parámetros de la
  memoria rápida no escala: con 7 a la vez rinde peor que los valores elegidos a mano (0.50 ± 0.36).
  Lo que mejora los dos regímenes es la memoria bayesiana, no ese ajuste. Entre familias de reglas
  opuestas no hay aceleración. Sin las opciones nuevas, la regla nueva **sobrescribe la vieja**
  y el anillo no cierra. Con ellas, la vieja vuelve a 1.00 y el anillo cierra con los pesos
  congelados, pero mientras aprende el cierre queda en 0.55.
- **Todo encendido a la vez todavía no funciona.** Por eso las opciones de arriba quedan apagadas
  por defecto. Con todas encendidas (rama `agent/razonamiento3`, 3 semillas):
  - En la multitarea en bloques en orden inverso, C1 cae de 1.00 a 0.52, C4 a 0.68, C2 a 0.78 y
    C0 a 0.85: lo compuesto antes de aprender f no se repara cuando f llega.
  - C1 con 256 símbolos queda en 0.805 y 0.835 (1–2 y 3–4 saltos). La v0.3 da 0.835 y 0.888.
  - Con `task_sets`, fallan tres pruebas de multitarea con consignas. Volver a una tarea vieja
    queda entre 0.0 y 0.75.
  - Cada paso tarda 2.09 veces lo de la v0.3; el tope fijado es 2.
- **Selección de hiperparámetros.** Parte de la exploración previa miró subconjuntos del test de
  MNIST y Fashion: esas cifras son algo optimistas. Los resultados de la v0.3 usan validación
  separada.
- **Escala.** La mayor instancia medida tiene 10 millones de neuronas. La escala humana es una
  extrapolación de costo (~0.9 PB), no un experimento.

## En curso: mecanismos del aprendizaje humano

Cada mecanismo se implementa como opción del código, con tests, ≥3 semillas y ablación, y al final
se prueban todos juntos.

| Cómo aprende un humano | Fuente | Rama |
|---|---|---|
| Predicción de arriba hacia abajo; sube el error residual; plasticidad por error en cada nivel | Rao & Ballard 1999 | **medido**: subir el residuo empeora (Fashion −4.4 puntos); modular la plasticidad con el error no gana al control barajado. El error de predicción sí detecta lo nuevo (AUROC 0.81 en V2) |
| Detección del propio error por conflicto entre respuestas | Yeung, Botvinick & Cohen 2004 | **medido**: el conflicto predice el error (AUROC 0.874) |
| Los errores cometidos con mucha confianza se corrigen más | Butterfield & Metcalfe 2001, 2006 | **medido**: sin efecto propio |
| Incertidumbre esperada (ACh) e inesperada (NE) modulan inferencia y plasticidad | Yu & Dayan 2005 | **medido**: NE separa error de novedad (0.89) |
| Aprender pares palabra-objeto acumulando evidencia de escenas ambiguas | Smith & Yu 2008 | **medido**: con la regla conocida liga (97.6 %); sin ancla no |
| Etapas: "uno", "dos", "tres" y después el principio cardinal | Carey; Le Corre; Elman 1993 contra Rohde & Plaut 1999 | **medido**: contar primero los números chicos rinde peor que el orden aleatorio (pureza 0.30 contra 0.43) |
| Dos sistemas: memoria rápida y conocimiento lento intercalado | Kumaran, Hassabis & McClelland 2016 | **medido**: con el sistema lento aprendiendo de su propio error, 0.791 contra 0.784 de ER-ACE en clases parecidas; las compuertas de recuerdo empeoran |
| Lo que encaja en un esquema se consolida en un ensayo | Tse et al. 2007 | **medido**: asimila rápido solo con un esquema que encaja (0.81 contra 0.68 con uno inconsistente); los ítems en conflicto dañan lo viejo (−0.21) |
| Un recuerdo consolidado sólo se modifica si hay error de predicción | Sevenster, Beckers & Kindt 2013–2014 (y la replicación fallida de 2022) | **medido**: actualiza rápido sin romper vecinos, pero queda debajo de la consolidación fija (0.764 contra 0.775) |
| Recordar enseña más que releer; el espaciado retiene mejor | Roediger & Karpicke 2006 | **medido**: sin ganancia contra el repaso pasivo con el mismo presupuesto |
| Sueño: repaso activo y renormalización sináptica global | Klinzing, Niethard & Born 2019 | **medido**: en microfases baja la latencia máxima de 2 107 a 74 ms, sin mejorar la exactitud |
| Inferir la causa de un cambio antes de actualizar (cambió el nombre, o es algo nuevo) | Gershman et al. 2017 | **medido**: las causas latentes llevan el cambio de etiqueta de 0.118 a 0.528 y el renombre de 0.395 a 0.865, a costa de 0.04 en clases viejas parecidas |
| La dopamina entrena a la prefrontal para que sea un aprendiz rápido propio | Wang et al. 2018 | **medido** (ver resultados) |
| Mapas cognitivos: estructura relacional reutilizable | Behrens et al. 2018 | **medido** (ver resultados) |
| Las reglas se agrupan en conjuntos de tareas reutilizables; una causa latente decide cuál rige | Collins & Frank 2013 | **medido** como opción; con todo encendido, abierto |
| Los neuromoduladores fijan los meta-parámetros del aprendizaje | Doya 2002 | **medido** como opción; en los dos regímenes no suma |

## Hoja de ruta hacia el aprendizaje humano

Nadie tiene hoy un camino demostrado hasta la inteligencia humana con aprendizaje local y
online: es un problema abierto. Esta hoja de ruta ordena el trabajo por dependencias, mide
cada escalón contra un benchmark externo y fija de antemano qué resultado obligaría a cambiar
el diseño.

**Regla de diseño** (sale de lo medido). Antes de cambiar un peso, inferir qué causó el
resultado —qué módulo, qué recuerdo, qué causa latente, error o novedad— y aplicar la
plasticidad local en proporción a esa responsabilidad. Lo que funcionó la cumple: error local
por área, crédito conjunto en la suma con imágenes, conjuntos de tareas, búsqueda
contrafáctica que separa estrategia de conocimiento. Lo que falló la viola: crédito casilla
por casilla, reconsolidación ante cualquier error, compuertas gobernadas por una confianza
sin calibrar.

| Escalón | Qué tiene que lograr | Medida externa | Estado |
|---|---|---|---|
| A. Saber cuándo se equivoca | confianza calibrada; separar error de novedad | apuesta por confianza (ratas, Kepecs 2008); ECE ≤ 0.05; AUROC ≥ 0.85 | cumplido en el flujo continuo de `Cortex`, que no es equivalente: AUROC 0.855, 0.886 y 0.909 en tres escalas; ECE 0.010; firma de Kepecs; NE separa error de novedad (0.89). Con el protocolo equivalente de Fleming da 0.748, dentro del rango humano (0.55–0.75), donde el umbral de 0.85 no aplica. En el Thinker: AUROC 0.83 de sus errores de cálculo, 0.63 de los símbolos mal percibidos que recibe |
| B. Símbolos anclados en lo que ve | crédito conjunto entre percepción y razonamiento | MNIST-addition (DeepProbLog: 97.2 % con 30 000 pares); después con la regla desconocida | regla conocida: 97.6 ± 0.2 % en test (3 semillas, una pasada), a la par de lo publicado. Regla desconocida: sin ancla no funciona; con 10 imágenes nombradas por dígito, 75.7 ± 20.9 % (2 de 3 semillas ≥ 80 %) |
| C. Abstracción por niveles | que cada nivel aporte; invariancia | CIFAR-10 contra Hebb local publicado (SoftHebb: 80.3 %, con otro protocolo); sonda por nivel; desplazamientos | criterio de abandono **no cumplido** en el régimen válido. Con todo el presupuesto de una pasada, el backprop sí gana con la segunda capa (Fashion +1.1, CIFAR-10 +4.5 puntos); nuestro V2 queda 2.7 puntos debajo de V1 en Fashion (1 semilla), y la ridge fuera de línea da lo mismo (−3.0). Ningún objetivo local probado le da información propia: ancho, decorrelación, residuo, traza, CLAPP, WTA blando. Aprobado para la próxima sesión: un área recurrente de atención en lugar de V2/V3 |
| D. Memoria sin interferencia | actualizar o crear según la causa inferida | Split-MNIST y Split-Fashion incremental por clases; ER-ACE con los mismos bytes | Split-MNIST: 97.0 %, por encima de ER-ACE (95.0). Split-Fashion: 83.3 % con `ace`, 1 punto debajo de ER-ACE (84.3), y olvida más (0.152 contra 0.092). La causa es el sesgo de la lectura lenta, no la capacidad (curva plana de 1 000 a 8 000 neuronas) |
| E. Razonamiento composicional | generalizar a combinaciones nuevas | SCAN, bAbI; datos de personas de Lake y Baroni 2023 | medido, falla: SCAN 0 %, MLC 0 %, bAbI 52.8 % de error. Modo serial (rama `agent/composicion`, opción apagada): aprende el léxico y a decir FIN; en una gramática de juguete propia generaliza a comandos nuevos (84.8 ± 2.6 %), en SCAN al 1 % todavía no (0–10 % en validación, 1 semilla) |
| F. Lenguaje | predecir y componer sobre secuencias | BabyLM (10 M de palabras, escala de un niño) a escala reducida | medido en 0.2, 0.6 y 1.8 MB: la red queda 1.7–1.9 bits/byte por encima de un n-grama con los mismos datos en las tres escalas. Ya en 0.2 MB, sin llegar al tope de neuronas (18 857 de 30 000), la brecha es 1.84, y de 0.6 a 1.8 MB no mejora (tope alcanzado). Criterio F: no supera a los n-gramas; falta la serie con tope triple para separar capacidad de mecanismo |
| G. Razonar con neuronas | memoria de trabajo y ligadura en neuronas de spikes, no en estructuras programadas | las pruebas de E, sin pérdida | sin empezar |
| H. Actuar y explorar | elegir qué mirar y qué preguntar; metas propias | pruebas de cognición animal (igual/distinto, inferencia transitiva, Harlow); BabyAI | las pruebas animales ya se miden en la batería (ver la comparación); el mecanismo de actuar y explorar, sin empezar |

**Hardware o arquitectura.** Una capacidad sólo se da por «limitada por el hardware» si su
curva de escalado (exactitud contra neuronas y datos, en 3 escalas o más) sigue subiendo. Si
se aplana, el límite es de diseño y hay que corregirlo.

**Criterios de abandono.**
- C: si con ancho creciente por nivel y aprendizaje de invariancia por continuidad temporal la
  sonda del nivel 2 no supera a la del 1, la pila de áreas hacia adelante es el diseño
  equivocado y se reemplaza.
- F: si a escala reducida no supera a un modelo de n-gramas con los mismos datos, el mecanismo
  de secuencias es el equivocado.
- G: si al pasar la memoria de trabajo a neuronas se pierde el razonamiento ya logrado, el
  Thinker es un programa con control aprendido y no una red.

## Estado de las ramas y cómo continuar

`main`, `master` y `v0.3-dev` tienen todo lo integrado. Lo nuevo de cada rama entra como opción
apagada por defecto. Con la configuración de siempre, la percepción da lo mismo bit a bit que la
medida por la batería (`experiments/battery/identidad.py`), y el Thinker no cambió. Por eso las
cifras de la comparación valen para este `main`. Las ramas de trabajo también están publicadas.
Todo se corre con `OMP_NUM_THREADS=1`; las corridas largas se reanudan con `--skip-existing`.

| Rama | Último commit | Estado | Qué sigue |
|---|---|---|---|
| `agent/composicion` | 89da0ae | **no fusionada**: se apoya en `agent/razonamiento3` (10eed10), que enciende por defecto opciones con regresiones (F5). Sus resultados están en `results/composicion_*` | Que el control fije lo que halla la búsqueda: imitar la trayectoria entera alineada en retrospectiva. Después la curva del experimento 1 de SCAN con 3 semillas, MLC, add-jump y length, bAbI, y las regresiones de F5 (`back` primero, en bloques en orden inverso) |
| `integracion3` | 246cf9b | fusionada | Tercer punto de presupuesto en la curva de anclas (16 000 episodios). Generalización por dígito antes de ligar (g/h ≫ 2, medido con `f0g`). Una compuerta que solo deje aprender la regla con percepción confiable |
| `agent/consolidacion` | 276ff1b | fusionada | Medir `ace` en el flujo de 200 000 pasos, en clases parecidas, en cadenas y en cambio de etiqueta (fases `f7b`, `f7x`, `f7s`, `f7k`, `f7c`); el punto de capacidad sin tope (`f6cap0`) |
| `agent/predictivo` | 0a535bb | fusionada | **Área recurrente de atención en lugar de V2 y V3** (aprobada). Criterios: sonda ≥ V1 + 1 punto, red ≥ profundidad 2 + 0.5, 3 px en Fashion ≥ 56 %, latencia ≤ 2×. Diagnosticar el V2 nulo en CIFAR-10 con 45k. Que la poda libere rasgos viejos poco usados |
| `agent/medicion` | d77a60d | fusionada | Serie de lenguaje con el tope de neuronas triplicado (decide el criterio F). Pruebas de abejas con cifras verificadas: Howard 2018, Avarguès-Weber 2012, Giurfa 2001 en el original. Repetir la batería si cambia `bioelastic/` |
| `agent/razonamiento3` | 10eed10 | base de `agent/composicion` | se continúa en `agent/composicion` |

Sin empezar: el escalón G (el Thinker en neuronas de spikes) y el reemplazo del mecanismo de
secuencias del escalón F, si la serie de tope triple confirma el criterio.

Comandos para retomar:

```bash
# composición (escalón E), desde la rama agent/composicion
python3 -I experiments/exp_composicion.py scan_imitate      # imitación en validación (cambiar gains)
python3 -I experiments/exp_composicion.py mini              # mini-SCAN, 3 semillas
python3 -I experiments/exp_composicion.py scan_exp1         # curva 1-2-4-8 %; toca el test una vez
# anclas (escalón B)
python3 experiments/exp_integration3.py f1a --seed S --variante 10 --val --ep 16000 --skip-existing   # S = 0, 1, 2; después sin --val
# consolidación (escalón D)
cd experiments && for ph in f7b f7x f7s f7k f7c f6cap0; do python3 exp_consolidation.py $ph; done
# jerarquía (escalón C): techo con más semillas y diagnóstico del V2 de CIFAR-10
cd experiments && python3 exp_predictive.py ceiling fashion --seeds 1,2 --val 2000 --tag f7f --views 2
cd experiments && python3 exp_predictive.py run cifar10 --depths 3 --seeds 0 --train 45000 --val 2000 --test 0 --perturb 0 --tag f7g
# batería: lenguaje con el tope triple e informe
python3 experiments/exp_battery.py correr --commit 80a7870 --pruebas blimp__curva_escalado_tope_triple --minutos 100 --skip-existing
python3 experiments/exp_battery.py informe --commit <sha de main>
```

Los datos externos no están versionados; las fuentes están en el docstring de
`experiments/exp_external.py` y en `experiments/battery/catalogo.json`:

- SCAN, MLC y bAbI, en `ext_data/`;
- AO-CHILDES, BLiMP, Omniglot y CIFAR-10H, en los datos de la batería.

## Escala: qué significa «capacidad humana»

| | Cerebro humano | Transformer abierto de referencia | Esta red |
|---|---|---|---|
| Unidades | 86 mil millones de neuronas (16 mil millones en la corteza) | Llama 3.1 405B: 126 capas × 16 384 de ancho | 10 millones instanciadas (medido) |
| Conexiones | ~1.5 × 10¹⁴ sinapsis en la neocorteza | 4.05 × 10¹¹ parámetros | 3.2 × 10⁸ sinapsis (medido) |
| Organización | 180 áreas por hemisferio, 6 láminas, ~10 niveles en la jerarquía visual | capas idénticas en serie | áreas apilables; probado hasta 3 sensoriales + asociación + Thinker |

Cada neurona guarda a lo sumo `fan_in` sinapsis y las columnas son independientes: el costo por
sinapsis medido es constante (2.4–2.9 ns) de 100 mil a 10 millones de neuronas.

![escala](results/scaling.png)

| Escala | Neuronas | Sinapsis | Memoria | Núcleo-segundos por paso |
|---|---|---|---|---|
| 10⁻⁴ | 1.6 M | 11.2 mil M | 89.6 GB | 30 |
| 10⁻² | 160 M | 1.12 × 10¹² | 8.96 TB | 3 000 |
| **1 (corteza humana)** | **16 mil M** | **1.12 × 10¹⁴** | **896 TB** | **303 000** |

Igualar la cantidad no iguala la capacidad. Lo que mide la profundidad útil de esta red es la
sección de resultados, no esta tabla.

## Uso

```bash
pip install -e ".[dev]"
pytest                                          # 307 tests
python experiments/exp_vision.py mnist          # idem: fashion, cifar10
python experiments/exp_fsdd.py                  # voz
python experiments/exp_continual.py             # flujo continuo no estacionario
python experiments/exp_interference.py          # clases parecidas
python experiments/exp_thinking_hard.py         # razonamiento: composición, cadenas
python experiments/exp_thinking_r2.py           # razonamiento: ablaciones y pruebas adversariales
python experiments/exp_integration2.py conjunta # percepción + razonamiento desde cero
python experiments/plot_v03.py                  # figura de esta página
```

```python
from bioelastic import Cortex, ImageEncoder
from bioelastic.thinker import Thinker

enc = ImageEncoder((28, 28), blur=0.4)
net = Cortex((28, 28, 1), n_outputs=10,                 # = "depth2" de exp_vision.py
             areas=[dict(rf=5, pool=2, max_features=256, theta=0.95, floor=0.3, k=8,
                         local=dict(lr=0.5))],            # error local del área
             heads=[dict(source="input")], projection=dict(source="all", balance=True),
             arbiter=0.01, replay=0.5, joint=False)
pred = net.step(enc(img).reshape(28, 28, 1), label)   # predice y después aprende

th = Thinker(n_symbols=20, memory=True, registers=2)
th.observe((3, 7)); out = th.answer((3,)); th.feedback(target=7)
```

## Datos

Todos públicos, descargados de repositorios públicos la primera vez que se usan:
MNIST ([fgnt/mnist](https://github.com/fgnt/mnist)),
Fashion-MNIST ([zalandoresearch/fashion-mnist](https://github.com/zalandoresearch/fashion-mnist)),
CIFAR-10 (copia binaria en [NTU-ALComLab/IWLS2021](https://github.com/NTU-ALComLab/IWLS2021)),
Free Spoken Digit Dataset ([Jakobovski/free-spoken-digit-dataset](https://github.com/Jakobovski/free-spoken-digit-dataset)),
tiny-shakespeare ([karpathy/char-rnn](https://github.com/karpathy/char-rnn)).

## Fuentes

- Neuronas: Azevedo et al. 2009 — https://pubmed.ncbi.nlm.nih.gov/19226510/
- Sinapsis: Pakkenberg et al. 2003 — https://pubmed.ncbi.nlm.nih.gov/12543266/
- Áreas corticales: Glasser et al. 2016 — https://pmc.ncbi.nlm.nih.gov/articles/PMC4990127
- Jerarquía visual: Felleman & Van Essen 1991
- K computer: RIKEN 2013 — https://www.riken.jp/en/pr/press/2013/20130802_1/
- Llama 3.1 405B (config): https://huggingface.co/neuralmagic/Meta-Llama-3.1-405B-Instruct-FP8
- Dopamina y error de predicción: Schultz, Dayan & Montague 1997 — https://pubmed.ncbi.nlm.nih.gov/9054347/
- Codificación predictiva: Rao & Ballard 1999 — https://pubmed.ncbi.nlm.nih.gov/10195184/
- Sistemas complementarios: Kumaran, Hassabis & McClelland 2016 — https://web.stanford.edu/~jlmcc/papers/KumaranHassabisMcClelland16FinalMS.pdf
- Esquemas: Tse et al. 2007 — https://pubmed.ncbi.nlm.nih.gov/17412951/
- Hipercorrección: Butterfield & Metcalfe 2006 — https://link.springer.com/article/10.1007/s11409-006-6894-z
- Detección de errores: Yeung, Botvinick & Cohen 2004 — https://www.ece.uvic.ca/~bctill/papers/learning/Yeung_etal_2004a.pdf
- ACh y NE: Yu & Dayan 2005 — https://princetonuniversity.github.io/NEU-PSY-502/_static/pdf/Class%208/Yu2005.pdf
- Reconsolidación: Sevenster, Beckers & Kindt 2014 — https://pmc.ncbi.nlm.nih.gov/articles/PMC4201815 · replicación fallida 2022 — https://dare.uva.nl/id/13529732-f331-4b02-9d1f-b520022508c3
- Efecto de testeo: Roediger & Karpicke 2006 — https://psychology.ecu.edu/wp-content/pv-uploads/sites/216/2019/03/Roediger-Karpicke-2006.pdf
- Sueño: Klinzing, Niethard & Born 2019 — https://www.nature.com/articles/s41593-019-0467-3
- Aprendizaje trans-situacional: Smith & Yu 2008 — https://pmc.ncbi.nlm.nih.gov/articles/PMC2271000
- Número: Le Corre & Carey; Piantadosi, Tenenbaum & Goodman 2012 — https://web.stanford.edu/%7engoodman/papers/COGNIT2374.pdf
- Empezar simple: Elman 1993 — https://web.cs.swarthmore.edu/~meeden/cogs1/s07/elman_cognition1993.pdf · Rohde & Plaut 1999 — https://www.cnbc.cmu.edu/~plaut/papers/pdf/RohdePlaut99Cog.startingSmall.pdf
- Prefrontal como meta-RL: Wang et al. 2018 — https://www.nature.com/articles/s41593-018-0147-8
- Mapas cognitivos: Behrens et al. 2018 — https://www.pubmed.gov/30359611/
- Suma con imágenes: Manhaeve et al., DeepProbLog — https://arxiv.org/pdf/1907.08194 · Dai et al. 2019, aprendizaje abductivo — https://proceedings.neurips.cc/paper/2019/hash/9c19a2aa1d84e04b0bd4bc888792bd1e-Abstract.html · atajos de razonamiento: Marconato et al. 2023 — https://arxiv.org/pdf/2302.01242
- Causas latentes: Gershman, Monfils, Norman & Niv 2017 — https://elifesciences.org/articles/23763v2
- Conjuntos de tareas: Collins & Frank 2013 — https://pmc.ncbi.nlm.nih.gov/articles/PMC3974273
- Neuromoduladores y meta-aprendizaje: Doya 2002 — https://doi.org/10.1016/S0893-6080(02)00044-8
- Mapa estructural: Whittington et al. 2020 — https://pmc.ncbi.nlm.nih.gov/articles/PMC7707106
- Inhibición lateral anti-Hebbiana: Földiák 1990 — https://link.springer.com/doi/10.1007/BF02331346
- Hebb local multicapa: Journé et al. 2023, SoftHebb — https://arxiv.org/abs/2209.11883v2
- Invariancia con plasticidad predictiva: Halvagal & Zenke 2023 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10620089/
- Escalado de algoritmos biológicos: Bartunov et al. 2018 — https://arxiv.org/abs/1807.04587
- Composicionalidad con meta-aprendizaje: Lake & Baroni 2023 — https://pmc.ncbi.nlm.nih.gov/articles/PMC10620072
- Lenguaje con datos de escala infantil: BabyLM — https://arxiv.org/html/2504.08165v1
- Prueba de Turing encarnada: Zador et al. 2023 — https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10033876/
- Confianza en ratas: Kepecs et al. 2008 — https://doi.org/10.1038/nature07200 · conceptos en abejas: Giurfa et al. 2001 — https://www.doi.org/10.1038/35073582
