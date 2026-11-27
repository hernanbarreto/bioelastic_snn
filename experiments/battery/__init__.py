"""Batería de medición: la red contra experimentos publicados con humanos y animales.

Cada prueba reproduce, tan cerca como se puede, el protocolo de un experimento publicado
y mide la red con la API pública (Thinker, Cortex, BioElasticNet, encoders). Las cifras de
referencia salen de `catalogo.json`, donde cada una tiene su fuente, su URL, su protocolo
y su nivel de verificación. Nada de la red se ajusta para pasar una prueba: el código que
traduce la tarea al idioma de la red vive acá.

  esquema.py         resultado común, índice normalizado, comparación con especies, perfil
  almacen.py         resultados por prueba y semilla (reanudable)
  version.py         medir cualquier commit: extrae `bioelastic/` de ese commit
  base.py            descripción de una prueba (`Prueba`) y del «no aplicable hoy»
  harlow.py          conjuntos de aprendizaje de Harlow (símbolos)
  reversion.py       reversión serial y reversión probabilística con volatilidad
  inferencia.py      inferencia transitiva (series de 5, 6 y 7)
  memoria_trabajo.py detección de cambio y capacidad K
  ingesta.py         resultados de otros agentes y de la propia rama (sólo lectura)

El índice normalizado es (red - azar) / (referencia - azar), como el puntaje normalizado a
humano de Atari, sin recortar. No hay etiquetas globales: el perfil es por prueba y por
dimensión, y el índice global (si se informa) es el promedio simple entre dimensiones.
"""
