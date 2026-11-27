"""Descripción de una prueba de la batería."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable


class NoAplicable(Exception):
    """La prueba no se puede correr hoy sin cambiar la arquitectura (o el commit medido no
    tiene la API que necesita). El motivo es el resultado."""


@dataclass
class Prueba:
    id: str                                   # clave en el catálogo
    variante: str                             # p. ej. "simbolos", "imagenes"
    condicion: str                            # "simbolos" | "imagenes" | "texto"
    metrica: str                              # qué es el puntaje principal
    protocolo: str                            # lo que hace la batería
    desviaciones: list                        # diferencias con el protocolo publicado
    correr: Callable                          # (semilla, escala) -> dict con "puntaje" y "medidas"
    version: int = 1                          # cambia si cambia el adaptador: invalida lo guardado
    semillas: tuple = (0, 1, 2)
    equivalente: bool = False                 # ¿la condición de entrada iguala a la del experimento?
    motivo_equivalencia: str = ""
    azar: float | None = None                 # None: el del catálogo
    azar_derivacion: str | None = None
    referencia: str | None = "catalogo"       # id de la referencia del índice; "catalogo": la del catálogo
    techo: float | None = None                # máximo posible de la métrica (para cotas)
    mayor_es_mejor: bool = True
    comparar: dict = field(default_factory=dict)      # id de referencia -> medida de la red con su métrica
    experiencia: dict = field(default_factory=dict)   # medida -> problemas que tuvo la red al medirla
    firmas: dict = field(default_factory=dict)        # medida booleana -> descripción de la firma
    curva: list = field(default_factory=list)         # [(tramo, medida de la red, id de referencia)]
    # Si la prueba admite más de una configuración documentada de la red, se elige una en
    # validación: `validar(config, semilla, escala)` mide sólo la fase de entrenamiento con
    # semillas de validación (nunca las de prueba ni la medida de prueba); gana el mayor
    # (o el menor, con `validar_mayor=False`). La prueba se corre una sola vez, con la elegida.
    configs: dict | None = None                       # nombre -> opciones de la red
    validar: Callable | None = None
    validar_mayor: bool = True
    semillas_validacion: tuple = (100, 101, 102)
    criterio_validacion: str = ""

    @property
    def clave(self) -> str:
        return f"{self.id}__{self.variante}"
