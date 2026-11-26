"""SNN bio-elástica de aprendizaje online, agnóstica a la modalidad."""
from .area import FeatureArea
from .cortex import Cortex
from .encoders import AudioEncoder, ImageEncoder, LatencyEncoder, PopulationEncoder, TextEncoder
from .layer import ElasticLayer, Volley
from .network import BioElasticNet
from .neuron import first_spike_times
from .readout import Readout

__all__ = ["Cortex", "FeatureArea", "BioElasticNet", "ElasticLayer", "Readout", "Volley", "first_spike_times",
           "ImageEncoder", "AudioEncoder", "TextEncoder", "PopulationEncoder", "LatencyEncoder"]
__version__ = "0.2.0"
