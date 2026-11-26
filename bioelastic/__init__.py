"""SNN bio-elástica de aprendizaje online, agnóstica a la modalidad."""
from .encoders import AudioEncoder, ImageEncoder, LatencyEncoder, PopulationEncoder, TextEncoder
from .layer import ElasticLayer, Volley
from .network import BioElasticNet
from .neuron import first_spike_times
from .readout import Readout

__all__ = ["BioElasticNet", "ElasticLayer", "Readout", "Volley", "first_spike_times",
           "ImageEncoder", "AudioEncoder", "TextEncoder", "PopulationEncoder", "LatencyEncoder"]
__version__ = "0.1.0"
