import numpy as np
from . transform import TF
from . bandwidth import BandwidthLimiter
from . scale import LogScaler
from . quantize import Quantizer
from . image import Coder
from . cli import audio2img
from . eval import peaqb

class Processor(object):
    """Pipeline Processor for invertible deejaypeg Modules

    The processor takes a pipeline as a list of deejaypeg modules and
    allows to conveniently compute the forward and backward path.

    Parameters
    ----------
    pipeline : list of deejaypeg modules

    """
    def __init__(self, pipeline):
        super(Processor, self).__init__()

        # set up modules
        self.pipeline = pipeline

    def forward(self, input):
        output = input
        for module in self.pipeline:
            output = module(output, forward=True)
        return output

    def backward(self, input):
        output = input
        for module in reversed(self.pipeline):
            output = module(output, forward=False)
        return output
