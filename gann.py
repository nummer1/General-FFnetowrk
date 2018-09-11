import tensorflow as tf
import tflowtools as TFT

class GANN():
    def __init__(self, dims):
        self.dims = dims

    def build(self):
        self.dims[0]  # first layer
        self.dims[-1]  # last layer
