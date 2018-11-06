
import tensorflow as tf
from keras.engine.topology import Layer
from keras.engine import InputSpec

## Define reflection padding
class ReflectionPadding2D(Layer):
    """
    inputs:
                1. Layer: layer that needs to be padded
                2. padding: for input layer, the dim is (batch size, height, width, channels), and padding means how many pad
                        that you need for height and width dimensions.
    outputs:
                1. padded layer
    """
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 2 * self.padding[0], input_shape[2] + 2* self.padding[1], input_shape[3])

    def call(self, x, mask=None):
        w_pad, h_pad = self.padding
        return tf.pad(x, [[0, 0], [h_pad, h_pad], [w_pad, w_pad], [0, 0]], 'REFLECT')


class GradientPadding(Layer):
    """
    inputs:
                1. Layer: layer that needs to be padded
                2. padding: for input layer, the dim is (batch size, height, width, channels), and padding means how many pad
                        that you need for height and width dimensions.

    outputs:
                1. padded layer
    """
    def __init__(self, padding=((1, 1), (1, 1)), **kwargs):
        self.padding = tuple(padding)
        self.input_spec = [InputSpec(ndim=4)]
        super(GradientPadding, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + self.padding[0][0] + self.padding[0][1], input_shape[2] + self.padding[1][0] + self.padding[1][1], input_shape[3])

    def call(self, x, mask=None):
        left_pad, right_pad = self.padding[1]
        up_pad, down_pad = self.padding[0]
        return tf.pad(x, [[0, 0], [up_pad, down_pad], [left_pad, right_pad], [0, 0]], 'SYMMETRIC')
