from tensorflow.keras.layers import Layer
import tensorflow as tf

class AggregationLayer(Layer):
    def __init__(self, **kwargs):
        super(AggregationLayer, self).__init__(**kwargs)

    def call(self, inputs):
        return tf.reduce_sum(inputs, axis=1)

    def get_config(self):
        config = super(AggregationLayer, self).get_config()
        return config
