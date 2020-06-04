from ddsp.core import linear_lookup, tf_float32
from ddsp import processors
from tensorflow.keras.layers import Layer, RNN
import tensorflow as tf


def delay_line(delay, audio):
    return tf.linalg.diag_part(linear_lookup(delay,audio))


class ProcessorCell(processors.Processor):
    
    def call(self, inputs: TensorDict, states) -> tf.Tensor:

        controls = self.get_controls(**inputs)
        signal = self.get_signal(**controls, states=states)
        return signal

    
class RNNProcessor(processors.Processor):
    
    def __init__(self, cell, name, trainable=False):
        self._layer = RNN(cell, return_sequences=True)
        super().__init__(name=name, trainable=trainable)
    
    def get_signal(self, **control):
        return self._layer(control)

    
class FeedbackDelayNetwork(RNNProcessor):
    
    def __init__(self, trainable=False, n_channels=3, max_delay=50, name="FDN"):
        super().__init__(FeedbackDelayNetworkCell(n_channels=n_channels, max_delay=max_delay, trainable=trainable),
                         name=name,
                         trainable=trainable)
    
    def get_controls(self, audio, delay=None, matrix=None):
        
        n_timesteps = audio.shape[1]
        if self.trainable:
            return {'audio': audio}
        # Repeat tensors along time if time dimension if not present.
        if len(delay.shape) == 2:
            delay = delay[:, tf.newaxis, :]
            delay = tf.repeat(delay, repeats=n_timesteps, axis=1)
        if len(matrix.shape) == 2:
            matrix = matrix[:, tf.newaxis, :]
            matrix = tf.repeat(matrix, repeats=n_timesteps, axis=1)
        
        return {'audio': audio, 'delay': delay, 'matrix': matrix}

    
class FeedbackDelayNetworkCell(ProcessorCell):
    
    """feedback delay network
    
    audio shape [n_channels]
    delay shape [n_channels]
    matrix shape [n_channels, n_channels]
    states shape [n_channels, max_delay]
    """
    
    def __init__(self,
               trainable=False,
               n_channels=3,
               max_delay=50,
               name='FDN_cell'):
        super().__init__(name=name, trainable=trainable)
        self._n_channels = n_channels
        self._max_delay = max_delay
        self.state_size = n_channels * max_delay
    
    def build(self, unused_input_shape):
   
        if self.trainable:
            initializer = tf.random_normal_initializer(mean=0, stddev=1e-6)
            self._matrix = self.add_weight(
                name='mixing_matrix',
                shape=[self._n_channels, self._n_channels],
                dtype=tf.float32,
                initializer=initializer)
            self._delay = self.add_weight(
                name='delays',
                shape=[1,self._n_channels],
                dtype=tf.float32,
                initializer=initializer)
        self.built = True
        
    def get_controls(self, audio, delay=None, matrix=None):
        if self.trainable:
            delay = self._delay
            matrix = self._matrix
        else:
            if delay is None:
                raise ValueError('Must provide "delay" tensor if FDN trainable=False.')
            if matrix is None:
                raise ValueError('Must provide "matrix" tensor if FDN trainable=False.')

        delay = tf.nn.sigmoid(delay) # softmax for probability delay [n_channels, max_delay]
        matrix= tf.nn.sigmoid(matrix)

        matrix = tf.reshape(matrix, [n_channels, n_channels])
        return {'audio': audio, 'delay': delay, 'matrix': matrix}
    
    def get_signal(self, audio, delay, matrix, states):
        
        audio, delay, matrix = tf_float32(audio),tf_float32(delay),tf_float32(matrix)
        states = tf.reshape(states, [n_channels,max_delay])
        delayed = delay_line(delay,states)
        audio = delayed + audio
        mixed = tf.keras.backend.dot(audio, matrix)
        
        states = tf.slice(states, [0,1], [n_channels,max_delay-1])
        states = tf.concat([states, tf.transpose(mixed)], axis=1)
        audio = tf.math.reduce_sum(mixed, axis=1)
        
        return audio, [tf.reshape(states,[1,n_channels*max_delay])]

    
