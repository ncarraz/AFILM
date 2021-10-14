import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
from tensorflow.keras.layers import MaxPooling1D, LSTM, Conv1D, LeakyReLU, Dropout, ReLU, Concatenate, Add
from .layers.subpixel import SubPixel1D
from .layers.transformer import TransformerBlock


class AFiLM(layers.Layer):
    def __init__(self, n_step, block_size, n_filters, **kwargs):
        super(AFiLM, self).__init__(**kwargs)
        self.block_size = block_size
        self.n_filters = n_filters
        self.n_step = n_step

    def get_config(self):
        config = super(AFiLM, self).get_config()
        config.update({"block_size": self.block_size,
                "n_filters": self.n_filters,
                "n_step":self.n_step})
        return config
    
    def build(self, input_shape):
        max_len = int(self.n_step/self.block_size)
        self.transformer = TransformerBlock(num_layers=4, embed_dim=input_shape[2], 
                                    num_heads=8, ff_dim=2048, maximum_position_encoding=max_len)
        self.transformer.build(input_shape)
        self._trainable_weights = self.transformer.trainable_weights
        super(AFiLM, self).build(input_shape)  # Be sure to call this at the end
    
    def make_normalizer(self, x_in):
        """ Pools to downsample along 'temporal' dimension and then 
            runs LSTM to generate normalization weights.
        """
        x_in_down = (MaxPooling1D(pool_size=self.block_size, padding='valid'))(x_in)
        x_transformer = self.transformer(x_in_down, training=True)
        return x_transformer

    def apply_normalizer(self, x_in, x_norm):
        """
        Applies normalization weights by multiplying them into their respective blocks.
        """
        n_blocks = tf.shape(x_in)[1] / self.block_size
        #n_filters = tf.shape(x_in)[2]
        # reshape input into blocks
        x_norm = tf.reshape(x_norm, [-1, n_blocks, 1, self.n_filters])
        x_in = tf.reshape(x_in, [-1, n_blocks, self.block_size, self.n_filters])
        # multiply
        x_out = x_norm * x_in
        # return to original shape
        x_out = tf.reshape(x_out, [-1, n_blocks * self.block_size, self.n_filters])
        return x_out
    
    def call(self, x):
        assert len(x.shape) == 3, 'Input should be tensor with dimension \
                                   (batch_size, steps, num_features).'
        #assert x.shape[1] % self.block_size == 0, 'Number of steps must be a multiple of the block size.'
        x_norm = self.make_normalizer(x)
        x = self.apply_normalizer(x, x_norm)
        return x

    def compute_output_shape(self, input_shape):
        return tf.TensorShape([input_shape[0],self.n_step,self.n_filters])


def get_afilm(n_layers=4, scale=4):
    n_filters = [  128,  256,  512, 512, 512, 512, 512, 512]
    n_blocks = [ 128, 64, 32, 16, 8]
    n_filtersizes = [65, 33, 17,  9,  9,  9,  9, 9, 9]
    n_step = [4096, 2048, 1024, 512, 256, 512, 1024, 2048, 4096]
    downsampling_l = []

    inputs = tf.keras.Input(shape=[8192, 1])
    x = inputs

    # DOWNSAMPLING LAYERS
    for l, nf, fs, ns in zip(range(n_layers), n_filters, n_filtersizes, n_step):
        x = (Conv1D(filters=nf, kernel_size=fs, dilation_rate=2,
                  activation=None, padding='same', kernel_initializer='orthogonal'))(x)
        x = (MaxPooling1D(pool_size=2, padding='valid'))(x)
        x = LeakyReLU(0.2)(x)
        nb = int(128 / (2**l))
        x = (AFiLM(ns, nb, nf))(x)
        downsampling_l.append(x)
    
    # BOTTLENECK LAYERS
    x = (Conv1D(filters=n_filters[-1], kernel_size=n_filtersizes[-1], dilation_rate=2,
                    activation=None, padding='same', kernel_initializer='orthogonal'))(x)
    x = (MaxPooling1D(pool_size=2, padding='valid'))(x)
    x = Dropout(0.5)(x)
    x = LeakyReLU(0.2)(x)
    nb = int(128 / (2**n_layers))
    x = (AFiLM(n_step[n_layers], nb, n_filters[-1]))(x)

    # UPSAMPLING LAYERS
    for l, nf, fs, l_in, ns in reversed(list(zip(range(n_layers), n_filters, n_filtersizes, downsampling_l, n_step))):
        x = (Conv1D(filters=2*nf, kernel_size=fs, dilation_rate=2,
                        activation=None, padding='same', kernel_initializer='orthogonal'))(x)
        x = Dropout(0.5)(x)
        x = ReLU()(x)
        x = SubPixel1D(x, r=2) 
        x = (AFiLM(ns, nb, nf))(x)
        x = Concatenate(axis=-1)([x, l_in])
 
    x = Conv1D(filters=2, kernel_size=9, 
                    activation=None, padding='same', kernel_initializer="normal")(x)  
    x = SubPixel1D(x, r=2)
    outputs = Add()([x, inputs]) 

    model = keras.Model(inputs, outputs)
    return model