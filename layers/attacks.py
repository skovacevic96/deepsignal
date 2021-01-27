import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Add

 class InversionLayer(keras.layers.Layer):
    def __init__(self):
        super(InversionLayer, self).__init__()

    def call(self, inputs):
        return tf.math.scalar_mul(-1,inputs)
        
class UniformNoiseLayer(keras.layers.Layer):
    def __init__(self, noise_strength=0.009):
        super(AdditiveNoiseLayer, self).__init__()
        self.noise_strength = noise_strength

    def call(self, inputs):
        return Add()([tf.random.uniform(tf.shape(inputs), maxval=self.noise_strength), inputs])
        
class CuttingSamplesLayer(keras.layers.Layer):
  def __init__(self, num_samples=100, batch_size=64, input_dim=(32768, 1),  **kwargs):
    super(CuttingSamplesLayer, self).__init__(**kwargs)
    self.num_samples = num_samples
    self.batch_size = batch_size
    self.input_dim = input_dim
  
  def call(self, inputs):
    indices = tf.random.uniform_candidate_sampler(true_classes=tf.zeros(shape=(self.batch_size, self.input_dim[0]), dtype='int64'), num_true=self.input_dim[0], unique=True, num_sampled=self.batch_size*self.num_samples, range_max=self.batch_size*self.input_dim[0]).sampled_candidates
    idx = tf.scatter_nd(tf.reshape(indices, shape=(self.batch_size,self.num_samples,1)), tf.ones((self.batch_size,self.num_samples)), shape=(self.batch_size*self.input_dim[0],))
    idx = tf.reshape(idx, shape=(self.batch_size,self.input_dim[0],1))
    idx_keep = tf.where(idx==0)
    idx_remove = tf.where(idx!=0)
    values_remove = tf.tile([0.0], [tf.shape(idx_remove)[0]])
    values_keep = tf.gather_nd(inputs, idx_keep)
    inputs_remove = tf.SparseTensor(idx_remove, values_remove, tf.shape(inputs, out_type=tf.dtypes.int64))
    inputs_keep = tf.SparseTensor(idx_keep, values_keep, tf.shape(inputs, out_type=tf.dtypes.int64))
    output = tf.add(tf.sparse.to_dense(inputs_remove, default_value = 0. ), tf.sparse.to_dense(inputs_keep, default_value = 0.))
    
    return output
