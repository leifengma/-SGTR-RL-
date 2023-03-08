import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

input_shape = (4, 10, 128)
x = tf.random.normal(input_shape)
y = layers.Conv1D(32, 3, activation='relu',input_shape=input_shape[1:])(x)

print(x)