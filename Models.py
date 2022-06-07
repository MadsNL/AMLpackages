import tensorflow as tf
from tensorflow import keras


def make_autoencoder(latent_dim=128):
  image_size= 128
  unit_n = 32

  encoder_input = keras.Input(shape=(image_size,image_size,1))
  x = keras.layers.Conv2D(filters=32, kernel_size=3, strides=(2,2), activation='relu')(encoder_input)
  x = keras.layers.Conv2D(filters=64, kernel_size=3, strides=(2,2), activation='relu')(x)
  x = keras.layers.Flatten()(x)
  encoder_output = keras.layers.Dense(latent_dim)(x)

  encoder = keras.Model(encoder_input, encoder_output, name='encoder')

  #decoder_input = keras.layers.InputLayer(input_shape=latent_dim)(encoder_output)
  x = keras.layers.Dense(units=unit_n*unit_n*32, activation=tf.nn.relu)(encoder_output)
  x = keras.layers.Reshape(target_shape=(unit_n, unit_n, 32))(x)
  x = keras.layers.Conv2DTranspose(
      filters=64, kernel_size=3, strides=2, padding='same',
      activation='relu')(x)
  x = keras.layers.Conv2DTranspose(
      filters=32, kernel_size=3, strides=2, padding='same',
      activation='relu')(x)
  # No activation
  decoder_output = keras.layers.Conv2DTranspose(
      filters=1, kernel_size=3, strides=1, padding='same')(x)

  autoencoder = keras.Model(encoder_input, decoder_output, name='autoencoder')
  autoencoder.summary()

  opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

  encoder.compile(opt, loss='mse')
  autoencoder.compile(opt, loss='mse')
  
  return encoder, autoencoder
