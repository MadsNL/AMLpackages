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


def make_NN(n_layers=3, n1=376, n2=475, input_length=162):

  dense_input = keras.Input(shape=(input_length))
  x = keras.layers.Dense(n1, activation='relu')(dense_input)
  x = keras.layers.Dropout(0.2)(x)
  x = keras.layers.BatchNormalization(momentum=0.9, center=True, scale=True)(x)
  for i in range(n_layers):
    x = keras.layers.Dense(n2, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.BatchNormalization(momentum=0.9, center=True, scale=True)(x)
  x = keras.layers.Dense(n1, activation='relu')(x)
  
  dense_last_layer = keras.Model(dense_input, x, name='last_layer')
  
  x = keras.layers.Dropout(0.2)(x)
  x = keras.layers.BatchNormalization(momentum=0.9, center=True, scale=True)(x)
  dense_output = keras.layers.Dense(7, activation='softmax')(x)

  dense_model = keras.Model(dense_input, dense_output, name='dense_model')

  opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

  dense_last_layer.compile(opt, loss='SparseCategoricalCrossentropy', metrics=['accuracy'])
  dense_model.compile(opt, loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

  return dense_model, dense_last_layer


def make_NN_3(n_layers=3, n1=376, n2=475, input_length=162):

  dense_input = keras.Input(shape=(input_length))
  x = keras.layers.Dense(n1, activation='relu')(dense_input)
  x = keras.layers.Dropout(0.2)(x)
  x = keras.layers.BatchNormalization(momentum=0.9, center=True, scale=True)(x)
  for i in range(n_layers):
    x = keras.layers.Dense(n2, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.BatchNormalization(momentum=0.9, center=True, scale=True)(x)
  x = keras.layers.Dense(n1, activation='relu')(x)
  
  dense_last_layer = keras.Model(dense_input, x, name='last_layer')
  
  x = keras.layers.Dropout(0.2)(x)
  x = keras.layers.BatchNormalization(momentum=0.9, center=True, scale=True)(x)
  dense_output = keras.layers.Dense(3, activation='softmax')(x)

  dense_model = keras.Model(dense_input, dense_output, name='dense_model')

  opt = keras.optimizers.Adam(lr=0.001, decay=1e-6)

  dense_last_layer.compile(opt, loss='SparseCategoricalCrossentropy', metrics=['accuracy'])
  dense_model.compile(opt, loss='SparseCategoricalCrossentropy', metrics=['accuracy'])

  return dense_model, dense_last_layer
