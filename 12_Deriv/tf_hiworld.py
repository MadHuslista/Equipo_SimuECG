
import tensorflow as tf
import numpy as np
from tensorflow import keras

#Defino el modelo 
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])

#Defino como optimiza y como calcula el error 
model.compile(optimizer='sgd', loss='mean_squared_error')

#defino la data y los labels.
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-2.0, 1.0, 4.0, 7.0, 10.0, 13.0], dtype=float)

#Ejecuto el aprendizaje
model.fit(xs, ys, epochs=500)

#Uso el modelo para ejecutar una predicción. 
print(model.predict([10.0]))