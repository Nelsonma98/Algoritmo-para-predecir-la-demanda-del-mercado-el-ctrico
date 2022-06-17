import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

X = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
Y = np.array([-40, 14, 32, 46, 50, 72, 100], dtype=float)

lr = 0.01  # learning rate
nn = [1, 8, 16, 8, 4, 1]

# Estructura que contendara al modelo
model = tf.keras.Sequential()

# Capa 1
model.add(tf.keras.layers.Dense(nn[1], input_shape=[nn[0]]))

# Capa 2
model.add(tf.keras.layers.Dense(nn[2], input_shape=[nn[1]]))

# Capa 3
model.add(tf.keras.layers.Dense(nn[3], input_shape=[nn[2]]))

# Capa 4
model.add(tf.keras.layers.Dense(nn[4], input_shape=[nn[3]]))

# Capa 5
model.add(tf.keras.layers.Dense(nn[5], input_shape=[nn[4]]))

# Compilacion del modelo
model.compile(optimizer=tf.keras.optimizers.Adam(lr=lr), loss='mean_squared_error', metrics=['acc'])

# Entrenamiento
print("Comenzando el entrenamiento...")
history=model.fit(X, Y, epochs=1000, verbose=False)
print("Modelo entrenado!")

# Grafica del error
plt.xlabel("#Epoca")
plt.ylabel("Magnitud de perdida")
plt.plot(history.history["loss"])
plt.show()