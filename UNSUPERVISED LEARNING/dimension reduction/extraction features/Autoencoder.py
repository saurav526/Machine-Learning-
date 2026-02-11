# --------------------------------------
# Autoencoder : Dimensionality Reduction
# --------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# Load dataset
data = load_iris()
X = data.data
y = data.target

# Standardize
X_scaled = StandardScaler().fit_transform(X)

# Autoencoder architecture
input_layer = Input(shape=(4,))
encoded = Dense(2, activation='relu')(input_layer)
decoded = Dense(4, activation='linear')(encoded)

autoencoder = Model(input_layer, decoded)
encoder = Model(input_layer, encoded)

autoencoder.compile(optimizer='adam', loss='mse')

# Train
autoencoder.fit(X_scaled, X_scaled, epochs=100, batch_size=16, verbose=0)

# Get reduced features
X_encoded = encoder.predict(X_scaled)

# Plot result
plt.figure(figsize=(7,5))
plt.scatter(X_encoded[:, 0], X_encoded[:, 1], c=y, cmap='winter')
plt.xlabel("Encoded Feature 1")
plt.ylabel("Encoded Feature 2")
plt.title("Autoencoder - Dimensionality Reduction")
plt.colorbar()
plt.show()
