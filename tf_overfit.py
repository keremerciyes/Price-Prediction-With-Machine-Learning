import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Generate a random time series dataset
time = np.arange(1000)
value = np.sin(time * 0.1) + np.random.randn(1000) * 0.1

# Split the dataset into training and validation sets
split_time = 700
train_time = time[:split_time]
train_value = value[:split_time]
valid_time = time[split_time:]
valid_value = value[split_time:]

# Define a function to create the time series model
def create_model():
    model = tf.keras.Sequential([
        layers.Dense(10, activation='relu', input_shape=[1]),
        layers.Dense(10, activation='relu'),
        layers.Dense(1)
    ])
    model.compile(loss='mse', optimizer='adam')
    return model

# Train the model with different epochs to demonstrate underfitting and overfitting
model = create_model()
history = model.fit(train_time, train_value, epochs=10, validation_data=(valid_time, valid_value))
history_overfit = model.fit(train_time, train_value, epochs=50, validation_data=(valid_time, valid_value))
history_underfit = model.fit(train_time, train_value, epochs=1, validation_data=(valid_time, valid_value))

# Plot the training and validation loss for each epoch
import matplotlib.pyplot as plt

def plot_loss(history):
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.legend()
    plt.show()

plot_loss(history)
plot_loss(history_overfit)
plot_loss(history_underfit)