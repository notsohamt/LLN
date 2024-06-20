import tensorflow as tf
from tensorflow.keras.layers import Dense, RNN
from tensorflow.keras.models import Sequential
import numpy as np

class LTCCell(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        self.units = units
        self.state_size = units
        super(LTCCell, self).__init__(**kwargs)
        
    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.U = self.add_weight(shape=(self.units, self.units),
                                 initializer='random_normal',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.units,),
                                 initializer='zeros',
                                 trainable=True)
        self.alpha = self.add_weight(shape=(self.units,),
                                     initializer='ones',
                                     trainable=True)
        
    def call(self, inputs, states):
        prev_output = states[0]
        h = tf.tanh(tf.matmul(inputs, self.W) + tf.matmul(prev_output, self.U) + self.b)
        new_output = prev_output + self.alpha * (h - prev_output)
        return new_output, [new_output]

# Example of initializing an LTC cell
ltc_cell = LTCCell(units=10)


input_dim = 5
output_dim = 3

# Build the model
model = Sequential([
    RNN(ltc_cell, return_sequences=True, input_shape=(None, input_dim)),
    Dense(output_dim, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Generate some dummy data
input_data = np.random.random((100, 10, input_dim))
output_data = np.random.random((100, 10, output_dim))

# Train the model
model.fit(input_data, output_data, epochs=10)

# Generate some dummy test data
test_data = np.random.random((10, 10, input_dim))
test_labels = np.random.random((10, 10, output_dim))

# Evaluate the model
loss, accuracy = model.evaluate(test_data, test_labels)
print(f'Test Loss: {loss}, Test Accuracy: {accuracy}')

# Make predictions
predictions = model.predict(test_data)
print(predictions)
