
import tensorflow as tf
from tensorflow.keras import layers, models
import pennylane as qml

n_qubits = 4
dev = qml.device("default.qubit", wires=n_qubits)


# Define a simple quantum circuit
@qml.qnode(dev)
# Define the quantum device and quantum circuit
def quantum_circuit(inputs, weights):
    qml.templates.AmplitudeEmbedding(inputs, wires=range(n_qubits), normalize=True)
    for i in range(n_qubits):
        qml.RX(weights[i], wires=i)
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

class QuantumLayer(layers.Layer):
    def __init__(self, n_qubits, **kwargs):
        super(QuantumLayer, self).__init__(**kwargs)
        self.n_qubits = n_qubits

    def build(self, input_shape):
        self.q_weights = self.add_weight(shape=(self.n_qubits,), initializer="random_normal", trainable=True, name="q_weights")
        super(QuantumLayer, self).build(input_shape)

    def call(self, inputs):
	    # Replace NaNs with zeros and cast the result to float32
	    print('call')
	    inputs = tf.where(tf.math.is_nan(inputs), tf.zeros_like(inputs, dtype=tf.float32), inputs)
	    # Reshape inputs to the required shape
	    inputs = tf.reshape(inputs, [-1, 16])
	    inputs = tf.cast(inputs, tf.float32)  # Ensure inputs are float32
	    print(inputs.dtype)
	    # Apply the quantum circuit function to each element in the inputs
	    quantum_output = tf.map_fn(
	        lambda x: tf.cast(
	            tf.py_function(quantum_circuit, [x, self.q_weights], tf.float32),
	            tf.float32
	        ),
	        inputs
	    )
	    # Set the shape of the output tensor
	    quantum_output.set_shape([None, self.n_qubits])
	    print(quantum_output)
	    
	    return quantum_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.n_qubits)

def create_qcnn_model(input_shape):
    inputs = layers.Input(shape=(input_shape,))
    x = layers.Reshape((input_shape, 1))(inputs)
    x = layers.Conv1D(32, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling1D(2)(x)
    x = layers.Conv1D(64, 3, activation="relu", padding="same")(x)
    x = layers.Flatten()(x)
    x = QuantumLayer(n_qubits)(x)
    outputs = layers.Dense(4, activation="softmax")(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model
    