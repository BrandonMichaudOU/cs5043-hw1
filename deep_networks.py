import tensorflow as tf
from tensorflow.keras.layers import InputLayer, Dense
from tensorflow.keras.models import Sequential


def deep_network_basic(n_inputs, hidden_layers, n_output, activation='elu', activation_output='elu', lrate=0.001,
                       metrics=None):
    '''
    Construct a network with given architecture
    - Adam optimizer
    - MSE loss

    :param n_inputs: Number of input dimensions
    :param hidden_layers: Number of neurons in each hidden layer
    :param n_output: Number of ouptut dimensions
    :param activation: Activation function to be used for hidden units
    :param activation_output: Activation function to be used for output units
    :param lrate: Learning rate for Adam Optimizer
    :param metrics: Metrics to record after each epoch
    '''
    # Build dense sequential model
    model = Sequential()
    model.add(InputLayer(input_shape=(n_inputs,)))
    for i, n_hidden in enumerate(hidden_layers):
        model.add(Dense(n_hidden, use_bias=True, name='Hidden_%d' % i, activation=activation))
    model.add(Dense(n_output, use_bias=True, name='Output', activation=activation_output))

    # Optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=lrate, amsgrad=False)

    # Bind the optimizer and the loss function to the model
    model.compile(loss='mse', optimizer=opt, metrics=metrics)

    # Generate an ASCII representation of the architecture
    print(model.summary())
    return model
