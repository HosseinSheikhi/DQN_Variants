import tensorflow as tf


class ControlDQNetwork(tf.keras.Model):
    def __init__(self, output_size):
        super(ControlDQNetwork, self).__init__()
        self.output_size = output_size

        self.dense1 = tf.keras.layers.Dense(128, activation=tf.keras.activations.relu)
        self.dense2 = tf.keras.layers.Dense(64, activation=tf.keras.activations.relu)
        self.dense3 = tf.keras.layers.Dense(32, activation=tf.keras.activations.relu)

        # kernel initializer values are adopted from tensorflow rl agent -> dqn
        self.out = tf.keras.layers.Dense(self.output_size, activation=tf.keras.activations.linear,
                                            kernel_initializer=tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.03),
                                            bias_initializer=tf.keras.initializers.Constant(-0.2))

    def call(self, inputs, training=None, mask=None):
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        return self.out(x)
