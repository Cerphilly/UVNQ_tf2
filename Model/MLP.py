import numpy as np
import tensorflow as tf

from Layer.UVNQDense import UVNQDense
from Layer.UVNQConv2d import UVNQConv2d

class UVNQMLP(tf.keras.Model):
    def __init__(self, n_class=10, total_N=4, beta=1.5):
        super(UVNQMLP, self).__init__()
        self.n_class = n_class
        self.total_N = total_N
        self.beta = beta

        self.clip_alpha = 4 * (self.beta ** 2) / 12

        self.flat = tf.keras.layers.Flatten()
        self.fc1 = UVNQDense(300, activation='relu')
        self.fc2 = UVNQDense(100, activation='relu')
        self.fc3 = UVNQDense(10, activation='softmax')

    @tf.function
    def call(self, input, training=True):
        x = self.flat(input)
        x = self.fc1(x, training)
        x = self.fc2(x, training)
        x = self.fc3(x, training)

        return x

    def calculate_step_size(self):
        theta_np = []

        for layer in self.layers:
            if isinstance(layer, UVNQDense) or isinstance(layer, UVNQConv2d):
                theta = layer.theta.numpy()
                mask = layer.boolean_mask.numpy()

                theta_np.append(theta[mask == True].reshape((-1, 1)))

        theta_std = np.std(np.vstack(theta_np))
        print("theta_std: ", theta_std)

        STEP_SIZE = np.sqrt(12 * self.clip_alpha / np.power(4, self.total_N) * theta_std * theta_std)

        print("step_size: ", STEP_SIZE)
        return STEP_SIZE

    def quantization_test(self, input):
        step_size = self.calculate_step_size()

        x = self.flat(input)
        x = self.fc1.quantization_test(x, step_size)
        x = tf.nn.relu(x)
        x = self.fc2.quantization_test(x, step_size)
        x = tf.nn.relu(x)
        x = self.fc3.quantization_test(x, step_size)
        x = tf.nn.softmax(x)

        return x

    def regularization(self):
        total_reg = 0

        for layer in self.layers:
            if isinstance(layer, UVNQDense) or isinstance(layer, UVNQConv2d):
                total_reg += layer.regularization

        return total_reg

    def count_sparsity(self):
        total_remain, total_param = 0, 0
        for layer in self.layers:
            if isinstance(layer, UVNQDense) or isinstance(layer, UVNQConv2d):
                a, b = layer.sparsity()
                total_remain += a
                total_param += b

        return 1 - (total_remain / total_param)

    def unique_params(self):
        param_list = []
        for layer in self.layers:
            if isinstance(layer, UVNQDense) or isinstance(layer, UVNQConv2d):
                print(layer.quantized_theta)
                exit()
                layer_param = np.unique(layer.quantized_theta.numpy())
                param_list.extend(layer_param)

        return np.unique(param_list).sort()

