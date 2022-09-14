import numpy as np
import tensorflow as tf

from Layer.UVNQDense import UVNQDense
from Layer.UVNQConv2d import UVNQConv2d

class UVNQLeNet_5(tf.keras.Model):
    def __init__(self, n_class=10, total_N=4, beta=1.5):
        super(UVNQLeNet_5, self).__init__()
        self.n_class = n_class
        self.total_N = total_N
        self.total_Nb = np.power(2, self.total_N)
        self.beta = beta

        self.clip_alpha = 4 * (self.beta ** 2) / 12

        self.conv1 = UVNQConv2d(filters=6, kernel_size=5, strides=1, padding='VALID', activation='relu', total_N=total_N, beta=beta)
        self.pooling1 = tf.keras.layers.MaxPooling2D(padding='SAME')
        self.conv2 = UVNQConv2d(filters=16, kernel_size=5, strides=1, padding='VALID', activation='relu', total_N=total_N, beta=beta)
        self.pooling2 = tf.keras.layers.MaxPooling2D(padding='SAME')

        self.flat = tf.keras.layers.Flatten()
        self.fc1 = UVNQDense(output_dim=120, activation='relu', total_N=total_N, beta=beta)
        self.fc2 = UVNQDense(output_dim=84, activation='relu', total_N=total_N, beta=beta)
        self.fc3 = UVNQDense(output_dim=n_class, activation='softmax', total_N=total_N, beta=beta)

    @tf.function
    def call(self, x, training=True):
        x = self.conv1(x, training=training)
        x = self.pooling1(x)
        x = self.conv2(x, training=training)
        x = self.pooling2(x)
        x = self.flat(x)
        x = self.fc1(x, training=training)
        x = self.fc2(x, training=training)
        x = self.fc3(x, training=training)

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

    def quantization_test(self, x):
        step_size = self.calculate_step_size()
        # step = tf.concat([tf.range(-step_size * int(self.total_Nb / 2), 0, step_size, dtype=tf.float32),
        #                         tf.range(0, step_size * (int(self.total_Nb / 2) + 0.001), step_size, dtype=tf.float32)], 0).numpy()

        negative_step = np.arange(-step_size * int(np.power(2, self.total_N) / 2), 0, step_size, dtype=np.float32)
        positive_step = np.sort(-negative_step)[:-1]
        step = np.concatenate([negative_step, tf.constant([0.]), positive_step], axis=0)
        step = sorted(step, key=lambda s: abs(s))

        print("step: ", step)
        x = self.conv1.quantization_test(x, step)
        x = self.pooling1(x)
        x = self.conv2.quantization_test(x, step)
        x = self.pooling2(x)
        x = self.flat(x)
        x = self.fc1.quantization_test(x, step)
        x = self.fc2.quantization_test(x, step)
        x = self.fc3.quantization_test(x, step)

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
                layer_param = np.unique(layer.quantized_theta.numpy())
                param_list.extend(layer_param)

        return np.sort(np.unique(param_list))



