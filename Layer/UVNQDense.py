import tensorflow as tf
import numpy as np

from Utils import quantization_model, NanConstraint, bit_steps
from tensorflow.python.keras import activations
from copy import deepcopy

class UVNQDense(tf.keras.layers.Layer):
    def __init__(self, output_dim, use_bias=True, total_N=4, beta=1.5, log_alpha_c=-8.605, threshold=None, activation=None, kernel_initializer='glorot_normal', bias_initializer='zeros'):
        super(UVNQDense, self).__init__()

        self.output_dim = output_dim
        self.use_bias = use_bias
        self.total_N = total_N
        self.total_Nb = np.power(2, self.total_N)

        self.beta = beta
        self.threshold = 4 * (beta ** 2) / 12 if threshold is None else threshold

        self.activation = None if activation is None else activations.get(activation)

        self.log_alpha_c = log_alpha_c #-8.605, -7.219, -5.833, -4.447

        self.kernel_initializer = kernel_initializer
        self.bias_initializer = bias_initializer

    def build(self, input_shape):
        self.theta = self.add_weight("kernel", shape=(int(input_shape[-1]), self.output_dim),
                                     initializer=self.kernel_initializer, constraint=NanConstraint(),  trainable=True)

        self.log_sigma2 = self.add_weight("log_sigma2", shape=(int(input_shape[-1]), self.output_dim),
                                          initializer=tf.constant_initializer(-10.0),constraint=NanConstraint(), trainable=True)

        if self.use_bias:
            self.bias = self.add_weight("bias", shape=(self.output_dim,),
                                        initializer=self.bias_initializer, constraint=NanConstraint(), trainable=True)

    @property
    def log_alpha(self):
        log_alpha = tf.clip_by_value(self.log_sigma2 - tf.math.log(tf.square(self.theta) + 1e-10), self.log_alpha_c, 8.0)
        log_alpha = tf.where(tf.math.is_nan(log_alpha), tf.zeros_like(log_alpha), log_alpha)

        return log_alpha

    @property
    def regularization(self):
        k1, k2, k3 = 0.63576, 1.8732, 1.48695
        C = -k1
        log_alpha = self.log_alpha
        mdkl = k1 * tf.sigmoid(k2 + k3 * log_alpha) - 0.5 * tf.math.log(1 + (tf.exp(-log_alpha))) + C

        return -tf.reduce_sum(mdkl)

    @property
    def sparse_theta(self):
        #theta = tf.where(tf.math.is_nan(self.theta), tf.zeros_like(self.theta), self.theta)
        return tf.where(self.boolean_mask, self.theta, tf.zeros_like(self.theta))

    @property
    def boolean_mask(self):
        return self.log_alpha < self.threshold

    # @property
    # def weight(self):
    #     sigma = tf.sqrt(tf.exp(self.log_alpha) * self.theta * self.theta)
    #     return self.theta + tf.random.normal(tf.shape(self.theta), 0.0, 1.0) * sigma
    #
    # @property
    # def sparse_weight(self):
    #     return tf.where(self.boolean_mask, self.weight, tf.zeros_like(self.weight))

    def sparsity(self):
        try:
            total_param = np.prod(tf.shape(self.quantized_theta))
            remaining_param = tf.math.count_nonzero(self.quantized_theta).numpy()

        except:
            total_param = np.prod(tf.shape(self.boolean_mask))
            remaining_param = tf.math.count_nonzero(tf.cast(self.boolean_mask, dtype=tf.uint8)).numpy()

        return remaining_param, total_param


    # def theta_quantization2(self, step):
    #     theta = deepcopy(self.theta)
    #     sparse_theta_ = deepcopy(self.sparse_theta)
    #
    #     quan_model_W_Q4 = quantization_model(self.total_N, 4, theta, step, sparse_theta_)
    #     quan_model_W_Q3 = quantization_model(self.total_N, 3, theta, step, sparse_theta_)
    #     quan_model_W_Q2 = quantization_model(self.total_N, 2, theta, step, sparse_theta_)
    #     quan_model_W_Q1 = quantization_model(self.total_N, 1, theta, step, sparse_theta_)
    #
    #     # print(quan_model_W_Q1)
    #     # #print(np.unique(quan_model_W_Q1.numpy()))
    #     # print('-'*100)
    #     # print(quan_model_W_Q2)
    #     #
    #     # #print(np.unique(quan_model_W_Q2.numpy()))
    #     # print('-'*100)
    #     # print(quan_model_W_Q3)
    #     #
    #     # #print(np.unique(quan_model_W_Q3.numpy()))
    #     # print('-'*100)
    #     # print(quan_model_W_Q4)
    #     #
    #     # #print(np.unique(quan_model_W_Q4.numpy()))
    #     # print('-'*100)
    #     # exit()
    #     log_alpha = deepcopy(self.log_alpha)
    #
    #     model_W_s1 = tf.where(
    #         tf.logical_and(tf.less_equal(tf.exp(log_alpha), self.threshold),
    #                        tf.greater(tf.exp(log_alpha), self.threshold / 4)),
    #         quan_model_W_Q1, sparse_theta_)
    #     model_W_s2 = tf.where(
    #         tf.logical_and(tf.less_equal(tf.exp(log_alpha), self.threshold / 4),
    #                        tf.greater(tf.exp(log_alpha), self.threshold / 16)),
    #         quan_model_W_Q2, model_W_s1)
    #     model_W_s3 = tf.where(
    #         tf.logical_and(tf.less_equal(tf.exp(log_alpha), self.threshold / 16),
    #                        tf.greater(tf.exp(log_alpha), self.threshold / 64)),
    #         quan_model_W_Q3, model_W_s2)
    #
    #     if self.total_N == 2:
    #         quantized_theta = tf.where(tf.less_equal(tf.exp(log_alpha), self.threshold / 4), quan_model_W_Q2, model_W_s1)#이게 2비트?
    #     elif self.total_N == 3:
    #         quantized_theta = tf.where(tf.less_equal(tf.exp(log_alpha), self.threshold / 16), quan_model_W_Q3, model_W_s2)#이게 3비트
    #     elif self.total_N == 4:
    #         quantized_theta = tf.where(tf.less_equal(tf.exp(log_alpha), self.threshold / 64), quan_model_W_Q4, model_W_s3)#4비트?
    #     else:
    #         raise ValueError
    #
    #     return quantized_theta

    def theta_quantization(self, step):

        bit_step_list = []
        for i in range(self.total_N):
            bit_step_list.append(sorted(step[:2**(i + 1)]))

        log_alpha = deepcopy(self.log_alpha)
        alpha = tf.exp(log_alpha)

        N_b = tf.zeros_like(log_alpha)
        for i in range(1, self.total_N + 1):
            N_b = tf.where(tf.logical_and(tf.greater(alpha, self.threshold / np.power(4, i)),tf.less_equal(alpha, self.threshold / np.power(4, i - 1))), tf.ones_like(log_alpha) * i, N_b)

        N_b = tf.where(tf.less_equal(alpha, self.threshold / np.power(4, self.total_N)), tf.ones_like(log_alpha) * self.total_N, N_b)

        quantized_theta_list = [tf.zeros_like(self.theta)]

        for i in range(self.total_N):
            dif_list = []
            for step in bit_step_list[i]:
                dif_list.append(tf.abs(self.theta - step))

            diff = tf.stack(dif_list, axis=0)
            idx = tf.argmin(diff, axis=0)

            quantized_theta_list.append(tf.gather(bit_step_list[i], idx))

        quantized_theta_list = tf.stack(quantized_theta_list, axis=0)

        N_b_onehot = tf.one_hot(tf.cast(N_b, dtype=tf.int32), depth=self.total_N + 1, axis=0, dtype=tf.float32)
        quantized_theta = tf.reduce_sum(quantized_theta_list * N_b_onehot, axis=0)

        return quantized_theta

    @tf.function
    def call(self, input, training=True):

        if training:
            sigma = tf.sqrt(tf.exp(self.log_alpha) * self.theta * self.theta)
            weight = self.theta + tf.random.normal(tf.shape(self.theta), 0.0, 1.0) * sigma
            output = tf.matmul(input, weight)

            if self.use_bias:
                output += self.bias

            if self.activation is not None:
                output = self.activation(output)

            return output

        else:

            output = tf.matmul(input, self.sparse_theta)

            if self.use_bias:
                output += self.bias
            if self.activation is not None:
                output = self.activation(output)

            return output

    def quantization_test(self, input, step):

        self.quantized_theta = self.theta_quantization(step)

        # a, _ = tf.unique(tf.reshape(self.quantized_theta, -1))
        # print(self.theta.shape, tf.sort(a))
        output = tf.matmul(input, self.quantized_theta)

        if self.use_bias:
            output += self.bias

        if self.activation is not None:
            output = self.activation(output)

        return output



