import tensorflow as tf
import numpy as np

def quantization_model(total_N, N, theta, steps, clip_model_W):

    Nb = np.power(2, N)  # 2
    total_Nb = np.power(2, total_N)  # 8
    criteria = []
    quan_condition = []
    for i in range(Nb - 1):
        criteria.append((steps[int(i * total_Nb / Nb)] + steps[int((i + 1) * total_Nb / Nb)]) / 2)


    quan_condition.append(tf.less_equal(theta, criteria[0]))
    for i in range(Nb - 2):
        quan_condition_1 = tf.greater(theta, criteria[i])
        quan_condition_2 = tf.less_equal(theta, criteria[i + 1])
        quan_condition.append(tf.logical_and(quan_condition_1, quan_condition_2))

    quan_condition.append(tf.greater(theta, criteria[-1]))

    quan_model_W = tf.where(quan_condition[0], steps[0] * tf.ones_like(clip_model_W), clip_model_W)
    for i in range(1, Nb):
        next_quan_model_W = tf.where(quan_condition[i], steps[int(i * total_Nb / Nb)] * tf.ones_like(clip_model_W),
                                     quan_model_W)
        quan_model_W = next_quan_model_W

    return quan_model_W


def bit_steps(step_size, bit):
    step = tf.concat([np.arange(np.multiply(-step_size, int(np.power(2, bit) / 2)), 0, step_size, dtype=np.float32),
               np.arange(0, np.multiply(step_size, (int(np.power(2, bit) / 2) - 1)) + 0.0001, step_size, dtype=np.float32)],
              0)
    #
    # negative_step = tf.range(-step_size * int(np.power(2, bit) / 2), 0, step_size, dtype=tf.float32)
    # positive_step = tf.sort(-negative_step)[:-1]
    # step = tf.concat([negative_step, tf.constant([0.]), positive_step], axis=0)
    # step = tf.sort(step)


    return step

def quantize(theta, step_list):
    dif_list = []
    for step in step_list:
        dif_list.append(tf.abs(theta - step))
    diff = tf.concat(dif_list, axis=0)




def rw_schedule(epoch):  # 0 ~ 0.05
    if epoch < 2:
        return 0.0
    if epoch < 32:
        return (epoch - 2) / 30 / 20
    return 1 / 20

class ClipConstraint(tf.keras.constraints.Constraint):
    def __init__(self, min_value=-8.0, max_value=8.0):
        super(ClipConstraint, self).__init__()
        self.min_value = min_value
        self.max_value = max_value

    def __call__(self, w):
        return tf.clip_by_value(w, self.min_value, self.max_value)

    def get_config(self):
        return {'Min_Value': self.min_value,
                'Max_Value': self.max_value}

class NanConstraint(tf.keras.constraints.Constraint):
    def __init__(self, value=0.0):
        super(NanConstraint, self).__init__()
        self.value = value

    def __call__(self, w):
        return tf.where(tf.math.is_nan(w), tf.zeros_like(w), w)

    def get_config(self):
        return {'Value': self.value}

