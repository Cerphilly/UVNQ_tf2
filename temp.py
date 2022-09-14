import tensorflow as tf
import numpy as np

step_size = 0.03

bit1_step = tf.concat([tf.range(-step_size * int(2 / 2), 0, step_size, dtype=tf.float32),
                       tf.range(0, step_size * (int(2 / 2) - 1) + 0.0001, step_size, dtype=tf.float32)],
                      0)

bit2_step = tf.concat([tf.range(-step_size * int(4 / 2), 0, step_size, dtype=tf.float32),
                       tf.range(0, step_size * (int(4 / 2) - 1) + 0.0001, step_size, dtype=tf.float32)],
                      0)

bit3_step = tf.concat([tf.range(-step_size * int(8 / 2), 0, step_size, dtype=tf.float32),
                       tf.range(0, step_size * (int(8 / 2) - 1) + 0.0001, step_size, dtype=tf.float32)],
                      0)

bit4_step = tf.concat([tf.range(-step_size * int(16 / 2), 0, step_size, dtype=tf.float32),
                       tf.range(0, step_size * (int(16 / 2) - 1) + 0.0001, step_size, dtype=tf.float32)],
                      0)

print(bit1_step)
print(bit2_step)
print(bit3_step)
print(bit4_step)