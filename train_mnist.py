import tensorflow as tf
import numpy as np
import os

from Model.LeNet import UVNQLeNet_5
from Model.MLP import UVNQMLP

from sklearn.utils import shuffle


def rw_schedule(epoch):  # 0 ~ 0.05
    if epoch < 2:
        return 0.0
    if epoch < 32:
        return (epoch - 2) / 30 / 20
    return 1 / 20


def rw_schedule2(epoch):
    if epoch < 2:
        return 0.0
    if epoch < 32:
        return (epoch - 2) / 30 / 1000
    return 1 / 1000


def compute_loss(label, pred, reg):
    return criterion(label, pred) + reg


# @tf.function
def compute_loss2(label, pred):
    return criterion(label, pred)


def train_step(x, t, epoch):
    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        reg = rw_schedule(epoch) * model.regularization() / (len(x_train))
        loss = compute_loss(t, preds, reg)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)
    train_acc(t, preds)

    return preds


def pretrain_step(x, t, epoch):
    with tf.GradientTape() as tape:
        preds = model(x, training=True)
        loss = compute_loss(t, preds, 0)

    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    train_loss(loss)
    train_acc(t, preds)

    return preds


# @tf.function
def test_step(x, t):
    preds = model(x, training=False)
    loss = compute_loss2(t, preds)
    test_loss(loss)
    test_acc(t, preds)

    return preds


def quantization_step(x, t):
    preds = model.quantization_test(x)

    loss = compute_loss2(t, preds)
    test_loss(loss)
    test_acc(t, preds)

    return preds


if __name__ == '__main__':
    np.random.seed(1234)
    tf.random.set_seed(1234)
    #tf.config.set_visible_devices([], 'GPU')

    '''
    Load data
    '''
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = (x_train.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    x_test = (x_test.reshape(-1, 28, 28, 1) / 255).astype(np.float32)
    y_train = np.eye(10)[y_train].astype(np.float32)
    y_test = np.eye(10)[y_test].astype(np.float32)

    print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)

    epochs = 100
    batch_size = 100
    N = 4
    beta = 1.5
    pretrain = False#pretrain to train without network compression
    pretrain_path = 'Checkpoint/Pretrain/LeNet/'
    compression_path = 'Checkpoint/Compression/LeNet/'
    if not os.path.exists(pretrain_path): os.mkdir(pretrain_path)
    if not os.path.exists(compression_path): os.mkdir(compression_path)

    '''
    Build model
    '''
    model = UVNQLeNet_5(n_class=10, total_N=N, beta=beta)
    criterion = tf.losses.CategoricalCrossentropy()
    optimizer = tf.keras.optimizers.Adam()
    if not pretrain:
        model.load_weights(pretrain_path)
    '''
    Train model
    '''

    n_batches = x_train.shape[0] // batch_size

    train_loss = tf.keras.metrics.Mean()
    train_acc = tf.keras.metrics.CategoricalAccuracy()
    test_loss = tf.keras.metrics.Mean()
    test_acc = tf.keras.metrics.CategoricalAccuracy()

    for epoch in range(epochs):

        _x_train, _y_train = shuffle(x_train, y_train, random_state=42)

        for batch in range(n_batches):
            start = batch * batch_size
            end = start + batch_size
            if pretrain:
                pretrain_step(_x_train[start:end], _y_train[start:end], epoch)
            else:
                train_step(_x_train[start:end], _y_train[start:end], epoch)

        if epoch % 1 == 0 or epoch == epochs - 1:

            preds = test_step(x_test, y_test)
            print('-'*100)
            print('Epoch: {}, Train_Cost: {:.3f}, Train_Acc: {:.3f}, Test Cost: {:.3f}, Test Acc: {:.3f}'.format(
                epoch+1,
                train_loss.result(),
                train_acc.result(),
                test_loss.result(),
                test_acc.result()
            ))

            train_loss.reset_states()
            train_acc.reset_states()
            test_loss.reset_states()
            test_acc.reset_states()

            #if not pretrain:


        if pretrain:
            model.save_weights(pretrain_path)
        else:
            model.save_weights(compression_path)

        preds = quantization_step(x_test, y_test)
        print("Quantization Cost: {:.3f}, Quantization Acc: {:.3f}".format(test_loss.result(), test_acc.result()))
        # print("Unique Params: ", model.unique_params())
        print("Sparsity: ", model.count_sparsity())

    test_loss.reset_states()
    test_acc.reset_states()