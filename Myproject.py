# Tensorflow implementation of VAE on MNIST
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, losses, optimizers, datasets, Model, Sequential

IMAGE_SIZE = 28
LABELS_TYPE = 10

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255
    x = tf.reshape(x, [IMAGE_SIZE**2])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=LABELS_TYPE)
    return x, y


def dataload():
    (x, y), (x_val, y_val) = datasets.mnist.load_data()
    traindata = tf.data.Dataset.from_tensor_slices((x, y))
    traindata = traindata.map(preprocess).shuffle(1000).batch(100)
    testdata = tf.data.Dataset.from_tensor_slices((x_val, y_val))
    testdata = testdata.map(preprocess).batch(100)
    return traindata, testdata

class Autocoder(Model):
    def __init__(self, z_size, im_size):
        super(Autocoder, self).__init__()
        self.ec1 = layers.Dense(512, activation='relu')
        self.ec2 = layers.Dense(256, activation='relu')
        self.ec3 = layers.Dense(128, activation='relu')
        self.ec4 = layers.Dense(z_size*2)
        self.z_size = z_size

        self.dc1 = layers.Dense(128, activation='relu')
        self.dc2 = layers.Dense(256, activation='relu')
        self.dc3 = layers.Dense(512, activation='relu')
        self.dc4 = layers.Dense(im_size, activation='sigmoid')

    def encoder(self, input):
        h1 = self.ec1(input)
        h2 = self.ec2(h1)
        h3 = self.ec3(h2)
        h4 = self.ec4(h3)

        mu = h4[:, :self.z_size]
        sigma = h4[:, self.z_size:]
        return mu, sigma

    def decoder(self, z):
        d1 = self.dc1(z)
        d2 = self.dc2(d1)
        d3 = self.dc3(d2)
        output = self.dc4(d3)
        return output

    def call(self, im):
        mu, sigma = self.encoder(im)
        z = mu + tf.math.exp(sigma)*tf.random.normal(tf.shape(mu), 0, 1, dtype=tf.float32)
        y = self.decoder(z)
        y = tf.clip_by_value(y, 1e-8, 1-1e-8)
        return y, mu, sigma


def main():
    n_epoch = 10
    z_size = 30
    im_size = IMAGE_SIZE**2
    traindata, testdata = dataload()
    acc_train = []
    rate_dropout = 0.2

    # training
    coder = Autocoder(z_size, im_size)
    coder.build(input_shape=(None, im_size))
    optimizer = optimizers.Adam(lr=0.001)

    for epoch in range(n_epoch):
        for step, (x, _) in enumerate(traindata):
            with tf.GradientTape() as tape:
                y, mu, sigma = coder(x)
                KL_loss = 0.5 * tf.reduce_sum(tf.square(mu) + tf.square(tf.math.exp(sigma)) - 2*sigma - 1, 1)
                err_loss = tf.nn.l2_loss(x-y)
                KL_loss = tf.reduce_mean(KL_loss)
                err_loss = tf.reduce_mean(err_loss)
                loss = err_loss+KL_loss

            grad = tape.gradient(loss, coder.trainable_variables)
            optimizer.apply_gradients(zip(grad, coder.trainable_variables))

            if step % 100 == 0:
                print('Step: {:03d}, Epoch: {:03d}, loss: {:.3%},  Accuracy: {:.3%}'.format(step, epoch, err_loss, loss))

        acc_train.append(loss)

        # Data reconstruction
        x_val, y_val = next(iter(testdata))
        _y_val, _, _ = coder(x_val)
        x_reconst = tf.concat([x_val[:50], _y_val[:50]], axis=0)
        x_reconst = tf.reshape(x_reconst, (-1, 28, 28)).numpy()*255
        x_reconst = x_reconst.astype(np.uint8)
        show_images(x_reconst, 'Reconstructed Image in {:03d} epoch.png'.format(epoch))

        # Data generation
        z = tf.random.normal((100, z_size), 0, 1, tf.float32)
        reconst = coder.decoder(z)
        reconst = tf.reshape(reconst, [-1, IMAGE_SIZE, IMAGE_SIZE]).numpy()*255
        reconst = reconst.astype(np.uint8)
        show_images(reconst, 'Generated Image in {:03d} epoch.png'.format(epoch))


def show_images(im, title):
    global_im = Image.new('L', (IMAGE_SIZE*10, IMAGE_SIZE*10))

    index = 0
    for i in range(10):
        for j in range(10):
            cur_im = im[index]
            cur_im = Image.fromarray(cur_im, mode='L')
            global_im.paste(cur_im, (i*IMAGE_SIZE, j*IMAGE_SIZE))
            index += 1

    global_im.save(title)


if __name__ == '__main__':
    main()

