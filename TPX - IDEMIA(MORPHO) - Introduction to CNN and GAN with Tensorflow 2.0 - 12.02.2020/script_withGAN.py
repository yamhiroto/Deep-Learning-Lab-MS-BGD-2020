import tensorflow as tf
import Layers
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

class ConvNeuralNet(tf.Module):
    def __init__(self):
        self.unflat = Layers.unflat('unflat', 48, 48, 1)
        self.cv1 = Layers.conv('conv_1', output_dim=3, filterSize=3, stride=1)
        self.mp = Layers.maxpool('pool', 2)
        self.cv2 = Layers.conv('conv_2', output_dim=6, filterSize=3, stride=1)
        self.cv3 = Layers.conv('conv_3', output_dim=12, filterSize=3, stride=1)
        self.flat = Layers.flat()
        self.fc = Layers.fc('fc', 2)
    def __call__(self, x, log_summary):
        x = self.unflat(x, log_summary)
        x = self.cv1(x, log_summary)
        x = self.mp(x)
        x = self.cv2(x, log_summary)
        x = self.mp(x)
        x = self.cv3(x, log_summary)
        x = self.mp(x)
        x = self.flat(x)
        x = self.fc(x, log_summary)
        return x
    def eval(self, x):
        return tf.nn.softmax(self(x, False))

loaded_cnn = ConvNeuralNet()
optimizer = tf.optimizers.Adam(1e-3)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=loaded_cnn)
ckpt.restore("saved_model-1")

def print_prediction_and_img(img, net):
    plt.imshow(img, cmap='gray', interpolation='bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    #plt.show()
    # img = img.reshape(1, -1)
    to_be_predicted = tf.dtypes.cast(img, tf.float32)
    pred = net.eval(to_be_predicted)
    # print(pred['output_0'])
    print(pred.numpy()
          )
    if pred.numpy()[0][0] > 0.5:
        print('Prediction: homme')
    else:
        print('Prediction: femme')

img = "./../andrew_small.jpg"
to_be_predicted = cv2.imread(img, 0)
print_prediction_and_img(to_be_predicted, loaded_cnn)

# introduce perturbation
print(to_be_predicted)
X = tf.dtypes.cast(to_be_predicted, tf.float32)
X = tf.reshape(X,[1,2304])
X = (X - 128) / 256.0
print(X)
DXinit = tf.constant(0.0, shape=[1,2304])
DX = tf.Variable(DXinit)

def create_adversarial_pattern(model, optimizer, image, label, DX):
    with tf.GradientTape() as tape:
        prediction = model(image + DX, False)
        prediction = tf.nn.log_softmax(prediction)
        loss = - tf.reduce_sum(label * prediction)
        gradient = tape.gradient(loss, [DX])
        optimizer.apply_gradients(zip(gradient, [DX]))
    return loss

adversarial_label = [0.0, 1.0]
for iter in range(10):
    loss = create_adversarial_pattern(loaded_cnn, optimizer, X, adversarial_label, DX)
    y = loaded_cnn.eval(X + DX)
    correct_prediction = tf.equal(tf.argmax(y, -1), tf.argmax(adversarial_label, -1))
    result = tf.cond(correct_prediction, lambda: True, lambda: False)
    if (result):
        print(f"C'est une femme à la {iter + 1} itération")
        dx_values = DX.read_value()
        break

plt.subplot(131)
plt.title("Original image")
plt.imshow(to_be_predicted, cmap='gray', interpolation='bicubic')
plt.xticks([]), plt.yticks([])
X_mod = X + dx_values
plt.subplot(132)
plt.title("Modified + DX image")
plt.imshow(tf.reshape(X_mod, [48,48]), cmap='gray')
plt.xticks([]), plt.yticks([])
plt.subplot(133)
plt.title("Amplified image")
plt.imshow(tf.reshape(abs(dx_values)*50, [48,48]), cmap='gray')
plt.xticks([]), plt.yticks([])
plt.show()
print_prediction_and_img(X_mod, loaded_cnn)
