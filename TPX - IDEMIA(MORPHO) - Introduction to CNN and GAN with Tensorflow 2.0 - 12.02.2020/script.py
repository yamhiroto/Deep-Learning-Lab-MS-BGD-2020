import tensorflow as tf
import cv2
from matplotlib import pyplot as plt



loaded = tf.saved_model.load("./my_model")
f = loaded.signatures["serving_default"]





def print_prediction_and_img(img, net):
	plt.imshow(img, cmap='gray', interpolation='bicubic')
	plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
	plt.show()
	to_be_predicted = tf.dtypes.cast(img, tf.float32)
	result = tf.nn.softmax(net(to_be_predicted, False))
	if result[0][0] > 0.5:
		print('Prediction: homme')
	else:
		print('Prediction: femme')

img = "./../andrew_small.jpg"
to_be_predicted = cv2.imread(img, 0)
print_prediction_and_img(to_be_predicted,loaded)