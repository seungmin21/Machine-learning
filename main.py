
import tensorflow as tf
# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# print(tf.__version__)

fashion_mnist = tf.keras.datasets.fashion_mnist
# print(fashion_mnist)

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 데이터 탐색
# print(train_images.shape) # (60000, 28, 28)

# print(len(train_labels)) # 60000

# print(train_labels) # [9 0 0 ... 3 0 5]

# print(test_images.shape) # (10000, 28, 28)

# print(len(test_labels)) # 10000