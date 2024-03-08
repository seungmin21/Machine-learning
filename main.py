
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

train_images = train_images / 255.0

test_images = test_images / 255.0

# 새로운 그림을 생성하는 함수
# plt.figure()
# 이미지를 플로팅하는 함수
# 플로팅이란 데이터를 시각적으로 표시하는 것
# plt.imshow(train_images[0])
# 이미지의 각 픽셀 값에 대한 컬러 스케일을 보여줍니다.
# plt.colorbar()
# 그리드를 비활성화하는 함수
# 그리드는 이미지 표시 위에 격자를 표시하는 데 사용됩니다.
# plt.grid(False)
# plt.show()

# plt.figure(figsize=(10,10))
# for i in range(25):
    # plt.subplot(5,5,i+1)
    # plt.xticks([])
    # plt.yticks([])
    # plt.grid(False)
    # plt.imshow(train_images[i], cmap=plt.cm.binary)
    # plt.xlabel(class_names[train_labels[i]])
# plt.show()

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])

# model.fit(train_images, train_labels, epochs=10)

# 정확도 평가
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
# print('\nTest accuracy:', test_acc) # Test accuracy: 0.8848999738693237

# 확률 모델
probability_model = tf.keras.Sequential([model, 
                                        tf.keras.layers.Softmax()])

# 예측 값
predictions = probability_model.predict(test_images)

# print(predictions)
# print(predictions[0])

result = np.argmax(predictions[0])
# print(result)

# print(test_labels[0])


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    # x축의 눈금을 표시하지 않는 함수
    plt.xticks([])
    # y축의 눈금을 표시하지 않는 함수
    plt.yticks([])

    #  cmap=plt.cm.binary 흑백 이미지로 플로팅하도록 지정
    plt.imshow(img, cmap=plt.cm.binary)

    # 모델의 예측된 레이블
    # np.argmax(predictions_array) 가장 높은 확률을 가진 클래스를 예측
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    # x축 레이블을 지정하는 함수
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)