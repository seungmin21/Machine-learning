import tensorflow as tf
# tensorFlow 버전 확인
# print(tf.__version__)

# 데이터 셋을 로드하는 참조 형태
mnist = tf.keras.datasets.mnist
# print(mnist) # module keras.api

# 튜플 형태의 로직 여러 변수에 값을 할당하는 형태
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# print(x_train)
# print(x_test)

# Sequential 일련의 함수을 순차적으로 쌓은 것(연속적인 변환)
# layer 입력 데이터를 받아 변환하여 출력을 생성(get으로 데이터를 받아 post로 변환을 요청 console로 출력)
# Flatten 다차원 배일을 1차원 배열로 평탄화하는 함수
# Dense 완전 연결 함수
# Dropout 무작위로 선택한 일부 뉴런의 출력을 제거하는 함수
# softmax 다중 클래스 분류 문제에서 각 클래스에 속할 확률을 출력하기 위해서 사용됩니다.
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

# 모델을 컴파일 하는 메서드
# optimizer='adam' 최적화 알고리즘을 설정하는 매개변수
# Adam 경사 하강법의 종류 가중치를 변경(업데이트)하는 방법
# loss = 'sparse_categorical' 손실함수를 설정하는 매개변수
# 모델의 예측값과 실제 레이블 간의 차이를 측정하는데 사용
# metrics=['accuracy'] 모델을 평가할 지표를 설정하는 매개변수
# 모델이 올바른 클래스를 얼마나 정확하게 예측하는지를 측정합니다.
model.compile(optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

# 모델을 사용하여 입력 데이터에 대한 예측을 수행하고 이를 이차원 배열 형태로 반환하는 것
predictions = model(x_train[:1]).numpy()
predictions

# 모델이 예측한 각 클래스에 대한 확률을 계산하여 이를 이차원 배열로 반환
result = tf.nn.softmax(predictions).numpy()
result

# tf.keras.losses keras API에서 제공하는 손실 함수를 포함하는 모듈
# sparse 정수 형태의 클래스 레이블을 다룸
# CategoricalCrossentropy 정수로 주어진 경우에 사용되는 것을 나타냄
# from_logits=True 모델의 출력값이 확률로 변환되기 이전에 로직값으로 주어진다는 것을 나타냄
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
loss_fn(y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
            loss=loss_fn,
            metrics=['accuracy'])

# fit 입력 데이터와 레이블을 사용하여 모델을 학습
# x_train 모델에 입력되는 학습 레이블
# y_train 학습 데이터에 대한 정답(레이블)데이터입니다.
# epochs 학습을 반복하는 횟수를 지정하는 매개변수
# epochs=5 5회 반복하여 학습한다는 의미
model.fit(x_train, y_train, epochs=5)

# evaluate 주어진 입력 데이터와 레이블을 사용하여 모델을 평가하고, 지정된 평가 지표(정확도)를 반환합니다.
# verbose 출력 정보의 양을 조절하는 매개변수
# verbose=2 verbose 매개변수를 설정하는 값 중 하나, 출력 정보를 상세하게 표시합니다.
# 모델을 테스트 데이터셋으로 평가하는 과정을 나타냄
model.evaluate(x_test,  y_test, verbose=2)


probability_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

print(probability_model(x_test[:5]))