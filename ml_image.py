import cv2
from PIL import Image
import numpy as np
from keras.datasets import fashion_mnist  # Для работы с данными
from keras.models import Sequential, load_model  # Для представления нейронной сети
from keras.layers import Dense  # Подключение полносвязного слоя
from keras import utils  # Для приведения данных в удобный формат для обработки
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image

# Название классов

classes = ["футболка", "брюки", "свитер", "платье", "пальто", "туфли", "рубашка", "кроссовки", "сумка", "ботинки"]


def load_date():
    # Загружаем данные
    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
    # Преобразование размерности изображений
    x_train = x_train.reshape(60000, 784)
    # Нормализация данных
    x_train = x_train / 255
    # Преобразуем метки в категории

    y_train = utils.to_categorical(y_train, 10)
    return (x_train, y_train), (x_test, y_test)


def models_creat(x_train, y_train):
    # Создание последовательной модели
    model = Sequential()

    # Добавляем уровни сети
    model.add(Dense(800, input_dim=784, activation="relu"))
    model.add(Dense(10, activation='softmax'))

    # Компилируем модель
    model.compile(loss="categorical_crossentropy", optimizer="SGD", metrics=['accuracy'])
    # Просмотр архитектуры модели
    print(model.summary())

    # Обучение сети
    callbacks = [ModelCheckpoint("ML.keras", monitor='val_loss')]
    model.fit(x_train, y_train, batch_size=200, epochs=100, verbose=1, callbacks=callbacks)
    predictions = model.predict(x_train)

    # Вывод номера класса, предсказанный нейросетью
    print(np.argmax(predictions[0]))
    print(classes[np.argmax(predictions[0])])
    return model


def model_load(name):
    return load_model(name)


def test_model(model, x_test):
    f = model.predict(x_test)
    print("Номер класса:", np.argmax(f[0]))
    print("Название класса:", classes[np.argmax(f[0])])


def import_images(img_path, size):
    img = image.load_img(img_path, target_size=(size, size), color_mode="grayscale")

    # Преобразование картинки в массив
    x = image.img_to_array(img)

    # Меняем форму массива в плоский вектор
    x = x.reshape(1, 784)
    # Инвертируем изображение
    x = 255 - x
    # Нормализация данных
    x /= 255
    return x


model = model_load("ML.keras")
x_test = import_images("data/image/fut.jpg", 28)
test_model(model, x_test)
