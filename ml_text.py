from keras.datasets import imdb
from keras.models import Sequential
from keras.layers import Dense, Embedding, Flatten, Dropout, SimpleRNN, LSTM, GRU
from keras import utils
from keras.preprocessing.sequence import pad_sequences
import numpy as np
import matplotlib.pyplot as plt

max_words = 10000
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

word_index = imdb.get_word_index()
maxlen = 200


def coding_data(type: str, x_train, x_test):
    # Представление данных в виде токенизации на уровне слов
    if type == "pad_sequences":
        x_train = pad_sequences(x_train, maxlen=maxlen, padding="post")
        x_test = pad_sequences(x_test, maxlen=maxlen, padding='post')
    # Представление данных в One hot encoding
    elif type == "vectorize_sequences":
        x_train = vectorize_sequences(x_train, max_words)
        x_test = vectorize_sequences(x_test, max_words)
    return x_train, x_test


# Функция для кодирования one hot encoding
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1
    return results


def create_model(maxlen, x_train, y_train):
    model = Sequential()
    model.add(Dense(128, activation="relu", input_shape=(maxlen,)))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=25, batch_size=128, validation_split=0.1)

    return model, history


def create_model_vector(x_train, y_train):
    model = Sequential()
    # Слой плотных векторных представлений
    model.add(Embedding(max_words, 2, input_length=maxlen))

    # Снижение переобучения
    model.add(Dropout(0.25))
    # Преобразование в плоский вектор
    model.add(Flatten())
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=25, batch_size=128, validation_split=0.1)
    return model, history


def create_model_RNN(x_train, y_train):
    model = Sequential()
    model.add(Embedding(max_words, 2, input_length=maxlen))  # Представление слов в плоском числовом векторе
    model.add(SimpleRNN(8))  # Рекурентный слой
    model.add(Dense(1, activation='sigmoid'))  # Классификация
    model.compile(optimizer='rmsprop', loss="binary_crossentropy", metrics=['accuracy'])
    history = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_split=0.1)
    return model, history


def create_model_LSTM(x_train, y_train):
    model = Sequential()
    model.add(Embedding(max_words, 8, input_length=maxlen))
    model.add(LSTM(32, recurrent_dropout= 0.2)) # recurrent_dropout - метод регуляризации
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=25, batch_size=128, validation_split=0.1)

    return model, history


def create_model_GRU(x_train, y_train):
    model = Sequential()
    model.add(Embedding(max_words, 8, input_length=maxlen))
    model.add(GRU(32))
    model.add(Dense(1, activation="sigmoid"))
    model.compile(optimizer='adam', loss="binary_crossentropy", metrics=['accuracy'])

    history = model.fit(x_train, y_train, epochs=25, batch_size=128, validation_split=0.1)

    return model, history


def graph(history):
    plt.plot(history.history["accuracy"], label="Доля верных ответов на обучающем наборе")
    plt.plot(history.history["val_accuracy"], label="Доля верных ответов на проверочном наборе")
    plt.xlabel("Эпоха обучения")
    plt.ylabel("Доля верных ответов")
    plt.legend()
    plt.show()


def test(model, x_test, y_test):
    scopes = model.evaluate(x_test, y_test, verbose=1)
    print("Доля верных ответов на тестовых данных, в процентах:", round(scopes[1] * 100, 4))


x_train, x_test = coding_data("pad_sequences", x_train, x_test)

model, history = create_model_LSTM(x_train, y_train)
graph(history)
test(model, x_test, y_test)
