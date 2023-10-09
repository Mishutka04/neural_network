from keras import utils
from keras.models import Sequential
from keras.layers import Dense, Embedding, MaxPool1D, Conv1D, GlobalMaxPooling1D, Dropout, LSTM, GRU
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.callbacks import ModelCheckpoint
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class Classification:
    def __init__(self):
        self.num_words = 10000  # Максимальное количество слов
        self.max_news_len = 30  # Максимальная длина новости
        self.nb_classes = 4  # Количество классов новостей
        self.x_train = None
        self.y_train = None
        self.history = None

    def load_data(self):
        train = pd.read_csv('ag_news_csv/train.csv', header=None, names=['class', 'tutle', 'text'])
        self.x_train = train["text"]
        self.y_train = utils.to_categorical(train['class'] - 1, self.nb_classes)
        self.x_train = self.tokenizer(self.x_train, self.num_words)
        print(self.x_train)

    @staticmethod
    def tokenizer(x_train, num_words):
        tokenizer = Tokenizer(num_words=num_words)
        tokenizer.fit_on_texts(x_train)
        x_train = tokenizer.texts_to_sequences(x_train)
        return pad_sequences(x_train, maxlen=30)

    def cnn_network(self):
        model_cnn = Sequential()
        model_cnn.add(Embedding(self.num_words, 32, input_length=self.max_news_len))
        model_cnn.add(Conv1D(250, 5, padding='valid', activation='relu'))
        model_cnn.add(GlobalMaxPooling1D())
        model_cnn.add(Dense(128, activation='relu'))
        model_cnn.add(Dense(4, activation='softmax'))
        model_cnn.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        # print(model_cnn.summary())  # Описание нейронной сети

        self.history = model_cnn.fit(self.x_train, self.y_train, epochs=5, batch_size=128, validation_split=0.1)
        print(self.history)

    def lstm_network(self):
        model_lstm = Sequential()
        model_lstm.add(Embedding(self.num_words, 32, input_length=self.max_news_len))
        model_lstm.add(LSTM(16))
        model_lstm.add(Dense(4, activation='softmax'))
        model_lstm.compile(optimizer='adam',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
        model_lstm.summary()
        self.history = model_lstm.fit(self.x_train,
                                      self.y_train,
                                      epochs=5,
                                      batch_size=128,
                                      validation_split=0.1)

    def gru_network(self):
        model_gru = Sequential()
        model_gru.add(Embedding(self.num_words, 32, input_length=self.max_news_len))
        model_gru.add(GRU(16))
        model_gru.add(Dense(4, activation='softmax'))
        model_gru.compile(optimizer='adam',
                          loss='categorical_crossentropy',
                          metrics=['accuracy'])
        model_gru.summary()
        self.history = model_gru.fit(self.x_train,
                                     self.y_train,
                                     epochs=5,
                                     batch_size=128,
                                     validation_split=0.1)

    def create_chart(self, name_model):
        plt.plot(self.history.history['accuracy'], label=f"Доля верных ответов на обучающем наборе {name_model}")
        plt.plot(self.history.history['val_accuracy'], label=f"Доля верных ответов на проверочном наборе {name_model}")
        plt.xlabel("Эпоха обучения")
        plt.ylabel("Доля верных ответов")
        plt.legend()
        plt.savefig(f"{name_model}.png")


models = Classification()
models.load_data()
models.cnn_network()
models.create_chart("cnn_network")

models = Classification()
models.load_data()
models.lstm_network()
models.create_chart("lstm_network")

models = Classification()
models.load_data()
models.gru_network()
models.create_chart("gru_network")


