from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.layers import Dense, ReLU
from tensorflow.keras.layers import Embedding, BatchNormalization, Concatenate
from tensorflow.keras.layers import Conv1D, GlobalMaxPooling1D, Dropout
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

class CNN:
    
    def __init__(self, input_dim, ouput_dim):
        self.model = Sequential()
        self.model.add(Embedding(input_dim = input_dim, output_dim = ouput_dim))
        self.model.add(Conv1D(128, 5, activation = 'relu'))
        self.model.add(GlobalMaxPooling1D())
        self.model.add(Dense(10, activation = 'relu'))
        self.model.add(Dense(6, activation = 'softmax'))

    def compile(self, optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy']):
        self.model.compile(optimizer = optimizer, loss = loss, metrics = metrics)

    def summary(self):
        self.model.summary()

    def fit(self, X_train, y_train, X_val, y_val):
        self.model.fit(X_train, y_train, epochs = 10, validation_data = (X_val, y_val), batch_size = 32)
    
    def save(self, file_path):
        self.model.save(file_path)