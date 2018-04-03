from keras import optimizers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM, GRU
from keras.callbacks import ModelCheckpoint
from keras.layers import Dense, Dropout, Activation, Flatten, Bidirectional
from keras.datasets import imdb
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization

class NeuralNetwork(object):
    
    def modelNL(self):
        model = Sequential() 
        
        hidden = 100

        model.add(GRU(hidden, return_sequences=True, activation='relu', input_shape=(self.seq_lenght, self.features)))
        model.add(Bidirectional(GRU(hidden, return_sequences=False, activation='tanh')))
        
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(.2))
        model.add(Dense(32, activation='relu'))
        model.add(Dropout(.2))
        
        #model.add(Flatten())
        
        model.add(Dense(self.target, activation='relu'))
        
        model.add(Activation('softmax'))

        model.compile(loss='categorical_crossentropy', optimizer='adagrad', metrics=['mae', 'acc'])
        
        model.summary()
        
        return model
    
    def __init__(self, learning_rate, features, target, seq_lenght = 1):
        
        self.seq_lenght = seq_lenght
        self.features = features
        self.target = target
        self.lr = learning_rate
        self.model = self.modelNL()
    
    def train(self, X, Y, epochs=150):
        # Fit the model
        self.model.fit(X, Y, epochs=epochs, validation_split=0.3,  verbose=1)

    
    def run(self, inputs_list):
        predictions = self.model.predict(inputs_list.values)
        
        return predictions