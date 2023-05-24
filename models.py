import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers


class HL_model:
    def __init__(self, model, bins, input_shape):
        self.bins = bins #np.array
        model_inputs = keras.Input(shape=input_shape)
        x = model(model_inputs)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(np.size(bins), activation="softmax")(x)
        self.model = keras.Model(model_inputs, output)
        
    def __init__(self, filename, bins):
        self.bins = bins
        self.model = keras.models.load_model(filename)
        
    def train(self, dataset, epochs, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, save_file=None):
        self.model.compile(
            optimizer=keras.optimizers.ADAM(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon),
            loss=keras.losses.SparseCategoricalCrossentropy()
                          )
        history = self.model.fit(
            x=dataset,
            epochs=epochs,
            verbose=2,
        )
        if save_file
            self.save(save_file)
        
    def save(self, filename):
        self.model.save(filename)
        
        
    def validate(self, inputs, targets):
        # inputs can be a dataset tensor or np.array
        # targets must be a numpy array
        predictions = self.model.predict(inputs)
        regression = np.multiply(predictions, self.bins)
        regression = np.sum(regression, axis=1)
        mse = np.subtract(regression, targets)
        mse = np.square(mse)
        mse = np.sum(mse)/np.size(x)
        return mse
        
        
class Regression:
     def __init__(self, model, input_shape):
        model_inputs = keras.Input(shape=input_shape)
        x = model(model_inputs)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(1)(x)
        self.model = keras.Model(model_inputs, output)
        
    def __init__(self, filename):
        self.model = keras.models.load_model(filename)
        
    def train(self, dataset, epochs, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, save_file=None):
        self.model.compile(
            optimizer=keras.optimizers.ADAM(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon),
            loss=keras.losses.MeanSquaredError()
                          )
        history = self.model.fit(
            x=dataset,
            epochs=epochs,
            verbose=2,
        )
        if save_file
            self.save(save_file)
        
    def save(self, filename):
        self.model.save(filename)
        
        
    def validate(self, inputs, targets):
        # inputs can be a dataset tensor or np.array
        # targets must be a numpy array
        predictions = self.model.predict(inputs)
        mse = np.subtract(regression, targets)
        mse = np.square(mse)
        mse = np.sum(mse)
        return mse
        