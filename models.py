import tensorflow as tf
from tensorflow import keras
import numpy as np
from tensorflow.keras import layers
from transform import HistNormTransform


class HL_model:
    def __init__(self, model, bins, input_shape):
        self.bins = bins #np.array
        model_inputs = keras.Input(shape=input_shape)
        x = model(model_inputs)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(np.size(bins), activation="softmax")(x)
        self.model = keras.Model(model_inputs, output)
        y_min = 0
        y_max = 100
        self.transformer = HistNormTransform.from_bins(tf.linspace(y_min, y_max, bins), 1.0)
        
    def load(self, filename, bins):
        self.bins = bins
        self.model = keras.models.load_model(filename)
        
    def train(self, dataset, batch_size=256, epochs=1, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, save_file=None):
        dataset = dataset.batch(batch_size).map(lambda x, y: (x, self.transformer.transform(y)))
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon),
            loss=keras.losses.CategoricalCrossentropy()
                          )
        
        self.model.fit(
            x=dataset,
            epochs=epochs,
            verbose=1,
        )
        if save_file:
            self.save(save_file)
            
    def save(self, filename):
        self.model.save(filename)
        
        
    def validate(self, dataset):
        num_samples = dataset.cardinality().numpy()
        dataset = dataset.batch(num_samples)
        targets = dataset.map(lambda x, y: y)
        predictions = self.model.predict(dataset)
        regression = np.multiply(predictions, self.bins)
        regression = np.sum(regression, axis=1)
        mse = np.subtract(regression, targets)
        mse = np.square(mse)
        mse = np.sum(mse)/num_samples
        return mse
        
        
class Regression:
    def __init__(self, model, input_shape):
        model_inputs = keras.Input(shape=input_shape)
        x = model(model_inputs)
        x = layers.Dropout(0.5)(x)
        output = layers.Dense(1)(x)
        self.model = keras.Model(model_inputs, output)
        
    def load(self, filename):
        self.model = keras.models.load_model(filename)
        
    def train(self, dataset, batch_size=256, epochs=1, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, save_file=None):
        dataset = dataset.batch(batch_size)
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon),
            loss=keras.losses.MeanSquaredError()
                          )
        history = self.model.fit(
            x=dataset,
            epochs=epochs,
            verbose=1,
        )
        if save_file:
            self.save(save_file)
        
    def save(self, filename):
        self.model.save(filename)
        
        
    def validate(self, dataset):
        dataset = dataset.batch(dataset.cardinality().numpy())
        mse = self.model.evaluate(dataset)
        return mse
        