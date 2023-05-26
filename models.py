import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from transform import HistNormTransform


class HL_model:
    def __init__(self, model, bins, input_shape):
        
        model_inputs = keras.Input(shape=input_shape)
        x = model(model_inputs)
        #x = layers.Dropout(0.5)(x)
        output = layers.Dense(tf.size(bins) - 1, activation="softmax")(x)
        self.model = keras.Model(model_inputs, output)
        self.transformer = HistNormTransform.from_bins(bins, 1.0)
        self.bins = self.transformer.get_centers()
        
    def load(self, filename, bins):
        self.bins = bins
        self.model = keras.models.load_model(filename)
        
    def train(self, dataset, batch_size=256, epochs=1, learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, save_file=None):
        dataset = dataset.batch(batch_size).map(lambda x, y: (x, self.transformer.transform(y)))
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=epsilon),
            loss=keras.losses.CategoricalCrossentropy()
                          )
        
        history = self.model.fit(
            x=dataset,
            epochs=epochs,
            verbose=1,
        )
        if save_file:
            self.save(save_file)
        return history
            
    def save(self, filename):
        self.model.save(filename)
        
        
    def validate(self, dataset):
        num_samples = dataset.cardinality().numpy()
        dataset = dataset.batch(num_samples)
        targets = dataset.map(lambda x, y: y).get_single_element()
        predictions = self.model.predict(dataset)
        regression = tf.linalg.matvec(predictions, self.bins)
        mse = regression - targets
        mse = tf.math.square(mse)
        mse = tf.math.reduce_sum(mse)/num_samples
        return mse
        
        
class Regression:
    def __init__(self, model, input_shape):
        model_inputs = keras.Input(shape=input_shape)
        x = model(model_inputs)
        #x = layers.Dropout(0.5)(x)
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
        return history
        
    def save(self, filename):
        self.model.save(filename)
        
        
    def validate(self, dataset):
        dataset = dataset.batch(dataset.cardinality().numpy())
        mse = self.model.evaluate(dataset)
        return mse
        