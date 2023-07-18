import keras_tuner as kt
from tensorflow import keras
import tensorflow as tf
from keras import layers
import numpy as np


class TruncGaussHistTransform(keras.layers.Layer):
    """Layer that transforms a scalar target into a binned probability vector 
    that approximates a truncated Gaussian distribution with the target as the mean.
    
    Params:
        borders - the borders of the histogram bins
        sigma - the sigma parameter of the truncated Gaussian distribution
    """

    def __init__(self, borders, sigma):
        super().__init__(trainable=False, name="TruncGaussHistTransform")
        self.borders = borders
        self.sigma = sigma

    def call(self, inputs):
        """Transform the input and return it.
        
        Params:
            inputs - the tensor of targets to transform

        Returns:
            x_transformed - a tensor of shape (len(inputs), len(borders) - 1)
            consisting of the probability vectors for each target
        """
        border_targets = self.adjust_and_erf(self.borders, tf.expand_dims(inputs, 1), self.sigma)
        two_z = border_targets[:, -1] - border_targets[:, 0]
        x_transformed = (border_targets[:, 1:] - border_targets[:, :-1]) / tf.expand_dims(two_z, 1)
        return x_transformed

    def adjust_and_erf(self, a, mu, sig):
        """Calculate the erf of a after standardizing and dividing by sqrt(2)."""
        return tf.math.erf((a - mu)/(tf.math.sqrt(2.0)*sig))
    

class HistMean(keras.layers.Layer):
    """Layer that transforms a probability vector into its expected value.
    
    Params:
        centers - the centers of the histogram bins
    """

    def __init__(self, centers):
        super().__init__(trainable=False, name="HistMean")
        self.centers = centers

    def call(self, inputs):
        """Return the weighted average between the centers and probability vectors.
        
        Params:
            inputs - a tensor of probability vectors to transform

        Returns:
            a tensor of shape (len(inputs), 1) consisting of the expected values
        """
        return tf.linalg.matvec(inputs, self.centers)




class HyperLinear_L2(kt.HyperModel):
    def __init__(self, name="Linear_l2", input_length=256, output_length=256, metrics=None):
        super().__init__(name, True)
        self.metrics = metrics
        self.input_length = input_length
        self.output_length = output_length
        
    def build(self, hp):
        inputs = keras.Input(shape=(self.input_length,))
        dense = layers.Dense(self.output_length)
        outputs = dense(inputs)
        model = keras.Model(inputs=inputs, outputs=outputs)
        lr = hp.Float("learning_rate", default=1e-3, min_value=1e-4, max_value=1e-2, step=10, sampling="log")
        b1 = hp.Fixed("beta_1", 0.9)
        b2 = hp.Fixed("beta_2", 0.999)
        eps = hp.Fixed("epsilon", 1e-7)
        model.compile(
            loss=keras.losses.MeanSquaredError(),
            optimizer = keras.optimizer.Adam(learning_rate = lr, beta_1=b1, beta_2=b2, epsilon=eps),
            metrics=self.metrics
        )
        return model
    


class Linear_HL(keras.Model):
    def __init__(self, y_min, y_max, padding, n_bins, sigma, input_length, output_length, name="Linear_HL"):
        super().__init__()
        y_range = y_max-y_min
        centers = tf.linspace(y_min-(y_range*padding), y_max+(y_range*padding), n_bins)
        self.dense = layers.Dense(output_length*tf.size(centers))
        self.transform = TruncGaussHistTransform(centers, sigma)
        self.reshape = layers.Reshape((output_length, tf.size(centers)))
        self.softmax = layers.Softmax()
        self.mean = HistMean(centers)
        self.hist_loss = keras.metrics.Mean("loss")
        
    def call(self, inputs, training=None):
        x = self.dense(inputs, training=training)
        x = self.reshape(x, training=training)
        x = self.softmax(x, training=training)
        return self.mean(x, training=training)
    
    def train_step(self, data):
        x, y = data
        y_transformed = self.transform(y)
        
        with tf.GradientTape() as tape:
            features = self.dense(x, training=True)
            features = self.reshape(features, training=True)
            hist = self.softmax(features, training=True)
            loss = keras.losses.categorical_crossentropy(y_transformed, hist)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        y_pred = self.mean(hist)
        self.compiled_metrics.update_state(y, y_pred)
        self.hist_loss.update_state(loss)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data
        y_transformed = self.transform(y)
        
        
        features = self.dense(x, training=False)
        features = self.reshape(features, training=False)
        hist = self.softmax(features, training=False)
        
        loss = keras.losses.categorical_crossentropy(y_transformed, hist)
        self.hist_loss.update_state(loss)

        y_pred = self.mean(hist)
        self.compiled_metrics.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
        
    

class HyperLinear_HL(kt.HyperModel):
    def __init__(self, y_min, y_max, name="HyperLinear_HL", input_length=256, output_length=256, metrics=None, ):
        super().__init__(name, True)
        self.metrics = metrics
        self.input_length = input_length
        self.output_length = output_length
        self.y_min = y_min
        self.y_max = y_max
        
        
    def build(self, hp):
        model = self.get_model(hp)
        lr = hp.Float("learning_rate", default=1e-3, min_value=1e-4, max_value=1e-2, step=10, sampling="log")
        b1 = hp.Fixed("beta_1", 0.9)
        b2 = hp.Fixed("beta_2", 0.999)
        eps = hp.Fixed("epsilon", 1e-7)
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer = keras.optimizer.Adam(learning_rate = lr, beta_1=b1, beta_2=b2, epsilon=eps),
            metrics=self.metrics
        )
        return model
    
    def get_model(self, hp):
        sig_ratio = hp.Float("sig_ratio", default=1., min_value=0.5, max_value=2., step=2, sampling="log")
        padding = hp.Float("padding", default=0.1, min_value=0.025, max_value=0.1, step=2, sampling="log")
        n_bins = int(hp.Int("n_bins", default=100, min_value=25, max_value=400, step=2, sampling="log"))
        
        return Linear_HL(self.y_min, self.y_max, padding, n_bins, sig_ratio, self.input_length, self.output_length)
        


        
class NLinear_L2(keras.Model):
    def __init__(self, input_length, output_length, name="NLinear_L2"):
        super().__init__()
        self.dense = layers.Dense(output_length)
        self.NLinear_loss = keras.metrics.Mean("loss")
            
    def call(self, inputs, training=None):
        seq_last = inputs[:,-1].numpy()[:,np.newaxis]*np.ones(tf.size(inputs))
        x = inputs - seq_last
        x = self.dense(x, training=training)
        return x + seq_last
    
    def train_step(self, data):
        x, y = data
        seq_last = x[:,-1].numpy()[:,np.newaxis]*np.ones(tf.size(x))
        
        with tf.GradientTape as tape:
            features = x - seq_last
            features = self.dense(x, training=True)
            predictions = x + seq_last
            loss = keras.losses.mean_squared_error(y, predictions)
            
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        
        self.compiled_metrics.update_state(y, predictions)
        self.NLinear_loss.update_state(loss)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data
        seq_last = x[:,-1].numpy()[:,np.newaxis]*np.ones(tf.size(x))
        
        features = x - seq_last
        features = self.dense(x, training=True)
        predictions = x + seq_last
        
        loss = keras.losses.mean_squared_error(y, predictions)
        
        self.NLinear_loss.update_state(loss)

        self.compiled_metrics.update_state(y, predictions)
        
        return {m.name: m.result() for m in self.metrics}
    
    
    
class HyperNLinear_L2(kt.HyperModel):
    def __init__(self, name="HyperNLinear_L2", input_length=256, output_length=256, metrics=None):
        super().__init__(name, True)
        self.input_length = input_length
        self.output_length = output_length
        self.metrics = metrics 
        
    def build(self, hp):
        model = NLinear_L2(self.input_length, self.output_length)
        lr = hp.Float("learning_rate", default=1e-3, min_value=1e-4, max_value=1e-2, step=10, sampling="log")
        b1 = hp.Fixed("beta_1", 0.9)
        b2 = hp.Fixed("beta_2", 0.999)
        eps = hp.Fixed("epsilon", 1e-7)
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer = keras.optimizer.Adam(learning_rate = lr, beta_1=b1, beta_2=b2, epsilon=eps),
            metrics=self.metrics
        )
        return model
    


class NLinear_HL(keras.Model):
    def __init__(self, y_min, y_max, padding, n_bins, sigma, input_length, output_length, name="NLinear_HL"):
        super().__init__()
        y_range = y_max-y_min
        centers = tf.linspace(y_min-(y_range*padding), y_max+(y_range*padding), n_bins)
        self.dense = layers.Dense(output_length*tf.size(centers))
        self.transform = TruncGaussHistTransform(centers, sigma)
        self.reshape = layers.Reshape((output_length, tf.size(centers)))
        self.softmax = layers.Softmax()
        self.mean = HistMean(centers)
        self.hist_loss = keras.metrics.Mean("loss")
        
    
    def call(self, inputs, training=None):
        seq_last = inputs[:,-1].numpy()[:,np.newaxis]*np.ones(tf.size(inputs))
        x = inputs - seq_last
        x = self.dense(x, training=training)
        x = self.reshape(x, training=training)
        x = self.softmax(x, training=training)
        return self.mean(x, training=training) + seq_last
    
    def train_step(self, data):
        inputs, outputs = data
        
        seq_last = inputs[:,-1].numpy()[:,np.newaxis]*np.ones(tf.size(inputs))
        x = inputs - seq_last
        
        y = outputs - seq_last
        y_transformed = self.transform(y)
        
        with tf.GradientTape() as tape:
            features = self.dense(x, training=True)
            features = self.reshape(features, training=True)
            hist = self.softmax(features, training=True)
            loss = keras.losses.categorical_crossentropy(y_transformed, hist)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        y_pred = self.mean(hist)
        self.compiled_metrics.update_state(y, y_pred)
        self.hist_loss.update_state(loss)

        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        inputs, outputs = data
        
        seq_last = inputs[:,-1].numpy()[:,np.newaxis]*np.ones(tf.size(inputs))
        x = inputs - seq_last
        
        y = outputs - seq_last
        y_transformed = self.transform(y)
        
        
        features = self.dense(x, training=False)
        features = self.reshape(features, training=False)
        hist = self.softmax(features, training=False)
        
        loss = keras.losses.categorical_crossentropy(y_transformed, hist)
        self.hist_loss.update_state(loss)

        y_pred = self.mean(hist)
        self.compiled_metrics.update_state(y, y_pred)
        
        return {m.name: m.result() for m in self.metrics}
    
    
    

class HyperNLinear_HL(kt.HyperModel):
    def __init__(self, y_min, y_max, name="Hyper_NLinear_HL", input_length=256, output_length=256, metrics=None, ):
        super().__init__(name, True)
        self.metrics = metrics
        self.input_length = input_length
        self.output_length = output_length
        self.y_min = y_min
        self.y_max = y_max
        
        
    def build(self, hp):
        model = self.get_model(hp)
        lr = hp.Float("learning_rate", default=1e-3, min_value=1e-4, max_value=1e-2, step=10, sampling="log")
        b1 = hp.Fixed("beta_1", 0.9)
        b2 = hp.Fixed("beta_2", 0.999)
        eps = hp.Fixed("epsilon", 1e-7)
        model.compile(
            loss=keras.losses.CategoricalCrossentropy(),
            optimizer = keras.optimizer.Adam(learning_rate = lr, beta_1=b1, beta_2=b2, epsilon=eps),
            metrics=self.metrics
        )
        return model
    
    def get_model(self, hp):
        sig_ratio = hp.Float("sig_ratio", default=1., min_value=0.5, max_value=2., step=2, sampling="log")
        padding = hp.Float("padding", default=0.1, min_value=0.025, max_value=0.1, step=2, sampling="log")
        n_bins = int(hp.Int("n_bins", default=100, min_value=25, max_value=400, step=2, sampling="log"))
        
        return NLinear_HL(self.y_min, self.y_max, padding, n_bins, sig_ratio, self.input_length, self.output_length)
        
    

        
        
    