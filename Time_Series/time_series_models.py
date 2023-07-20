import tensorflow as tf 
from tensorflow import keras
from keras import layers
import numpy as np
import pandas as pd
import sys
import json



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
        k = len(self.borders.shape)
        self.perm_out = list(range(1, k+1)) + [0]

    def call(self, inputs):
        """Transform the input and return it.
        
        Params:
            inputs - the tensor of targets to transform

        Returns:
            x_transformed - a tensor of shape (len(inputs), len(borders) - 1)
            consisting of the probability vectors for each target
        """
        border_targets = self.adjust_and_erf(tf.expand_dims(self.borders, 1), inputs, self.sigma)
        two_z = border_targets[-1] - border_targets[0]
        x_transformed = (border_targets[1:] - border_targets[:-1]) / two_z
        return tf.transpose(x_transformed, self.perm_out)
    
    def adjust_and_erf(self, a, mu, sig):
        """Calculate the erf of a after standardizing and dividing by sqrt(2)."""
        return tf.math.erf((a - mu)/(tf.math.sqrt(2.0)*sig))


class HistMean(keras.layers.Layer):
    def __init__(self, centers):
        super().__init__(trainable=False, name="HistMean")
        self.centers = centers
        
    def call(self, inputs):
        x = tf.math.multiply(inputs, self.centers)
        x = tf.math.reduce_sum(x, axis=-1)
        return x


class TimeSerriesHL(keras.Model):
    def __init__(self, units, data_min, data_max, bins, train_len=20, pred_loops=36):
        super().__init__()
        if bins < 10:
            bins = 10
        data_range = data_max-data_min
        sigma = 4*(data_range)/(5*bins - 48)
        true_min = data_min - 6*sigma
        true_max = data_max + 6*sigma
        borders = tf.linspace(true_min, true_max, bins+1)
        centers = tf.transpose(borders, [1,0])
        centers = (centers[:,:-1] + centers[:,1:]) / 2
        
        self.pred_loops = pred_loops
        self.target_reshape = layers.Reshape((units*train_len,))
        self.predict_reshape = layers.Reshape((train_len, units))
        
        self.test_targets_reshape = layers.Reshape((pred_loops*train_len*units,))
        self.test_predict_reshape = layers.Reshape((pred_loops*train_len*units,))
        
        self.dense1 = layers.Dense(128, activation="relu")
        self.batchnorm1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.5)
        
        self.dense2 = layers.Dense(128, activation="relu")
        self.batchnorm2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.5)
        
        self.dense3 = layers.Dense(128, activation="relu")
        self.batchnorm3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(0.5)
        
        self.dense4 = layers.Dense(128, activation="relu")
        self.batchnorm4 = layers.BatchNormalization()
        self.dropout4 = layers.Dropout(0.5)
        
        self.dense5 = layers.Dense(128, activation="relu")
        self.batchnorm5 = layers.BatchNormalization()
        self.dropout5 = layers.Dropout(0.5)
        
        self.rnn_block = layers.LSTM(128, return_state=True)
        self.batchnorm6 = layers.BatchNormalization()
        self.dropout6 = layers.Dropout(0.5)
        
        self.dense6 = layers.Dense(128, activation="relu")
        self.batchnorm7 = layers.BatchNormalization()
        self.dropout7 = layers.Dropout(0.5)
        
        self.dense7 = layers.Dense(128, activation="relu")
        self.batchnorm8 = layers.BatchNormalization()
        self.dropout8 = layers.Dropout(0.5)
        
        self.dense8 = layers.Dense(128, activation="relu")
        self.batchnorm9 = layers.BatchNormalization()
        self.dropout9 = layers.Dropout(0.5)
        
        self.dense9 = layers.Dense(128, activation="relu")
        self.batchnorm10 = layers.BatchNormalization()
        self.dropout10 = layers.Dropout(0.5)
        
        self.dense10 = layers.Dense(train_len*units*bins)
        self.reshape = layers.Reshape((train_len*units, bins))
        self.softmax = layers.Softmax()
        
        
        self.hist_transform = TruncGaussHistTransform(borders, sigma)
        self.hist_mean = HistMean(centers)
        
        self.loss = keras.metrics.Mean("loss")
        self.mse = keras.metrics.Mean("mse")
        self.mae = keras.metrics.Mean("mae")
        
    
    def train_step(self, data):
        x, y = data # y (batchsize, timesteps, units)
        y = self.target_reshape(y) # (batch, units*train_len)
        targets = self.hist_transform(y) # (batch, units*train_len, bins)
        with tf.GradientTape() as tape:
            x = layers.TimeDistributed(self.dense1)(x)
            x = layers.TimeDistributed(self.batchnorm1)(x)
            x = layers.TimeDistributed(self.dropout1)(x)

            x = layers.TimeDistributed(self.dense2)(x)
            x = layers.TimeDistributed(self.batchnorm2)(x)
            x = layers.TimeDistributed(self.dropout2)(x)

            x = layers.TimeDistributed(self.dense3)(x)
            x = layers.TimeDistributed(self.batchnorm3)(x)
            x = layers.TimeDistributed(self.dropout3)(x)

            x = layers.TimeDistributed(self.dense4)(x)
            x = layers.TimeDistributed(self.batchnorm4)(x)
            x = layers.TimeDistributed(self.dropout4)(x)

            x = layers.TimeDistributed(self.dense5)(x)
            x = layers.TimeDistributed(self.batchnorm5)(x)
            x = layers.TimeDistributed(self.dropout5)(x)

            x = self.rnn_block(x)
            hiden_state = x[0]
            hiden_and_cell = x[1:]
            x = self.batchnorm6(hiden_state)
            x = self.dropout6(x)

            x = self.dense6(x)
            x = self.batchnorm7(x)
            x = self.dropout7(x)

            x = self.dense7(x)
            x = self.batchnorm8(x)
            x = self.dropout8(x)

            x = self.dense8(x)
            x = self.batchnorm9(x)
            x = self.dropout9(x)

            x = self.dense9(x)
            x = self.batchnorm10(x)
            x = self.dropout10(x)

            x = self.dense10(x) # (batch, train_len * units * bins)
            x = self.reshape(x) # (batch, train_len * units, bins)
            x = self.softmax(x)
            predictions = self.hist_mean(x)
            
            loss = keras.losses.categorical_crossentropy(targets, x)
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        mse = keras.losses.mean_squared_error(y, predictions)
        mae = keras.losses.mean_absolute_error(y, predictions)
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            elif metric.name == "mse":
                metric.update_state(mse)
            else:
                metric.update_state(mae)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics} 
    
    
    def test_step(self, data):
        x, y = data
        y = self.test_targets_reshape(y) # (batch, pred_loops*units*train_len)
        predictions = []
        
        x = layers.TimeDistributed(self.dense1)(x)
        x = layers.TimeDistributed(self.batchnorm1)(x)
        x = layers.TimeDistributed(self.dropout1)(x)
        
        x = layers.TimeDistributed(self.dense2)(x)
        x = layers.TimeDistributed(self.batchnorm2)(x)
        x = layers.TimeDistributed(self.dropout2)(x)
        
        x = layers.TimeDistributed(self.dense3)(x)
        x = layers.TimeDistributed(self.batchnorm3)(x)
        x = layers.TimeDistributed(self.dropout3)(x)
        
        x = layers.TimeDistributed(self.dense4)(x)
        x = layers.TimeDistributed(self.batchnorm4)(x)
        x = layers.TimeDistributed(self.dropout4)(x)
        
        x = layers.TimeDistributed(self.dense5)(x)
        x = layers.TimeDistributed(self.batchnorm5)(x)
        x = layers.TimeDistributed(self.dropout5)(x)
        
        x = self.rnn_block(x)
        hiden_state = x[0]
        hiden_and_cell = x[1:]
        x = self.batchnorm6(hiden_state)
        x = self.dropout6(x)
        
        x = self.dense6(x)
        x = self.batchnorm7(x)
        x = self.dropout7(x)
        
        x = self.dense7(x)
        x = self.batchnorm8(x)
        x = self.dropout8(x)
        
        x = self.dense8(x)
        x = self.batchnorm9(x)
        x = self.dropout9(x)
        
        x = self.dense9(x)
        x = self.batchnorm10(x)
        x = self.dropout10(x)
        
        x = self.dense10(x)
        x = self.reshape(x) # (batchsize, units, bins)
        x = self.softmax(x)
        predictions.append(self.hist_mean(x)) # (batchsize, values)
        
        for i in range(self.pred_loops-1):
            x = tf.identity(predictions[-1])
            x = self.predict_reshape(x)
            
            x = layers.TimeDistributed(self.dense1)(x)
            x = layers.TimeDistributed(self.batchnorm1)(x)
            x = layers.TimeDistributed(self.dropout1)(x)

            x = layers.TimeDistributed(self.dense2)(x)
            x = layers.TimeDistributed(self.batchnorm2)(x)
            x = layers.TimeDistributed(self.dropout2)(x)

            x = layers.TimeDistributed(self.dense3)(x)
            x = layers.TimeDistributed(self.batchnorm3)(x)
            x = layers.TimeDistributed(self.dropout3)(x)

            x = layers.TimeDistributed(self.dense4)(x)
            x = layers.TimeDistributed(self.batchnorm4)(x)
            x = layers.TimeDistributed(self.dropout4)(x)

            x = layers.TimeDistributed(self.dense5)(x)
            x = layers.TimeDistributed(self.batchnorm5)(x)
            x = layers.TimeDistributed(self.dropout5)(x)
        
            
            #x = tf.expand_dims(x, axis =1)
            x = self.rnn_block(x, initial_state=hiden_and_cell)
            hiden_state = x[0]
            hiden_and_cell = x[1:]
            x = self.batchnorm6(hiden_state)
            x = self.dropout6(x)

            x = self.dense6(x)
            x = self.batchnorm7(x)
            x = self.dropout7(x)

            x = self.dense7(x)
            x = self.batchnorm8(x)
            x = self.dropout8(x)

            x = self.dense8(x)
            x = self.batchnorm9(x)
            x = self.dropout9(x)

            x = self.dense9(x)
            x = self.batchnorm10(x)
            x = self.dropout10(x)

            x = self.dense10(x)
            x = self.reshape(x) # (batchsize, units, bins)
            x = self.softmax(x)
            predictions.append(self.hist_mean(x)) # (batchsize, values)
            
        predictions= tf.convert_to_tensor(predictions) #  time major:(timesteps, batches values)
        predictions= tf.transpose(predictions, [1,0,2]) # batch major
        predictions = self.test_predict_reshape(predictions)
        
        loss = keras.losses.mean_squared_error(predictions, y)
        mse = keras.losses.mean_squared_error(y, predictions)
        mae = keras.losses.mean_absolute_error(y, predictions)
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            elif metric.name == "mse":
                metric.update_state(mse)
            else:
                metric.update_state(mae)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    


        

class TimeSerriesRegression(keras.Model):
    def __init__(self, units, train_len=20, pred_loops = 36):
        super().__init__()
        self.pred_loops = pred_loops
        self.target_reshape = layers.Reshape((units*train_len,))
        self.predict_reshape = layers.Reshape((train_len, units))
        
        self.test_targets_reshape = layers.Reshape((pred_loops*train_len*units,))
        self.test_predict_reshape = layers.Reshape((pred_loops*train_len*units,))
        
        self.dense1 = layers.Dense(256, activation="relu")
        self.batchnorm1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(0.5)
        
        self.dense2 = layers.Dense(512, activation="relu")
        self.batchnorm2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(0.5)
        
        self.dense3 = layers.Dense(512, activation="relu")
        self.batchnorm3 = layers.BatchNormalization()
        self.dropout3 = layers.Dropout(0.5)
        
        self.dense4 = layers.Dense(512, activation="relu")
        self.batchnorm4 = layers.BatchNormalization()
        self.dropout4 = layers.Dropout(0.5)
        
        self.dense5 = layers.Dense(512, activation="relu")
        self.batchnorm5 = layers.BatchNormalization()
        self.dropout5 = layers.Dropout(0.5)
        
        self.rnn_block = layers.LSTM(512, return_state=True)
        self.batchnorm6 = layers.BatchNormalization()
        self.dropout6 = layers.Dropout(0.5)
        
        self.dense6 = layers.Dense(512, activation="relu")
        self.batchnorm7 = layers.BatchNormalization()
        self.dropout7 = layers.Dropout(0.5)
        
        self.dense7 = layers.Dense(512, activation="relu")
        self.batchnorm8 = layers.BatchNormalization()
        self.dropout8 = layers.Dropout(0.5)
        
        self.dense8 = layers.Dense(512, activation="relu")
        self.batchnorm9 = layers.BatchNormalization()
        self.dropout9 = layers.Dropout(0.5)
        
        self.dense9 = layers.Dense(512, activation="relu")
        self.batchnorm10 = layers.BatchNormalization()
        self.dropout10 = layers.Dropout(0.5)
        
        self.dense10 = layers.Dense(units*train_len)
        
        self.loss = keras.metrics.Mean("loss")
        self.mse = keras.metrics.Mean("mse")
        self.mae = keras.metrics.Mean("mae")
        
    
    def train_step(self, data):
        x, y = data # y should be data from one time step 
        # y (batchsize, value)
        
        targets = self.target_reshape(y) #(batch_size, train_len*units)
        
        with tf.GradientTape() as tape:
            x = layers.TimeDistributed(self.dense1)(x)
            x = layers.TimeDistributed(self.batchnorm1)(x)
            x = layers.TimeDistributed(self.dropout1)(x)

            x = layers.TimeDistributed(self.dense2)(x)
            x = layers.TimeDistributed(self.batchnorm2)(x)
            x = layers.TimeDistributed(self.dropout2)(x)

            x = layers.TimeDistributed(self.dense3)(x)
            x = layers.TimeDistributed(self.batchnorm3)(x)
            x = layers.TimeDistributed(self.dropout3)(x)

            x = layers.TimeDistributed(self.dense4)(x)
            x = layers.TimeDistributed(self.batchnorm4)(x)
            x = layers.TimeDistributed(self.dropout4)(x)

            x = layers.TimeDistributed(self.dense5)(x)
            x = layers.TimeDistributed(self.batchnorm5)(x)
            x = layers.TimeDistributed(self.dropout5)(x)

            x = self.rnn_block(x)
            hiden_state = x[0]
            hiden_and_cell = x[1:]
            x = self.batchnorm6(hiden_state)
            x = self.dropout6(x)

            x = self.dense6(x)
            x = self.batchnorm7(x)
            x = self.dropout7(x)

            x = self.dense7(x)
            x = self.batchnorm8(x)
            x = self.dropout8(x)

            x = self.dense8(x)
            x = self.batchnorm9(x)
            x = self.dropout9(x)

            x = self.dense9(x)
            x = self.batchnorm10(x)
            x = self.dropout10(x)

            predict = self.dense10(x) # (batch_size, units*train_len)
            loss = keras.losses.mean_squared_error(targets, predict)
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        mae = keras.losses.mean_absolute_error(targets, predict)
        
        for metric in self.metrics:
            if metric.name == "mae":
                metric.update_state(mae)
            else:
                metric.update_state(loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data # y:(batchsize, predict_len, units)
        y = self.test_targets_reshape(y)
        predictions = []
        
        x = layers.TimeDistributed(self.dense1)(x)
        x = layers.TimeDistributed(self.batchnorm1)(x)
        x = layers.TimeDistributed(self.dropout1)(x)
        
        x = layers.TimeDistributed(self.dense2)(x)
        x = layers.TimeDistributed(self.batchnorm2)(x)
        x = layers.TimeDistributed(self.dropout2)(x)
        
        x = layers.TimeDistributed(self.dense3)(x)
        x = layers.TimeDistributed(self.batchnorm3)(x)
        x = layers.TimeDistributed(self.dropout3)(x)
        
        x = layers.TimeDistributed(self.dense4)(x)
        x = layers.TimeDistributed(self.batchnorm4)(x)
        x = layers.TimeDistributed(self.dropout4)(x)
        
        x = layers.TimeDistributed(self.dense5)(x)
        x = layers.TimeDistributed(self.batchnorm5)(x)
        x = layers.TimeDistributed(self.dropout5)(x)
        
        x = self.rnn_block(x)
        hiden_state = x[0]
        hiden_and_cell = x[1:]
        x = self.batchnorm6(hiden_state)
        x = self.dropout6(x)
        
        x = self.dense6(x)
        x = self.batchnorm7(x)
        x = self.dropout7(x)
        
        x = self.dense7(x)
        x = self.batchnorm8(x)
        x = self.dropout8(x)
        
        x = self.dense8(x)
        x = self.batchnorm9(x)
        x = self.dropout9(x)
        
        x = self.dense9(x)
        x = self.batchnorm10(x)
        x = self.dropout10(x)
        
        x = self.dense10(x) # (batchsize, units*train_len)
        
        predictions.append(x) # (timestep, batchsize, units*train_len)
        
        for i in range(self.pred_loops-1):
            x = tf.identity(predictions[-1])
            x = self.predict_reshape(x) # (batch_size, train_len, units)
            
            x = layers.TimeDistributed(self.dense1)(x)
            x = layers.TimeDistributed(self.batchnorm1)(x)
            x = layers.TimeDistributed(self.dropout1)(x)

            x = layers.TimeDistributed(self.dense2)(x)
            x = layers.TimeDistributed(self.batchnorm2)(x)
            x = layers.TimeDistributed(self.dropout2)(x)

            x = layers.TimeDistributed(self.dense3)(x)
            x = layers.TimeDistributed(self.batchnorm3)(x)
            x = layers.TimeDistributed(self.dropout3)(x)

            x = layers.TimeDistributed(self.dense4)(x)
            x = layers.TimeDistributed(self.batchnorm4)(x)
            x = layers.TimeDistributed(self.dropout4)(x)
            
            x = layers.TimeDistributed(self.dense5)(x)
            x = layers.TimeDistributed(self.batchnorm5)(x)
            x = layers.TimeDistributed(self.dropout5)(x)
            
            #x = tf.expand_dims(x, axis =1)
            x = self.rnn_block(x, initial_state=hiden_and_cell)
            hiden_state = x[0]
            hiden_and_cell = x[1:]
            x = self.batchnorm6(hiden_state)
            x = self.dropout6(x)

            x = self.dense6(x)
            x = self.batchnorm7(x)
            x = self.dropout7(x)

            x = self.dense7(x)
            x = self.batchnorm8(x)
            x = self.dropout8(x)

            x = self.dense8(x)
            x = self.batchnorm9(x)
            x = self.dropout9(x)

            x = self.dense9(x)
            x = self.batchnorm10(x)
            x = self.dropout10(x)

            x = self.dense10(x)
            predictions.append(x) # (batchsize, train_len*units)
            
        predictions= tf.convert_to_tensor(predictions) #  time major:(pred_loops, batches, train_len*units)
        predictions= tf.transpose(predictions, [1,0,2]) # batch major: (batches, pred_loops, train_len*units)
        predictions = self.test_predict_reshape(predictions)
        
        loss = keras.losses.mean_squared_error(predictions, y)
        mae = keras.losses.mean_absolute_error(predictions, y)
        
        for metric in self.metrics:
            if metric.name == "mae":
                metric.update_state(mae)
            else:
                metric.update_state(loss)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    
        
def get_time_series_dataset(filename, drop=[], seq_len=720, train_len=20, pred_len=720, test_size=0.2, batch_size=64):
    # test_size is the portion of the dataset to use as test data must be between 0 and 1
    df = pd.read_csv(filename)
    df = df.drop(drop, axis = 1)
    mean = df.mean()
    std = df.std()
    df = (df-mean)/std
    data_max = np.float32(df.max(axis=0).values)
    data_min = np.float32(df.min(axis=0).values)
    n = df.shape[0] 
    split = round((1-test_size)*n)
    data = df.values
    train = data[:split]
    test = data[split:]
    inputs = train[:-(train_len)]
    target = train[seq_len:]
    x_train = keras.utils.timeseries_dataset_from_array(inputs, None, seq_len, batch_size=None)
    y_train = keras.utils.timeseries_dataset_from_array(target, None, train_len, batch_size=None)
    ds_train = tf.data.Dataset.zip((x_train, y_train)).batch(batch_size)
    #ds_train = keras.utils.timeseries_dataset_from_array(data=inputs, targets=target, sequence_length=seq_len, batch_size=batch_size)
    inputs = data[:-(pred_len)]
    targets = data[seq_len:]
    x = keras.utils.timeseries_dataset_from_array(inputs, None, seq_len, batch_size=None)
    y = keras.utils.timeseries_dataset_from_array(targets, None, pred_len, batch_size=None)
    ds_test = tf.data.Dataset.zip((x,y)).batch(batch_size)
    
    return ds_train, ds_test, data_max, data_min
        
def main():
    
    
    n_epochs = 1
    learning_rate = 1e-4
    
    
    train, test, data_max, data_min= get_time_series_dataset("ETTm2.csv", drop="date")
    data_max = np.array(data_max.tolist()*20, dtype=np.float32)
    data_min = np.array(data_min.tolist()*20, dtype=np.float32)
    
    model = TimeSerriesHL(units=7, data_min=data_min, data_max=data_max, bins=100)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    )
    
    
    TimeSerriesHLHistory =  model.fit(x=train, epochs=n_epochs, validation_data=test, verbose=2)
    with open("TSHL20.json", "w") as file:
        json.dump(TimeSerriesHLHistory.history, file)
        
    
    train, test, data_max, data_min= get_time_series_dataset("ETTm2.csv", drop="date")
    
    model = TimeSerriesRegression(units=7)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
    )
    TimeSerriesRegressionHistory =  model.fit(x=train, epochs=n_epochs, validation_data=test, verbose=2)
    with open("TSregression20.json", "w") as file:
        json.dump(TimeSerriesRegressionHistory.history, file)
    
    
    
  
    

    
        
if __name__ == "__main__":
    main()
     