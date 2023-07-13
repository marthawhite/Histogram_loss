import tensorflow as tf 
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd

class MultivariateHistTransform(keras.layers.Layer):
     """Layer that transforms a scalar target into a binned probability vector 
    that approximates a truncated Gaussian distribution with the target as the mean.
    
    Params:
        borders - the borders of the histogram bins as a 2d array of size (dimensions, borders)
        sigma - the sigma parameter of the truncated Gaussian distribution
    """
    def __init__(self, borders, sigma):
        super().__init__(trainable=False, name="MultivariateHistTransform")
        self.borders = borders
        self.sigma = sigma
        
    def call(self, inputs):
        
        

class HistMean(keras.layers.Layer):
    def __init__(self, centers):
        super().__init__(trainable=False, name="HistMean")
        self.centers = centers
        
    def call(self, inputs):
        x = tf.math.multiply(inputs, self.centers)
        x = tf.math.reduce_sum(x, axis=-1)
        return x


class TimeSerriesHL(keras.Model):
     def __init__(self, units, borders, sigma, num_timesteps_predicted=1):
        super().__init__()
        centers = (borders[:,:-1] + borders[:,1:]) / 2
        self.num_timesteps_predicted = num_timesteps_predicted
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(64, activation="relu")
        self.rnn_block = layers.LSTM(64, return_state=True)
        self.dense3 = layers.Dense(64, activation="relu")
        self.dense4 = layers.Dense(np.size(centers))
        self.reshape = layers.Reshape(np.shape(centers))
        
        self.loss = keras.metrics.Mean("loss")
        self.hist_transform = MultivariateHistTransform(borders, sigma)
        self.hist_mean = HistMean(centers)
        
        
    def call(self, inputs):
        predictions = []
        
        x = layers.TimeDistributed(self.dense1)(inputs)
        x = layers.TimeDistributed(self.dense2)(x)
        x = self.rnn_block(x)
        
        
        hiden_state = x[0]
        hiden_and_cell = x[1:]
        
        x = self.dense3(hiden_state)
        x = self.dense4(x)
        x = self.reshape(x) # (batchsize, units, bins)
        x = self.softmax(x)
        predictions.append(self.hist_mean(x)) # (batchsize, values)
        
        for i in range(self.num_timesteps_predicted-1):
            x = tf.indentity(predictions[-1])
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.rnn_block(x)


            hiden_state = x[0]
            hiden_and_cell = x[1:]

            x = self.dense3(hiden_state)
            x = self.dense4(x)
            x = self.reshape(x) # (batchsize, units, bins)
            x = self.softmax(x)
            predictions.append(self.hist_mean(x)) # (batchsize, values)
            
        predictions= tf.convert_to_tensor(predictions) #  time major:(timesteps, batches values)
        predictions= tf.transpose(predictions, [1,0,2]) # batch major
        return predictions
    
    
    def train_step(self, data):
        x, y = data # y should be data from one time step 
        # y (batchsize, value)
        targets = self.MultivariateHistTransform(y)
        with tf.GradientTape() as tape:
            x = layers.TimeDistributed(self.dense1)(x)
            x = layers.TimeDistributed(self.dense2)(x)
            x = self.rnn_block(x)
            x = self.dense3(x[0])
            x = self.dense4(x)
            x = self.reshape(x) # (batchsize, units, bins)
            x = self.softmax(x)
            predictions = self.hist_mean(x)
            
            loss = keras.losses.categorical_crossentropy(targets, predict)
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics} 
    
    
    def test_step(self, data):
        x, y = data
        predictions = []
        
        x = layers.TimeDistributed(self.dense1)(inputs)
        x = layers.TimeDistributed(self.dense2)(x)
        x = self.rnn_block(x)
        
        
        hiden_state = x[0]
        hiden_and_cell = x[1:]
        
        x = self.dense3(hiden_state)
        x = self.dense4(x)
        x = self.reshape(x) # (batchsize, units, bins)
        x = self.softmax(x)
        predictions.append(self.hist_mean(x)) # (batchsize, values)
        
        for i in range(self.num_timesteps_predicted-1):
            x = tf.indentity(predictions[-1])
            x = self.dense1(x)
            x = self.dense2(x)
            x = self.rnn_block(x)


            hiden_state = x[0]
            hiden_and_cell = x[1:]

            x = self.dense3(hiden_state)
            x = self.dense4(x)
            x = self.reshape(x) # (batchsize, units, bins)
            x = self.softmax(x)
            predictions.append(self.hist_mean(x)) # (batchsize, values)
            
        predictions= tf.convert_to_tensor(predictions) #  time major:(timesteps, batches values)
        predictions= tf.transpose(predictions, [1,0,2]) # batch major
        
        loss = keras.losses.mean_squared_error(predictions, y)
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    


        

class TimeSerriesRegression(keras.Model):
    def __init__(self, units, num_timesteps_predicted=1):
        super().__init__()
        self.num_timesteps_predicted = num_timesteps_predicted
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(64, activation="relu")
        self.rnn_block = layers.LSTM(64, return_state=True)
        self.dense3 = layers.Dense(64, activation="relu")
        self.dense4 = layers.Dense(units)
        
        self.loss = keras.metrics.Mean("loss")
        
    def call(self, inputs):
        predictions = []
        
        x = layers.TimeDistributed(self.dense1)(inputs)
        x = layers.TimeDistributed(self.dense2)(x)
        x = self.rnn_block(x)
        
        hiden_state = x[0]
        hiden_and_cell = x[1:]
        
        x = self.dense3(hiden_state)
        predictions.append(self.dense4(x))
        
        for i in range(self.num_timesteps_predicted-1):
            x = tf.identity(predictions[-1])
            x = self.dense1(x)
            x = self.dense2(x)
            x = tf.expand_dims(x, axis =1)
            x = self.rnn_block(x, initial_state=hiden_and_cell)
            hiden_state = x[0]
            hiden_and_cell = x[1:]
            x = self.dense3(hiden_state)
            predictions.append(self.dense4(x))
            
        predictions= tf.convert_to_tensor(predictions) #  time major:(timesteps, batches values)
        predictions= tf.transpose(predictions, [1,0,2]) # batch major
        return predictions
    
    def train_step(self, data):
        x, y = data # y should be data from one time step 
        # y (batchsize, value)
        
        
        with tf.GradientTape() as tape:
            x = layers.TimeDistributed(self.dense1)(x)
            x = layers.TimeDistributed(self.dense2)(x)
            x = self.rnn_block(x)
            x = self.dense3(x[0])
            predict = self.dense4(x)
            loss = keras.losses.mean_squared_error(predict, y)
            
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
    
    def test_step(self, data):
        x, y = data # batch major (batch, timesteps, values)
        predictions = []
        
        x = layers.TimeDistributed(self.dense1)(x)
        x = layers.TimeDistributed(self.dense2)(x)
        x = self.rnn_block(x)
        
        hiden_state = x[0]
        hiden_and_cell = x[1:]
        
        x = self.dense3(hiden_state)
        predictions.append(self.dense4(x))
        
        for i in range(self.num_timesteps_predicted-1):
            x = tf.identity(predictions[-1])
            x = self.dense1(x)
            x = self.dense2(x)
            x = tf.expand_dims(x, axis =1)
            x = self.rnn_block(x, initial_state=hiden_and_cell)
            hiden_state = x[0]
            hiden_and_cell = x[1:]
            x = self.dense3(hiden_state)
            predictions.append(self.dense4(x))
            
        predictions= tf.convert_to_tensor(predictions) #  time major:(timesteps, batches values)
        predictions= tf.transpose(predictions, [1,0,2]) # batch major
        
        loss = keras.losses.mean_squared_error(predictions, y)
        
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
            else:
                metric.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}


    
        
def get_time_series_dataset(filename, drop=[], seq_len=10, pred_len=10, test_size=0.2, batch_size=64):
    # test_size is the portion of the dataset to use as test data must be between 0 and 1
    df = pd.read_csv(filename)
    df = df.drop(drop, axis = 1)
    n = df.shape[0] 
    split = round((1-test_size)*n)
    data = df.values
    train = data[:split]
    test = data[split:]
    inputs = train[:-1]
    target = train[seq_len:]
    ds_train = keras.utils.timeseries_dataset_from_array(data=inputs, targets=target, sequence_length=seq_len, batch_size=batch_size)
    inputs = data[:-(pred_len)]
    targets = data[seq_len:]
    x = keras.utils.timeseries_dataset_from_array(inputs, None, seq_len, batch_size=None)
    y = keras.utils.timeseries_dataset_from_array(targets, None, pred_len, batch_size=None)
    ds_test = tf.data.Dataset.zip((x,y)).batch(batch_size)
    
    return ds_train, ds_test
        
def main():

    train, test = get_time_series_dataset("ETTh1.csv", "date", batch_size=64)
    
    model = TimeSerriesRegression(7,10)
    model.compile(
        optimizer="Adam"
    )
    history = model.fit(train)
    print(history.history)
    print(model.evaluate(test))
    
        
if __name__ == "__main__":
    main()
     