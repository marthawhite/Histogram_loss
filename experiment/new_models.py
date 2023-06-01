from tensorflow import keras
import tensorflow as tf

class TruncGaussHistTransform(keras.layers.Layer):
    def __init__(self, borders, sigma):
        super().__init__(trainable=False, name="TruncGaussHistTransform")
        self.borders = borders
        self.sigma = sigma

    def call(self, inputs):
        border_targets = self.adjust_and_erf(self.borders, tf.expand_dims(inputs, 1), self.sigma)
        two_z = border_targets[:, -1] - border_targets[:, 0]
        x_transformed = (border_targets[:, 1:] - border_targets[:, :-1]) / tf.expand_dims(two_z, 1)
        return x_transformed

    def adjust_and_erf(self, a, mu, sig):
        return tf.math.erf((a - mu)/(tf.math.sqrt(2.0)*sig))
    
class OneHotTransform(keras.layers.Layer):

    def __init__(self, borders):
        super().__init__(trainable=False, name="OneHotTransform")
        self.borders = borders
        self.bin_size = borders[1] - borders[0]
        self.low = tf.reduce_min(borders)
        self.n_classes = tf.size(borders) - 1

    def call(self, inputs):
        adjusted = (inputs - self.low) / self.bin_size
        indices = tf.cast(adjusted, tf.int32)
        return tf.one_hot(indices, self.n_classes, dtype=tf.float32)
    

class HistMean(keras.layers.Layer):

    def __init__(self, centers):
        super().__init__(trainable=False, name="HistMean")
        self.centers = centers

    def call(self, inputs):
        return tf.linalg.matvec(inputs, self.centers)


class Regression(keras.Model):
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.reg = keras.layers.Dense(1)

    def call(self, inputs, training=None):
        features = self.base(inputs, training=training)
        return self.reg(features)


class HistModel(keras.Model):
    def __init__(self, base, centers, transform, name="HistModel"):
        super().__init__(name=name)
        self.base = base
        self.softmax = keras.layers.Dense(tf.size(centers), activation="softmax")
        self.transform = transform
        self.mean = HistMean(centers)

    def call(self, inputs, training=None):
        features = self.base(inputs, training)
        hist = self.softmax(features, training=training)
        return self.mean(hist)

    def train_step(self, data):
        x, y = data
        y_transformed = self.transform(y)

        with tf.GradientTape() as tape:
            features = self.base(x, training=True)
            hist = self.softmax(features, training=True)
            loss = keras.losses.categorical_crossentropy(y_transformed, hist)
        
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        y_pred = self.mean(hist)
        self.compiled_metrics.update_state(y, y_pred)

        return {m.name: m.result() for m in self.metrics}
    
class HLGaussian(HistModel):

    def __init__(self, base, borders, sigma):
        centers = (borders[:-1] + borders[1:]) / 2
        transform = TruncGaussHistTransform(borders, sigma)
        super().__init__(base, centers, transform, "HL-Gaussian")


class HLOneBin(HistModel):

    def __init__(self, base, borders):
        centers = (borders[:-1] + borders[1:]) / 2
        transform = OneHotTransform(borders)
        super().__init__(base, centers, transform, "HL-OneBin")
    

def main():
    base = keras.layers.Dense(100)
    borders = tf.linspace(0, 100, 100)
    hl = HLGaussian(base, borders, 1.)
    hl.compile()
    print(hl.summary())




if __name__ == "__main__":
    main()
