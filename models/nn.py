'''
A dummy db storaing faces in memory
Feel free to make it fancier like hooking with postgres or whatever
This model here is just for simple demo app under apps
Don't use it for production dude.
'''
import numpy as np
import keras
import tensorflow as tf
import matplotlib.pyplot as plt
class Model(object):

    def __init__(self, output_shape=None, path_to_model=None):
        self.new_Graph = tf.Graph()
        with self.new_Graph.as_default():
            self.tfSess = tf.Session(graph=self.new_Graph)
            if path_to_model is None:
                self.model = keras.Sequential([
                    keras.layers.Dense(128, activation=tf.nn.sigmoid, name="hidden"),
                    keras.layers.Dense(output_shape, activation=tf.nn.softmax, name="output")
                ])
            else:
                self.model = keras.models.load_model(path_to_model)

            self.model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss="sparse_categorical_crossentropy", metrics=['accuracy'], batch_size=64)


    def train_model_from_weights(self, path_to_weight):

        with self.new_Graph.as_default():
            self.model.fit(np.array([]), np.array([]), epochs=0)
            self.model.load_weights(path_to_weight)


    def train_model_from_data(self, X, Y, epoch=800):

        X = np.append(X, np.reshape(Y, (len(Y), 1)), axis=1)
        np.random.shuffle(X)
        Y = X[:,-1].astype(int)
        X = np.delete(X, -1, axis=1)
        with self.new_Graph.as_default():
            history = self.model.fit(X, Y, epochs=epoch, validation_split=0.2)

            plt.figure(figsize=(10, 8))

            val = plt.plot(history.epoch, history.history['val_loss'],
                           '--', label=' val_loss')
            plt.plot(history.epoch, history.history['loss'], color=val[0].get_color(),
                     label='Train')

            plt.show()



    def add_class(self):
        with self.new_Graph.as_default():
            weights = self.model.get_layers[-1].get_weights()
            weights[0] = np.array([np.append(weights_ligne, np.random.normal(0, 0.01, 1)) for weights_ligne in weights[0]])
            weights[1] = np.append(weights[1], 0)
            self.model.summary()
            self.model.pop()
            self.model.summary()
            new_layer = keras.layers.Dense(weights[1].shape[0], activation=tf.nn.softmax, name="output")
            self.model.add(new_layer)
            self.model.get_layer("output").set_weights(weights)
            self.model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), loss="sparse_categorical_crossentropy", metrics=['accuracy'])
            self.model.summary()

    def save_weights(self, path):
        self.model.save_weights(path)

    def save_model(self, path):
        self.model.save(path)

    def __str__(self):
        return self.model.summary()



