# NN to find distance of a car given it's size (bounding box) on screen

# NN to determine if a frame is useful or if it should be removed

# with open("boxes.txt") as f:
#     lines = f.readlines()

# l = []
# for line in lines:
#     split = line.split()
#     split = [int(x) for x in split]
#     l.append(tuple(split))

# l = set(l)

# with open("clean.txt", "w") as f:
#     for line in l:
#         f.write(str(line[0]) + "\t" + str(line[1]) + "\t" + str(line[2]) + "\n")

from __future__ import absolute_import, division, print_function

# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.saved_model import tag_constants

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
import os

class RegressionModel:

    def __init__(self):

        # c = self.get_file("clean.txt")
        # train_data, train_labels, test_data, test_labels = self.get_values_labels(c[:35], c[35:])
        
        checkpoint_path = "training_1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, 
                                                    save_weights_only=True,
                                                    verbose=1, period=50)
        
        self.create_model()
        
        self.model.load_weights(checkpoint_path)

        # self.model.fit(train_data, train_labels, epochs=1000, callbacks=[cp_callback])

        # test_loss, mae, mse = self.model.evaluate(test_data, test_labels)

        # print('Test accuracy:', test_loss, mae)

        # predictions = self.model.predict(test_data)

        # print(predictions)

        # print(self.model.predict(np.array([[93, 90]])))

    def create_model(self):
        self.model = keras.Sequential([
            keras.layers.Dense(80, activation=tf.nn.relu, input_shape=[2]),
            keras.layers.Dense(80, activation=tf.nn.relu),
            keras.layers.Dense(80, activation=tf.nn.relu),
            keras.layers.Dense(80, activation=tf.nn.relu),
            keras.layers.Dense(1)
        ])

        self.model.compile(optimizer=tf.keras.optimizers.RMSprop(0.001), 
                        loss='mean_squared_error', 
                        metrics=['mean_absolute_error', 'mean_squared_error'])

        return self.model

    def get_file(self, filename):
        with open(filename, "r") as f:
            return [[int(n) for n in x.split()] for x in f.readlines()]

    def get_values_labels(self, pen_train, pen_test):
        train_data = np.array([x[:-1] for x in pen_train])
        train_labels = np.array([x[-1] for x in pen_train])

        test_data = np.array([x[:-1] for x in pen_test])
        test_labels = np.array([x[-1] for x in pen_test])
        
        return train_data, train_labels, test_data, test_labels

    def get_distance(self, width, height):
        return self.model.predict(np.array([[width, height]]))
