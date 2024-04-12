import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.wrappers.scikit_learn import KerasClassifier

# Define your Keras model function
def create_model(optimizer='adam', activation='relu', neurons=64):
    model = Sequential()
    model.add(Flatten(input_shape=(28, 28)))
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Load the MNIST dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0

# Create KerasClassifier
keras_classifier = KerasClassifier(build_fn=create_model, epochs=5, batch_size=32, verbose=1)

# Define hyperparameters grid for GridSearchCV
param_grid = {
    'optimizer': ['adam', 'sgd'],
    'activation': ['relu', 'tanh'],
    'neurons': [32, 64, 128]
}

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=keras_classifier, param_grid=param_grid, cv=3)
grid_search.fit(x_train, y_train)

# Print b
