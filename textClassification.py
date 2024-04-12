import numpy as np
from sklearn.model_selection import GridSearchCV
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.callbacks import ReduceLROnPlateau

# Load the MNIST dataset
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1).astype('float32') / 255.0
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1).astype('float32') / 255.0

# Define your Keras model function
def create_model(optimizer='adam', activation='relu', neurons=64):
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(neurons, activation=activation))
    model.add(Dense(10, activation='softmax'))
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# Create KerasClassifier
keras_classifier = KerasClassifier(build_fn=create_model, verbose=1)

# Define hyperparameters grid for GridSearchCV
param_grid = {
    'optimizer': ['adam', 'sgd'],
    'activation': ['relu', 'tanh'],
    'neurons': [32, 64, 128],
    'batch_size': [32, 64],
    'epochs': [5, 10]
}

# Define Callback function
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=0.001)

# Perform GridSearchCV
grid_search = GridSearchCV(estimator=keras_classifier, param_grid=param_grid, cv=3, verbose=1)
grid_result = grid_search.fit(x_train, y_train, validation_split=0.2, callbacks=[reduce_lr])

# Print best accuracy and hyperparameters
print("Best accuracy: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# Evaluate the best model on the test set
test_accuracy = grid_result.score(x_test, y_test)
print("Test accuracy:", test_accuracy)

# Access the history object to visualize training/validation loss and accuracy
history = grid_result.best_estimator_.model.history.history
# Plot training/validation loss and accuracy
# (code for plotting not included here)
