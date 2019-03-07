from tensorflow.python.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D, BatchNormalization
import keras
from config.config import *


class Model:
    # Initialized model
    def __init__(self):
        super(Model, self).__init__()
        self.model = Sequential()
        # Add convolution 2D
        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding="same",
                              kernel_initializer='he_normal', input_shape=(IMG_ROWS, IMG_COLS, 1)))

        self.model.add(BatchNormalization())

        self.model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32, kernel_size=5, strides=2, padding='same', activation='relu'))
        self.model.add(MaxPooling2D((2, 2)))
        self.model.add(BatchNormalization())
        # Add dropouts to the self.model
        self.model.add(Dropout(0.4))
        self.model.add(Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64, kernel_size=(3, 3), strides=2, padding='same', activation='relu'))
        # Add dropouts to the self.model
        self.model.add(Dropout(0.4))
        self.model.add(Flatten())
        self.model.add(Dense(128, activation='relu'))
        # Add dropouts to the self.model
        self.model.add(Dropout(0.4))
        self.model.add(Dense(NUM_CLASSES, activation='softmax'))
        # Compile the model, with the layers and optimized defined
        self.model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

    #  Check the model initialized.
    def check_model(self):
        self.model.summary()

    # run model
    def run_model(self, x_train, y_train, x_val, y_val):
        history = self.model.fit(x_train, y_train,
                                 batch_size=BATCH_SIZE,
                                 epochs=NO_EPOCHS,
                                 verbose=1,
                                 validation_data=(x_val, y_val))

        return history

    def save_model(self, model_name):
        self.model.save(model_name)

    def load_model(self, model_name):
        self.model = load_model(model_name)

