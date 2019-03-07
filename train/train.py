from tensorflow.python import keras
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import tqdm
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from config.config import *
from model.model import Model

if (IS_LOCAL):
    PATH = "../input/dataset/"
else:
    PATH = "../input/"

print(os.listdir(PATH))

# load train data
train_images = np.load(PATH + 'kmnist-train-imgs.npz')['arr_0']
test_images = np.load(PATH + 'kmnist-test-imgs.npz')['arr_0']
train_labels = np.load(PATH + 'kmnist-train-labels.npz')['arr_0']
test_labels = np.load(PATH + 'kmnist-test-labels.npz')['arr_0']

char_df = pd.read_csv(PATH + 'kmnist_classmap.csv', encoding='utf-8')

print("KMNIST train shape:", train_images.shape)
print("KMNIST test shape:", test_images.shape)
print("KMNIST train shape:", train_labels.shape)
print("KMNIST test shape:", test_labels.shape)

print("KMNIST character map shape:", char_df.shape)

print('Percent for each category:', np.bincount(train_labels) / len(train_labels) * 100)

# labels = char_df['char']
# f, ax = plt.subplots(1, 1, figsize=(8, 6))
# g = sns.countplot(train_labels)
# g.set_title("Number of labels for each class")
# g.set_xticklabels(labels)


def plot_sample_images_data(images, labels, name):
    plt.figure(name, figsize=(12, 12))
    for i in tqdm.tqdm(range(10)):
        imgs = images[np.where(labels == i)]
        lbls = labels[np.where(labels == i)]
        for j in range(10):
            plt.subplot(10, 10, i * 10 + j + 1)
            plt.xticks([])
            plt.yticks([])
            plt.grid(False)
            plt.imshow(imgs[j], cmap=plt.cm.binary)
            plt.xlabel(lbls[j])


# data preprocessing
def data_preprocessing(images, labels):
    out_y = keras.utils.to_categorical(labels, NUM_CLASSES)
    num_images = images.shape[0]
    x_shaped_array = images.reshape(num_images, IMG_ROWS, IMG_COLS, 1)
    out_x = x_shaped_array / 255
    return out_x, out_y


X, y = data_preprocessing(train_images, train_labels)
X_test, y_test = data_preprocessing(test_images, test_labels)

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)

print("KMNIST train -  rows:", X_train.shape[0], " columns:", X_train.shape[1:4])
print("KMNIST valid -  rows:", X_val.shape[0], " columns:", X_val.shape[1:4])
print("KMNIST test -  rows:", X_test.shape[0], " columns:", X_test.shape[1:4])


def plot_count_per_class(yd):
    ydf = pd.DataFrame(yd)
    f, ax = plt.subplots(1, 1, figsize=(12, 4))
    g = sns.countplot(ydf[0], order=np.arange(0, 10))
    g.set_title("Number of items for each class")
    g.set_xlabel("Category")


def get_count_per_class(yd):
    ydf = pd.DataFrame(yd)
    # Get the count for each label
    label_counts = ydf[0].value_counts()

    # Get total number of samples
    total_samples = len(yd)

    # Count the number of items in each class
    for i in range(len(label_counts)):
        label = label_counts.index[i]
        label_char = char_df[char_df['index'] == label]['char'].item()
        count = label_counts.values[i]
        percent = (count / total_samples) * 100
        print("{}({}):   {} or {}%".format(label, label_char, count, percent))

#
# # the class imbalance for the resulted training set.
# plot_count_per_class(np.argmax(y_train, axis=1))
# get_count_per_class(np.argmax(y_train, axis=1))
#
# # the class distribution of validation set.
# plot_count_per_class(np.argmax(y_val, axis=1))
# get_count_per_class(np.argmax(y_val, axis=1))


def plot_accuracy_and_loss(train_model):
    hist = train_model.history
    acc = hist['acc']
    val_acc = hist['val_acc']
    loss = hist['loss']
    val_loss = hist['val_loss']
    epochs = range(len(acc))
    f, ax = plt.subplots(1, 2, figsize=(14, 6))
    ax[0].plot(epochs, acc, 'g', label='Training accuracy')
    ax[0].plot(epochs, val_acc, 'r', label='Validation accuracy')
    ax[0].set_title('Training and validation accuracy')
    ax[0].legend()
    ax[1].plot(epochs, loss, 'g', label='Training loss')
    ax[1].plot(epochs, val_loss, 'r', label='Validation loss')
    ax[1].set_title('Training and validation loss')
    ax[1].legend()


# Init model
model_class = Model()
model_class.check_model()
history = model_class.run_model(X_train, y_train, X_val, y_val)
model_class.save_model(MODEL_NAME)
plot_accuracy_and_loss(history)


# get the predictions for the test data
predicted_classes = model_class.model.predict_classes(X_val)
# get the indices to be plotted
y_true = np.argmax(y_val, axis=1)
correct = np.nonzero(predicted_classes == y_true)[0]
incorrect = np.nonzero(predicted_classes != y_true)[0]
print("Correct predicted classes:", correct.shape[0])
print("Incorrect predicted classes:", incorrect.shape[0])

target_names = ["Class {} ({}):".format(i, char_df[char_df['index'] == i]['char'].item()) for i in range(NUM_CLASSES)]
print(classification_report(y_true, predicted_classes, target_names=target_names))

# plt.show()

