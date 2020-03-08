
import os
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt; plt.rcdefaults()

from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import random

random.seed(1618)
np.random.seed(1618)
tf.random.set_seed(1618)

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

#ALGORITHM = "guesser"
# ALGORITHM = "tf_net"
ALGORITHM = "tf_conv"

DATASET = "mnist_d"
# DATASET = "mnist_f"
# DATASET = "cifar_10"
# DATASET = "cifar_100_f"
# DATASET = "cifar_100_c"

if DATASET == "mnist_d":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "mnist_f":
    NUM_CLASSES = 10
    IH = 28
    IW = 28
    IZ = 1
    IS = 784
elif DATASET == "cifar_10":
    NUM_CLASSES = 10
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
    # TODO: Add this case.
elif DATASET == "cifar_100_f":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072
elif DATASET == "cifar_100_c":
    NUM_CLASSES = 100
    IH = 32
    IW = 32
    IZ = 3
    IS = 3072


#=========================<Classifier Functions>================================

def guesserClassifier(xTest):
    ans = []
    for entry in xTest:
        pred = [0] * NUM_CLASSES
        pred[random.randint(0, 9)] = 1
        ans.append(pred)
    return np.array(ans)


def buildTFNeuralNet(x, y, eps = 20):
    #TODO: Implement a standard ANN here.
    model = keras.Sequential([keras.layers.Flatten(), keras.layers.Dense(512, activation=tf.nn.relu), keras.layers.Dense(1024, activation=tf.nn.relu), keras.layers.Dense(256, activation=tf.nn.relu), tf.keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax)])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    invert_categorical_y = np.argmax(y, axis=1)
    model.fit(x, invert_categorical_y, epochs=eps, verbose=0)
    return model


def buildTFConvNet(x, y, eps = 10, dropout = True, dropRate = 0.2):
    #TODO: Implement a CNN here. dropout option is required.
    model = keras.Sequential()

    inShape = (IH, IW, IZ)
    opt = tf.optimizers.Adam()
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation = "elu", input_shape=inShape))
    model.add(keras.layers.Conv2D(32, kernel_size=(3, 3), activation="elu"))
    model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(keras.layers.Dropout(dropRate, noise_shape=None, seed=None))

    # model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="elu", input_shape=inShape))
    # model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation="elu"))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Dropout(dropRate, noise_shape=None, seed=None))
    #
    # model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="elu", input_shape=inShape))
    # model.add(keras.layers.Conv2D(128, kernel_size=(3, 3), activation="elu"))
    # model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
    # model.add(keras.layers.Dropout(dropRate, noise_shape=None, seed=None))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation="relu"))
    if dropout:
        model.add(keras.layers.Dropout(dropRate, noise_shape=None, seed=None))
    model.add(keras.layers.Dense(NUM_CLASSES, activation=tf.nn.softmax))

    model.compile(optimizer = opt, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    invert_categorical_y = np.argmax(y, axis=1)
    model.fit(x, invert_categorical_y, eps)
    return model

#=========================<Pipeline Functions>==================================

def getRawData():
    if DATASET == "mnist_d":
        mnist = tf.keras.datasets.mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "mnist_f":
        mnist = tf.keras.datasets.fashion_mnist
        (xTrain, yTrain), (xTest, yTest) = mnist.load_data()
    elif DATASET == "cifar_10":
        # TODO: Add this case.
        cifar10 = tf.keras.datasets.cifar10
        (xTrain, yTrain), (xTest, yTest) = cifar10.load_data()
    elif DATASET == "cifar_100_f":
        # TODO: Add this case.
        cifar100 = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar100.load_data(label_mode='fine')
    elif DATASET == "cifar_100_c":
        cifar100 = tf.keras.datasets.cifar100
        (xTrain, yTrain), (xTest, yTest) = cifar100.load_data(label_mode='coarse')
        # TODO: Add this case.
    else:
        raise ValueError("Dataset not recognized.")
    print("Dataset: %s" % DATASET)
    print("Shape of xTrain dataset: %s." % str(xTrain.shape))
    print("Shape of yTrain dataset: %s." % str(yTrain.shape))
    print("Shape of xTest dataset: %s." % str(xTest.shape))
    print("Shape of yTest dataset: %s." % str(yTest.shape))
    return ((xTrain, yTrain), (xTest, yTest))



def preprocessData(raw):
    ((xTrain, yTrain), (xTest, yTest)) = raw
    if ALGORITHM != "tf_conv":
        xTrainP = xTrain.reshape((xTrain.shape[0], IS))
        xTestP = xTest.reshape((xTest.shape[0], IS))
    else:
        xTrainP = xTrain.reshape((xTrain.shape[0], IH, IW, IZ))
        xTestP = xTest.reshape((xTest.shape[0], IH, IW, IZ))
    yTrainP = to_categorical(yTrain, NUM_CLASSES)
    yTestP = to_categorical(yTest, NUM_CLASSES)
    print("New shape of xTrain dataset: %s." % str(xTrainP.shape))
    print("New shape of xTest dataset: %s." % str(xTestP.shape))
    print("New shape of yTrain dataset: %s." % str(yTrainP.shape))
    print("New shape of yTest dataset: %s." % str(yTestP.shape))
    return ((xTrainP, yTrainP), (xTestP, yTestP))



def trainModel(data):
    xTrain, yTrain = data
    if ALGORITHM == "guesser":
        return None   # Guesser has no model, as it is just guessing.
    elif ALGORITHM == "tf_net":
        print("Building and training TF_NN.")
        return buildTFNeuralNet(xTrain, yTrain)
    elif ALGORITHM == "tf_conv":
        print("Building and training TF_CNN.")
        return buildTFConvNet(xTrain, yTrain)
    else:
        raise ValueError("Algorithm not recognized.")



def runModel(data, model):
    if ALGORITHM == "guesser":
        return guesserClassifier(data)
    elif ALGORITHM == "tf_net":
        print("Testing TF_NN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    elif ALGORITHM == "tf_conv":
        print("Testing TF_CNN.")
        preds = model.predict(data)
        for i in range(preds.shape[0]):
            oneHot = [0] * NUM_CLASSES
            oneHot[np.argmax(preds[i])] = 1
            preds[i] = oneHot
        return preds
    else:
        raise ValueError("Algorithm not recognized.")



def evalResults(data, preds):
    xTest, yTest = data
    acc = 0
    for i in range(preds.shape[0]):
        if np.array_equal(preds[i], yTest[i]):   acc = acc + 1
    accuracy = acc / preds.shape[0]
    print("Classifier algorithm: %s" % ALGORITHM)
    print("Classifier accuracy: %f%%" % (accuracy * 100))
    print()


def plot_ANN ():
    objects = ['mnist_d', 'mnist_f', 'cifar_10', 'cifar_100_f', 'cifar_100_c']
    y_pos = np.arange(len(objects))
    performance = [97.56, 87.44, 10, 1, 1]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('accuracy (percentage)')
    plt.title('ANN')
    plt.show()

def plot_CNN ():
    objects = ['mnist_d', 'mnist_f', 'cifar_10', 'cifar_100_f', 'cifar_100_c']
    y_pos = np.arange(len(objects))
    performance = [98.31, 85.22, 10, 1, 21.46]

    plt.bar(y_pos, performance, align='center', alpha=0.5)
    plt.xticks(y_pos, objects)
    plt.ylabel('accuracy (percentage)')
    plt.title('CNN')
    plt.show()

#=========================<Main>================================================

def main():
    raw = getRawData()
    data = preprocessData(raw)
    model = trainModel(data[0])
    preds = runModel(data[1][0], model)
    evalResults(data[1], preds)

    plot_ANN();
    plot_CNN();




if __name__ == '__main__':
    main()
