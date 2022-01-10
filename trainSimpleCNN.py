# Tensorflow 2 version

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras import __version__
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import (Activation, BatchNormalization, Dense, Dropout,
                                     Flatten, Input, MaxPooling2D, Conv2D)
from tensorflow.keras.models import Model

import utils.CNN_utils as cu


def main():
    print('Using Keras version:', __version__, 'with backend:', K.backend(), tf.__version__)

    # Choose size of validation set
    validationSetPortion = 0.1

    # Uses training data only between a given range
    magRange = [20, 26]

    # Load CSV training data
    filepathTrain = "../../Trainingdata/20pix_centered_train.csv"
    filepathTest = "../../Trainingdata/20pix_centered_test.csv"

    # Training hyperparameters
    subtract_pixel_mean = False
    epochs = 500
    early_stop_patience = 20
    learning_rate = 0.001
    batch_size = 256
    # dr = 5 / epochs  # Parameter for Learning rate decay

    print("\nLoading training data from: ", filepathTrain)
    trainSetX, trainSetY = cu.load_data_from_csv(filepathTrain, shuffle=True, only_positive=False, multichannel=False, magRange=magRange)
    print("Loading test data from: ", filepathTest)
    testSetX, testSetY = cu.load_data_from_csv(filepathTest, shuffle=False, only_positive=False, multichannel=False, magRange=[20, 26])

    # Make sure data is float32 to have enough decimals after normalization
    X_train = trainSetX.astype('float32')
    X_test = testSetX.astype('float32')

    # Normalize pixel values between 0 and 1
    X_train /= 2**16
    X_test /= 2**16

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_test -= X_train_mean

    Y_train = trainSetY[:, 0:5]
    Y_test = testSetY[:, 0:5]

    Y_trainLabels = trainSetY[:, 0]
    Y_testLabels = testSetY[:, 0]

    # input image dimensions
    img_rows, img_cols = X_train.shape[1:3]

    # Convert to correct Keras format
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    print()
    print('Data loaded: train:', len(X_train), 'test:', len(X_test))
    print('trainSetX:', trainSetX.shape)
    print('X_train:', X_train.shape)
    print('trainSetY:', trainSetY.shape)
    print('Y_train:', Y_train.shape)

    # number of convolutional filters to use
    nb_filters = 64
    # convolution kernel size
    kernel_size = (3, 3)
    # size of pooling area for max pooling
    pool_size = (2, 2)

    dropoutProb = 0.25

    input = Input(shape=input_shape)
    x = BatchNormalization()(input)
    x = Conv2D(nb_filters, kernel_size,
               padding='same',
               input_shape=input_shape,
               use_bias=True)(x)
    #x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(nb_filters, kernel_size,
               padding='same',
               use_bias=True)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Dropout(dropoutProb)(x)


    x = Conv2D(nb_filters*2, kernel_size,
               padding='same',
               use_bias=True)(x)
    #x = BatchNormalization()
    x = Activation("relu")(x)

    x = Conv2D(nb_filters*2, kernel_size,
               padding='same',
               use_bias=True)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = MaxPooling2D(pool_size=pool_size)(x)
    x = Dropout(dropoutProb)(x)


    x = Conv2D(nb_filters*3, kernel_size,
               padding='same',
               use_bias=True)(x)
    #x = BatchNormalization()(x)
    x = Activation("relu")(x)

    x = Conv2D(nb_filters*3, kernel_size,
               padding='same',
               use_bias=True)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    #x = MaxPooling2D(pool_size=pool_size)(x)
    x = Dropout(dropoutProb)(x)

    x = Flatten()(x)
    x = Dense(units=256, use_bias=True)(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Dropout(dropoutProb*2)(x)

    # out1 is the classification unit, out2-out5 are coordinate units
    out1 = Dense(units=1, activation='sigmoid', name='label')(x)
    out2 = Dense(units=1, activation='sigmoid')(x)
    out3 = Dense(units=1, activation='sigmoid')(x)
    out4 = Dense(units=1, activation='sigmoid')(x)
    out5 = Dense(units=1, activation='sigmoid')(x)

    outAll = tf.concat([out1, out2, out3, out4, out5], axis=1, name='outAll')

    optimizer = optimizers.Adam(
        lr=learning_rate,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=None,
        #decay=dr,
        amsgrad=False)

    metrics = [
        tf.keras.metrics.TruePositives(name='tp'),
        tf.keras.metrics.FalsePositives(name='fp'),
        tf.keras.metrics.TrueNegatives(name='tn'),
        tf.keras.metrics.FalseNegatives(name='fn'),
        tf.keras.metrics.BinaryAccuracy(name='accuracy'),
        tf.keras.metrics.Precision(name='precision'),
        tf.keras.metrics.Recall(name='recall'),
        tf.keras.metrics.AUC(name='auc'),
        cu.f1_metric
        ]

    model = Model(inputs=input, outputs=[outAll, out1])

    model.compile(loss=[cu.custom_YOLO_loss, 'binary_crossentropy'],
                  loss_weights=[1, 0],
                  optimizer=optimizer,
                  metrics=["MeanAbsoluteError", metrics])

    print(model.summary())

    # Callback to stop training if val_loss hasn't decreased recently.
    # Patience determines the number of epochs waited before stopping training.
    earlyStopCB = EarlyStopping(
        monitor='val_loss',
        patience=early_stop_patience,
        verbose=1,
        restore_best_weights=True)

    # Callback to save checkpoints of the best model so far.
    checkpointCB = ModelCheckpoint(
        filepath='checkpoint.hdf5',
        verbose=1,
        save_best_only=True,
        monitor='val_loss',
        save_weights_only=False,
        save_freq='epoch')

    # Callback to reduce learning rate if val_loss hasn't improved recently.
    LRCB = ReduceLROnPlateau(
        monitor='val_loss',
        verbose=1,
        factor=0.2,
        patience=5,
        min_lr=0.00001)

    history = model.fit(X_train,
                        [Y_train, Y_trainLabels],
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        validation_split=validationSetPortion,
                        callbacks=[earlyStopCB],  # Write desired callbacks between the brackets
                        shuffle=False)

    # Plot training loss and validation loss history.
    plt.figure(figsize=(5, 3))
    plt.plot(history.epoch, history.history['loss'], label="loss")
    plt.plot(history.epoch, history.history['val_loss'], label="val_loss")
    plt.legend()
    plt.title('loss')

    scoresTrain = model.evaluate(X_train, [Y_train, Y_trainLabels], verbose=2)
    scoresTest = model.evaluate(X_test, [Y_test, Y_testLabels], verbose=2)
    print(scoresTrain, scoresTest)

    predictionsTrain = model.predict(X_train)
    predictionsTest = model.predict(X_test)

    predictionsTrain = predictionsTrain[0]
    predictionsTest = predictionsTest[0]

    print("\nTraining set:")
    cu.analyze_5unit_errors(predictionsTrain, Y_train)
    print("\nTest set:")
    cu.analyze_5unit_errors(predictionsTest, Y_test)

    binsMag = 24  # 60 for single, 24 for heatmap
    binsLength = 20  # 56 for single, 20 for heatmap

    dataframe2Dtrain = cu.create_histogram_2d(predictionsTrain, trainSetY, binsMag, binsLength)
    cu.plot_results_heatmap(dataframe2Dtrain, binsMag, title="Train set completeness")

    dataframe2Dtest = cu.create_histogram_2d(predictionsTest, testSetY, binsMag, binsLength)
    cu.plot_results_heatmap(dataframe2Dtest, binsMag, title="Test set completeness")

    modelName = "test_model.h5"
    print("\nSaving model to", modelName)
    model.save(modelName)

    plt.show()


if __name__ == "__main__":
    main()
