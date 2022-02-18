# Tensorflow 2 version
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
import cv2
import os
from sklearn.metrics import confusion_matrix
import glob
import argparse

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
    
    #load training data
    trainSet, X_train, Y_train = cu.load_data_from_images(image_path, 'train', mode, size = (20,20))
    validSet, X_valid, Y_valid = cu.load_data_from_images(image_path, 'valid', mode, size = (20,20))
    testSet, X_test, Y_test = cu.load_data_from_images(image_path, 'test', mode, size = (20,20))

    # Training hyperparameters
    subtract_pixel_mean = False
    epochs = 100 #500
    early_stop_patience = 20
    learning_rate = 0.001
    batch_size = 256
    # dr = 5 / epochs  # Parameter for Learning rate decay

    # Make sure data is float32 to have enough decimals after normalization
    X_train = X_train.astype('float32')
    X_valid = X_valid.astype('float32')
    X_test = X_test.astype('float32')

    # Normalize pixel values between 0 and 1
    X_train /= 2**8
    X_valid /= 2**8
    X_test /= 2**8

    # If subtract pixel mean is enabled
    if subtract_pixel_mean:
        X_train_mean = np.mean(X_train, axis=0)
        X_train -= X_train_mean
        X_valid -= X_train_mean
        X_test -= X_train_mean

    # input image dimensions
    img_rows, img_cols = X_train.shape[1:3]

    # Convert to correct Keras format
    X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
    X_valid = X_valid.reshape(X_valid.shape[0], img_rows, img_cols, 1)
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

    print()
    print('Data loaded: train:', len(X_train), 'valid:', len(X_valid), 'test:', len(X_test))
    print('X_train:', X_train.shape)
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

    # out1 is the classification unit
    out1 = Dense(units=1, activation='sigmoid', name='label')(x)

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

    model = Model(inputs=input, outputs=out1)

    model.compile(loss='binary_crossentropy',
                  #loss_weights=[1, 1],
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
        filepath='/data/p301081/astronomy/Models/checkpoint.hdf5',
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

    # training
    history = model.fit(X_train,
                        Y_train,
                        epochs=epochs,
                        batch_size=batch_size,
                        verbose=2,
                        validation_data=(X_valid, Y_valid),
                        callbacks=[checkpointCB],  # Write desired callbacks between the brackets
                        shuffle=False)

    # Plot training loss and validation loss history.
    plt.figure(figsize=(5, 3))
    plt.plot(history.epoch, history.history['loss'], label="loss")
    plt.plot(history.epoch, history.history['val_loss'], label="val loss")
    plt.xlabel('Epoch')
    plt.ylabel('Binary crossentropy')
    plt.legend()
    plt.title(f'Loss {mode}')
    plt.tight_layout()
    if save:
        plt.savefig(f"{plots_out}/loss_{mode}.png")

    scoresTrain = model.evaluate(X_train, Y_train, verbose=2)
    scoresValid = model.evaluate(X_valid, Y_valid, verbose=2)
    scoresTest = model.evaluate(X_test, Y_test, verbose=2)
    #print(scoresTrain, scoresTest)

    predictionsTrain = model.predict(X_train)
    predictionsValid = model.predict(X_valid)
    predictionsTest = model.predict(X_test)

    predictionsTrain = [round(pred[0]) for pred in predictionsTrain]
    predictionsValid = [round(pred[0]) for pred in predictionsValid]
    predictionsTest = [round(pred[0]) for pred in predictionsTest]

    print("\nTraining set:")
    train_set_metrics = cu.analyze_5unit_errors(predictionsTrain, Y_train)
    print("\nValidation set:")
    train_set_metrics = cu.analyze_5unit_errors(predictionsValid, Y_valid)
    print("\nTest set:")
    test_set_metrics = cu.analyze_5unit_errors(predictionsTest, Y_test)

    if save:
        modelName = f"cnn_asteroids_{mode}.h5"
        print("\nSaving model to", modelName)
        model.save(f"/data/p301081/astronomy/Models/{modelName}")
        
    plt.figure(figsize=(10,8))
    cf_matrix = confusion_matrix(Y_test, predictionsTest)
    cf_matrix
    group_names = ['True Neg','False Pos','False Neg','True Pos']

    group_counts = ["{0:0.0f}".format(value) for value in
                    cf_matrix.flatten()]

    group_percentages = ["{0:.2%}".format(value) for value in
                         cf_matrix.flatten() / np.sum(cf_matrix)]

    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in
              zip(group_names,group_counts,group_percentages)]

    labels = np.asarray(labels).reshape(2,2)
    ax = sns.heatmap(cf_matrix, annot=labels, fmt='', cmap='Blues')

    ax.set_title(f'Confusion Matrix\n {mode}');
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values\n');

    ## Ticket labels - List must be in alphabetical order
    ax.xaxis.set_ticklabels(['False','True'])
    ax.yaxis.set_ticklabels(['False','True'])
    ax.set_ylim([0,2])

    ax.text(0, -0.2, test_set_metrics, wrap=True, horizontalalignment='left', fontsize=12, transform = ax.transAxes)

    plt.tight_layout()

    if save:
        plt.savefig(f"{plots_out}/confusionMatrix_{mode}.png")
    
    rows = int(group_counts[1]) + int(group_counts[2])
    fig = plt.figure(figsize=(10,14))
    plt.axis('off')
    fig.suptitle(f'Data: {mode}', y=.98, fontsize = 16)
    plt.tight_layout() 
    s = 1
    for idx, (first, second) in enumerate(zip(predictionsTest, Y_test)):
        if first == 1 and first != second:
            fig.add_subplot(rows, 1, s)
            plt.imshow(X_test[idx,...,0], cmap='gray')
            plt_txt = f"FP index: {idx}\nLabel: {int(Y_test[idx])}\nImage path: {testSet['Path'].iloc[idx]}"
            plt.text(0, -10, plt_txt, wrap=True, horizontalalignment='left', fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            s+=1
        if first == 0 and first != second:
            fig.add_subplot(rows, 1, s)
            plt.imshow(X_test[idx,...,0], cmap='gray')
            plt_txt = f"FN index: {idx}\nLabel: {int(Y_test[idx])}\nImage path: {testSet['Path'].iloc[idx]}"
            plt.text(0, -10, plt_txt, wrap=True, horizontalalignment='left', fontsize=14)
            plt.axis('off')
            plt.tight_layout()
            s+=1
    plt.tight_layout()       
    if save:
        plt.savefig(f"{plots_out}/fp_fn_{mode}.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--image_path', metavar='path', required=True,
                        help='the path to training image data')
    parser.add_argument('--mode', metavar='param', required=True,
                        help='choose image mode for training: Crop, Resize, or Original')
    parser.add_argument('--save', metavar='bool', required=True,
                        help='Whether or not you want to save performance output plots')
    parser.add_argument('--plots_out', metavar='path', required=False,
                        help='path where to save plots')
    args = parser.parse_args()
    
    image_path = args.image_path
    mode = args.mode
    save = args.save
    plots_out = args.plots_out
    
    main()
