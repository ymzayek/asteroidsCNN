# Tensorflow 2 version

import matplotlib.pyplot as plt
import numpy as np
import utils.CNN_utils as cu

from tensorflow.keras.models import load_model


def main():
    #modelName = "Models/model20PixAllMags_5unit_87.h5"
    #modelName = "Models/model20PixAllMagsSW_5unit_82.h5"
    #modelName = "Models/model20PixAllMagsSW_5unit_99prec.h5"
    modelName = "../Models/model20Pix20-23_5unit.h5"
    #modelName = "model20PixAllMags10-1_5unit_97.h5"

    # Selections for different CNN models
    resnet = False
    subtract_pixel_mean = False

    # Selections for result heatmap
    binsMag = 24
    binsLength = 20

    # Load CSV training data
    filepathTrain = "/Trainingdata/20pix_centered_train.csv"
    filepathTest = "/Trainingdata/20pix_centered_test.csv"
    #filepathTrain = "Trainingdata/20pix_10to1_train.csv"
    #filepathTest = "Trainingdata/20pix_10to1_test.csv"
    print("\nLoading training data from: ", filepathTrain)
    trainSetX, trainSetY = cu.load_data_from_csv(filepathTrain, shuffle=False, only_positive=False, magRange=[20, 26])
    print("Loading test data from: ", filepathTest)
    testSetX, testSetY = cu.load_data_from_csv(filepathTest, shuffle=False, only_positive=False, magRange=[20, 26])

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

    print()
    print('Data loaded: train:', len(X_train), 'test:', len(X_test))
    print('trainSetX:', trainSetX.shape)
    print('X_train:', X_train.shape)
    print('trainSetY:', trainSetY.shape)
    print('Y_train:', Y_train.shape)

    if resnet:
        model = load_model(modelName, custom_objects={'custom_YOLO_loss': cu.custom_YOLO_loss,
                                                      'f1_metric': cu.f1_metric})
    else:
        model = load_model(modelName, custom_objects={'custom_loss': cu.custom_YOLO_loss})

    print(model.summary())

    # Evaluate
    scoresTrain = model.evaluate(X_train, [Y_train, Y_trainLabels], verbose=2)
    scoresTest = model.evaluate(X_test, [Y_test, Y_testLabels], verbose=2)
    print(scoresTrain)
    print(scoresTest)

    # Make predictions
    predictionsTrain = model.predict(X_train)
    predictionsTest = model.predict(X_test)

    # ResNet predictions array is different shape that simple CNN
    if resnet:
        predictionsTrain = predictionsTrain[0]
        predictionsTest = predictionsTest[0]

    print("\nTraining set:")
    cu.analyze_5unit_errors(predictionsTrain, Y_train)
    print("\nTest set:")
    cu.analyze_5unit_errors(predictionsTest, Y_test)

    dataframe2Dtrain = cu.create_histogram_2d(predictionsTrain, trainSetY, binsMag, binsLength)
    cu.plot_results_heatmap(dataframe2Dtrain, binsMag, fig_name = 'Plot_2D_histogram_CNN_train.pdf', title="Deep learning training set completeness") #ym

    dataframe2Dtest = cu.create_histogram_2d(predictionsTest, testSetY, binsMag, binsLength)
    cu.plot_results_heatmap(dataframe2Dtest, binsMag, fig_name = 'Plot_2D_histogram_CNN_test.pdf', title="Deep learning test set completeness") #ym

    plt.show()


if __name__ == "__main__":
    main()
