# Tensorflow 2 version

import matplotlib.pyplot as plt
import numpy as np
import utils.CNN_utils as cu

from tensorflow.keras.models import load_model


def main():
    model_path = "/data/p301081/astronomy/Models"

    filepathTest = "/Trainingdata/20pix_centered_test.csv"
    print("Loading test data from: ", filepathTest)
    testSetX, testSetY = cu.load_data_from_csv(filepathTest, shuffle=False, only_positive=False, magRange=[20, 26])

    # Make sure data is float32 to have enough decimals after normalization
    X_test = testSetX.astype('float32')
    # Normalize pixel values between 0 and 1
    X_train /= 2**8
    X_test /= 2**8

    # If subtract pixel mean is enabled
    subtract_pixel_mean = False
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
