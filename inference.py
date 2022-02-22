# Tensorflow 2 version

import matplotlib.pyplot as plt
import numpy as np
import utils.CNN_utils as cu

from tensorflow.keras.models import load_model


def main():
    
    image_path = "/data/pg-ds_cit/Projects/Astronomy/AstronomyProject/Images"
    print("Loading test data from: ", image_path)
    testSet, X_test, Y_test = cu.load_data_from_images(image_path, 'test')

    # Make sure data is float32 to have enough decimals after normalization
    X_test = testSetX.astype('float32')
    # Normalize pixel values between 0 and 1
    X_test /= 2**8

    # input image dimensions
    img_rows, img_cols = X_test.shape[1:3]

    # Convert to correct Keras format
    X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

    print()
    print('Data loaded: test:', len(X_test))
    print('X_test:', X_test.shape)

    model = load_model(model_path)

    print(model.summary())

    # Make predictions
    predictionsTest = model.predict(X_test)
    predictionsTest = [round(pred[0]) for pred in predictionsTest]
    
    # Evaluate
    if Y_test:
        scoresTest = model.evaluate(X_test, Y_test, verbose=2)
        print(scoresTest)
        test_set_metrics = cu.analyze_5unit_errors(predictionsTest, Y_test)


if __name__ == "__main__":
    main()
