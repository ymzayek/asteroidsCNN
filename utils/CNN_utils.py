import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter
from tensorflow.keras import backend as K


def load_data_from_csv(filename, shuffle=False, only_positive=False, multichannel=False, magRange=[20, 26]):
    """
    ['Label', 'Angle', 'Magnitude', 'Length', 'Pixel data']
    If only_positive = True, loads only positive training examples.
    """

    with open(filename) as file:
        csv_reader = csv.reader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        first_line = True
        temp_images = []
        temp_labels = []
        width = 0
        skipped = 0
        for row in csv_reader:
            if first_line:
                first_line = False
                imageIndexStart = len(row) - 1
            elif ((only_positive is True and int(row[0]) == 1) or only_positive is False
                  ) and (magRange[0] <= float(row[6]) <= magRange[1] or float(row[6]) == 0):
                temp_labels.append(row[0:8])
                image_data = np.asarray(row[imageIndexStart:], dtype="uint32")
                if width == 0:
                    if multichannel is False:
                        width = int(np.sqrt(len(image_data)))
                    else:
                        width = int(np.sqrt(len(image_data)/4))
                if multichannel is False:
                    image_data_array = np.array_split(image_data, width)
                else:
                    image_data_array = image_data.reshape(width, width, 4)
                    # Converts image from channel first to channel last shape
                    #image_data_array = np.moveaxis(image_data_array, 0, 2)
                temp_images.append(image_data_array)
            else:
                #print("passing", only_positive, only_positive is False, ((only_positive is True and int(row[0]) == 1) or only_positive is False))
                skipped += 1
                pass
        labels = np.array(temp_labels).astype('float')
        images = np.array(temp_images).astype('uint32')
        if only_positive:
            print("  Only_positive mode:", only_positive, "  Magnitude range:", magRange, "  Skipped examples:", skipped)
    if shuffle:
        images, labels = unison_shuffled_copies(images, labels)

    return images, labels


def crop_center(im, new_w, new_h):
    """
    Crop center of image
    """
    width, height = im.size   # Get dimensions

    left = (width - new_w)/2
    top = (height - new_h)/2
    right = (width + new_w)/2
    bottom = (height + new_h)/2

    # Crop the center of the image
    im = im.crop((left, top, right, bottom))
    
    return im
    

def unison_shuffled_copies(a, b):
    """
    Shuffles two arrays the same way.
    """

    assert len(a) == len(b)
    p = np.random.permutation(len(a))

    return a[p], b[p]


def custom_YOLO_loss(y_true, y_pred):
    """
    Loss function for combined label and streak end coordinate prediction
    """

    # Weights for coordinates and labels
    params = [5.0, 0.5]

    lambda_noobj = K.abs(y_true[..., 0] - y_pred[..., 0]) * params[1]

    lp = K.sum(K.square(y_true[..., 0] - y_pred[..., 0]) * lambda_noobj)

    return lp


def f1_metric(y_true, y_pred):
    """
    F1 metric that combines precision and recall
    """

    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1_val


def analyze_5unit_errors(predictionsLabel, Y_testLabel): #,image_size=20
    """
    Compares predictions of labels and coordinates to ground truth
    """
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(predictionsLabel)):
        trueLabel = Y_testLabel[i]
        predictedLabel = round(predictionsLabel[i])
        if trueLabel == 1 and predictedLabel == 1:
            tp += 1
        elif trueLabel == 0 and predictedLabel == 0:
            tn += 1
        elif trueLabel == 0 and predictedLabel == 1:
            fp += 1
        elif trueLabel == 1 and predictedLabel == 0:
            fn += 1
        else:
            pass
    accuracy = (tp + tn) / len(predictionsLabel) * 100
    precision = tp / (tp + fp) * 100
    recall = tp / (tp + fn) * 100

    print("\nClassification accuracy, precision, recall:", f"{accuracy:.2f}", f"{precision:.2f}", f"{recall:.2f}")
    print("TP, TN, FP, FN:", tp, tn, fp, fn)

    return f"\nClassification accuracy, precision, recall: {accuracy:.2f} {precision:.2f} {recall:.2f}\n TP, TN, FP, FN: {tp} {tn} {fp} {fn}"


def create_histogram_2d(predictions, Y, binsMag, binsLength):
    """
    Prepares results for plotting with plot_results_heatmap.
    """

    minMag = 20
    maxMag = 26

    gtMags = Y[:, 6]
    gtLengths = Y[:, 7]
    predLabels = predictions[:, 0]

    gtLengths = convert_pixels_to_arcsec_per_h(gtLengths)

    minLength = np.min(gtLengths)
    maxLength = np.max(gtLengths)

    range2d = [[minMag, maxMag], [minLength, maxLength]]
    bins = [binsMag, binsLength]

    tptnLengths = []
    tptnMags = []
    for i in range(len(Y)):
        if int(np.round(predLabels[i])) == int(Y[i][0]):
            tptnLengths.append(Y[i][7])
            tptnMags.append(Y[i][6])

    tptnLengths = np.asarray(tptnLengths)
    tptnMags = np.asarray(tptnMags)
    tptnLengths = convert_pixels_to_arcsec_per_h(tptnLengths)

    histGT, xEdgesGT, yEdgesGT = np.histogram2d(gtMags, gtLengths, bins=bins, range=range2d, normed=None, weights=None, density=None)
    histTPTN, xEdgesTPTN, yEdgesTPTN = np.histogram2d(tptnMags, tptnLengths, bins=bins, range=range2d, normed=None, weights=None, density=None)

    print("\n2D histogram counts")
    print("Mean count", np.mean(histGT))
    print("Min count", np.min(histGT))
    print("Length range:", minLength, maxLength)

    histResult = histTPTN / histGT * 100

    dataframe2D = pd.DataFrame(histResult, columns=yEdgesGT[:-1], index=xEdgesGT[:-1])

    return dataframe2D


def convert_pixels_to_arcsec_per_h(pixels):
    """
    Converts streak length in pixels to arcsec/h.
    Values are specific to ESA Euclid mission.
    """

    coefficient = 565/3600*10
    arcsecPerH = pixels/coefficient

    return arcsecPerH


def plot_results_heatmap(dataframe2D, binsMag, title, fig_name = 'Plot_2D_histogram_CNN.pdf', savepdf=True): #ym
    """
    Plots the recall (completeness of predictions as a heatmap).
    Saves the heatmap to a pdf.
    """
    cmap = "PRGn"
    xticklabels = np.append(dataframe2D.columns.values.round(0).astype(int), 80)
    yticklabels = np.append(dataframe2D.index.values.round(2), 26.0)
    fig4, ax4 = plt.subplots(figsize=(10, 6), dpi=100)
    sns.heatmap(dataframe2D,
                ax=ax4,
                cmap=cmap,
                annot=True,
                fmt='3.0f',
                cbar_kws={'label': 'Completeness [%]'},
                xticklabels=xticklabels,
                yticklabels=yticklabels,
                vmin=0,
                vmax=100,
                linewidths=0.5)
    ax4.set_ylim(0, binsMag)
    ax4.set_title(title)
    ax4.set_xlabel(r'Sky motion [$\rm arcsec\,h^{-1}$]')
    ax4.set_ylabel('Magnitude')
    ax4.yaxis.set_major_formatter(FormatStrFormatter('%5.2f'))
    yticks = ax4.get_yticks() - 0.5
    ax4.set_yticks(yticks)
    xticks = ax4.get_xticks() - 0.5
    ax4.set_xticks(xticks)
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)  # , horizontalalignment='right')
    ax4.set_yticklabels(ax4.get_yticklabels(), rotation=0)
    #PRGn
    #PuBuGn
    fig4.tight_layout()
    if savepdf:
        fig4.savefig(fig_name, dpi=300, format="pdf")
    return
