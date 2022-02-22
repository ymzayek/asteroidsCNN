import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from matplotlib.ticker import FormatStrFormatter
from tensorflow.keras import backend as K


def load_data_from_images(image_path, datasplit):
    """Load images from directory

    Parameters:
    image_path (str): Path to images
    datasplit (str): Choose 'train', 'valid', or 'test'

    Returns:
    DataFrame: table with path, label, and dataset description for each image
    numpy array: image matrices
    numpy array: corresponding labels

    """
    image_path = image_path
    data = {
        "Path": [
                 glob.glob(f"{image_path}/{datasplit}/asteroids/" + '*'), 
                 glob.glob(f"{image_path}/{datasplit}/other/" + '*')
                ],
        "Label": [1,0],
        "Set": datasplit
         }
    df = pd.DataFrame(data).explode('Path')
    df = df.sample(frac=1, random_state=35) #shuffle
    x = []
    y = []
    for i, file in enumerate(df['Path']):
        im = Image.open(file)
        im = np.asarray(im)
        x.append(im)
        y.append(df['Label'].iloc[i])
    
    return df, np.array(x, dtype=int), np.array(y, dtype=float)


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


def convert_pixels_to_arcsec_per_h(pixels):
    """
    Converts streak length in pixels to arcsec/h.
    Values are specific to ESA Euclid mission.
    """

    coefficient = 565/3600*10
    arcsecPerH = pixels/coefficient

    return arcsecPerH


def plot_results_heatmap(dataframe2D, binsMag, title, fig_name = 'Plot_2D_histogram_CNN.pdf', savepdf=True):
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
