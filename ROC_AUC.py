import numpy as np
import matplotlib.pyplot as plt
import sklearn.metrics as metrics


def print_calculate(predicted, actual, name, change_data=True):
    # ignore for deterministic strategy
    if change_data:
        predicted = normalize_predicted(predicted)
        predicted = estimated_positive(predicted)

    # get false and true positive rates
    fpr, tpr, threshold = metrics.roc_curve(actual, predicted)
    roc_auc = metrics.auc(fpr, tpr)

    # ROC and auc visualization
    plt.title('ROC ' + name)
    plt.plot(fpr, tpr, 'b', label='AUC = %0.2f' % roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()


# make sum of probabilities = 1
def normalize_predicted(predicted):
    for i in range(len(predicted)):
        first = predicted[i][0]
        second = predicted[i][1]
        predicted[i][0] = first / (first + second)
        predicted[i][1] = second / (first + second)

    return predicted


# choose estimated positive
def estimated_positive(predicted):
    new_predicted = np.empty(len(predicted), float)
    for i in range(len(predicted)):
        if predicted[i][0] >= predicted[i][1]:
            new_predicted[i] = predicted[i][0]
        else:
            new_predicted[i] = predicted[i][1]

    return new_predicted
