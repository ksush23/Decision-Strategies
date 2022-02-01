import numpy as np


# compare actual with predicted
def get_accuracy_random(actual, predicted):
    correct = 0
    for i in range(len(actual)):
        if actual[i] == 1:
            if np.max(predicted[i]) == predicted[i][0]:
                correct += 1
        elif actual[i] == 0:
            if np.max(predicted[i]) == predicted[i][1]:
                correct += 1
        else:
            if np.max(predicted[i]) == predicted[i][2]:
                correct += 1

    return correct / len(actual)


# who is for 1 0 -1 (always home, always draw, always lose strategies)
def get_accuracy_deterministic(actual, who=1):
    predicted = who
    correct = 0
    for item in actual:
        if item == predicted:
            correct += 1

    return correct / len(actual)
