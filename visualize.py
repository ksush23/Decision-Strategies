import matplotlib.pyplot as plt
import numpy as np


def visualize_predictive_features(first, second, third, draw=True):
    x1 = first['RH'].to_numpy()
    y1 = first['RA'].to_numpy()
    x2 = second['RH'].to_numpy()
    y2 = second['RA'].to_numpy()
    x3 = third['RH'].to_numpy()
    y3 = third['RA'].to_numpy()

    color = 'yellow'
    if not draw:
        color = 'red'
    plt.scatter(x1, y1, c='red')
    plt.scatter(x2, y2, c='blue')
    plt.scatter(x3, y3, c=color)

    line_x1, line_y1 = [7, 7], [0, 20]
    line_x2, line_y2 = [7, 20], [7, 7]
    plt.plot(line_x1, line_y1, line_x2, line_y2, marker='o')
    plt.show()


