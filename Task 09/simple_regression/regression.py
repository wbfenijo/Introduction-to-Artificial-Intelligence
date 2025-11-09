from generate_data import Data
import numpy as np


def linear_regression(data):
    # Use linear regression to find a line that fits the data the best
    # data is a list of tuples (x, y)
    # Return parameters of the line a*x + b as a tuple (a, b)
    # # # YOUR CODE GOES HERE # # #

    data = np.array(data)
    n = data.shape[0]
    x, y = data[:, 0], data[:, 1]
    s_x, s_y = np.sum(x), np.sum(y)

    a = (n * x @ y - s_x * s_y) / (n * np.sum(x ** 2) - s_x ** 2)
    b = (s_y - a * s_x) / n

    return a, b


def quadratic_regression(data):
    # Use quadratic regression to find a parabola that fits the data the best
    # data is a list of tuples (x, y)
    # Return parameters of the parabola a*x**2 + b*x + c as a tuple (a, b, c)
    # # # YOUR CODE GOES HERE # # #

    data = np.array(data)
    n = data.shape[0]
    x, y = data[:, 0], data[:, 1]

    xx = np.sum(x ** 2) - (np.sum(x) ** 2) / n
    xy = x @ y - (np.sum(x) * np.sum(y)) / n
    xx2 = np.sum(x ** 3) - (np.sum(x) * np.sum(x ** 2)) / n
    x2y = np.sum(x ** 2 * y) - (np.sum(x ** 2) * np.sum(y)) / n
    x2x2 = np.sum(x ** 4) - (np.sum(x ** 2) ** 2) / n

    a = (x2y * xx - xy * xx2) / (xx * x2x2 - xx2 ** 2)
    b = (xy * x2x2 - x2y * xx2) / (xx * x2x2 - xx2 ** 2)
    c = (np.sum(y) - b * np.sum(x) - a * np.sum(x ** 2)) / n

    return a, b, c


if __name__ == "__main__":
    # generate linear data, calculate both linear and quadratic fit, plot along with regression error and test data
    lin = Data('linear')
    line_a, line_b = linear_regression(lin.points)
    parabola_a, parabola_b, parabola_c = quadratic_regression(lin.points)
    lin.plot_result(line_a, line_b, parabola_a, parabola_b, parabola_c)

    # generate quadratic data, calculate both linear and quadratic fit, plot along with regression error and test data
    quad = Data('quadratic')
    line_a, line_b = linear_regression(quad.points)
    parabola_a, parabola_b, parabola_c = quadratic_regression(quad.points)
    quad.plot_result(line_a, line_b, parabola_a, parabola_b, parabola_c)
