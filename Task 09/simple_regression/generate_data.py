import numpy as np
import matplotlib.pyplot as plt


class Data:
    def __init__(self, task='linear'):
        self.task = task
        if task == 'linear':
            self.a = np.random.rand() * 10 - 5
            self.b = np.random.rand() * 10 - 5
            x = np.linspace(-2, 2, 100)
            y = self.a * x + self.b
            y = y + np.random.randn(y.shape[0])
            self.points = list(zip(list(x), list(y)))
            x = np.linspace(3, 4.5, 75)
            y = self.a * x + self.b
            y = y + np.random.randn(y.shape[0])
            self.test_points = list(zip(list(x), list(y)))*10
        elif task == 'quadratic':
            self.a = np.random.rand() * 2 - 1
            self.b = np.random.rand() * 10 - 5
            self.c = np.random.rand() * 10 - 5
            x = np.linspace(-2, 2, 100)
            y = self.a * x ** 2 + self.b * x + self.c
            y = y + np.random.randn(y.shape[0])
            self.points = list(zip(list(x), list(y)))
            x = np.linspace(2.5, 3, 25)
            y = self.a * x ** 2 + self.b * x + self.c
            y = y + np.random.randn(y.shape[0])
            self.test_points = list(zip(list(x), list(y)))
        else:
            raise ValueError('Non-supported type of task')

    def plot_result(self, lin_estimate_a, lin_estimate_b, quad_estimate_a, quad_estimate_b, quad_estimate_c):
        x = np.array(self.points)[:, 0]
        lin_est = lin_estimate_a * x + lin_estimate_b
        quad_est = quad_estimate_a * x ** 2 + quad_estimate_b * x + quad_estimate_c
        true_data = np.array(self.points)[:, 1]

        lin_err = np.sum((lin_est - true_data)**2)/x.shape[0]
        quad_err = np.sum((quad_est - true_data)**2)/x.shape[0]

        x = np.array(self.test_points)[:, 0]
        lin_est = lin_estimate_a * x + lin_estimate_b
        quad_est = quad_estimate_a * x ** 2 + quad_estimate_b * x + quad_estimate_c
        true_data = np.array(self.test_points)[:, 1]

        lin_test_err = np.sum((lin_est - true_data)**2)/x.shape[0]
        quad_test_err = np.sum((quad_est - true_data)**2)/x.shape[0]

        fig, ax = plt.subplots()
        x = np.linspace(-2.5, 5, 4000)
        y1 = lin_estimate_a * x + lin_estimate_b
        y2 = quad_estimate_a * x ** 2 + quad_estimate_b * x + quad_estimate_c
        ax.scatter(np.array(self.points)[:, 0], np.array(self.points)[:, 1], s=1, c='b', marker='o',
                   label='fitted data')
        ax.scatter(np.array(self.test_points)[:, 0], np.array(self.test_points)[:, 1], s=1, c='k', marker='o',
                   label='test data')
        ax.plot(x, y1, 'r', label='linear regression')
        ax.plot(x, y2, 'g', label='quadratic regression')
        plt.title('Linear reg. mean error on fitted data: {:.3f} and on test data: {:.3f}\n'
                  'Quadratic reg. mean error on fitted data: {:.3f} and on test data: {:.3f}'
                  .format(lin_err, lin_test_err, quad_err, quad_test_err))
        plt.legend()
        plt.show()
