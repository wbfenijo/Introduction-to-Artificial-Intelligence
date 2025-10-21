import math
import os
import random
import atexit
from time import sleep
import copy

import numpy as np
import matplotlib

# Fix plotting backend if needed (try 'Qt5Agg' or 'Qt4Agg' if 'TkAgg' fails)
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def finish():
    """Prevent plots from closing automatically at the end of execution."""
    plt.show(block=True)


atexit.register(finish)


class OptimizeMax:
    """Abstract class for solving maximization problems using hill climbing."""

    def __init__(self):
        pass

    def hillclimb(self, max_steps=100, plot=True):
        """
        Finds the maximal value of the fitness function using
        the hill climbing algorithm.

        Returns:
            A state (e.g., x) for which fitness(x) is maximal.
        """

        # Task #1: Implement the hill-climb algorithm according to AIMA
        state = self.random_state()
        for step in range(123):
            best = max(self.neighbors(state),key= lambda x: self.fitness(x))
            if self.fitness(best) > self.fitness(state):
                state = best
            if plot:
                self.plot(state, self.fitness(state), title='Hill climb')

            
        return state

    # Abstract methods — to be implemented in subclasses
    def fitness(self, x):
        """Return the value of the fitness function for a given state."""
        raise NotImplementedError("This function must be implemented in a subclass.")

    def neighbors(self, x):
        """Return a list of neighboring states for a given state."""
        raise NotImplementedError("This function must be implemented in a subclass.")

    def random_state(self):
        """Return a random valid state for the problem."""
        raise NotImplementedError("This function must be implemented in a subclass.")

    def plot(self, x, fx):
        """
        Plot the point [x, fx] on a graph.
        If not overridden by a subclass, this function does nothing.
        """
        pass


class MysteryFunction(OptimizeMax):
    """
    Optimization problem: find x that maximizes sin(x)/x + cos(x/10)/3.
    """

    def __init__(self, span=30, delta=0.1):
        self.cfg = None
        self.hist_x = []
        self.hist_y = []
        self.span = span
        self.delta = delta

    def keypress(self, event):
        """Handle keypress events to close or quit plots."""
        if event.key in {'q', 'escape'}:
            # Unclean exit, as sys.exit() won't work in this context
            os._exit(0)
        if event.key in {' ', 'enter'}:
            plt.close()  # Skip blocking figures

    def plot(self, x, y, title, temperature=None):
        """Visualize the function and search progress."""
        if title != self.cfg:
            self.cfg = title
            self.hist_x = []
            self.hist_y = []
            plt.figure(num=title).canvas.mpl_connect('key_press_event', self.keypress)
            plt.axis([-self.span, self.span, -0.5, 2.5])
            plt.ion()

        plt.clf()
        xx = np.linspace(-self.span, self.span, 1000)
        plt.plot(xx, np.sin(xx) / xx + np.cos(xx / 10) / 3, c='k', lw=0.5)

        self.hist_x.append(x)
        self.hist_y.append(y)

        plt.scatter(x, y, s=30, c='r')

        if temperature:
            plt.title(
                f"T = {temperature:.5f}\n"
                f"p(-0.3) = {math.exp(-0.3 / temperature) * 100:.8f} %\n"
                "[Press ESC to quit]",
                loc='left',
            )
        else:
            plt.title("[Press ESC to quit]", loc='left')

        plt.gcf().canvas.flush_events()
        plt.waitforbuttonpress(timeout=0.001)

    def fitness(self, x):
        """Return the fitness value for x."""
        if x == 0:
            return 1
        return np.sin(x) / x + np.cos(x / 10) / 3

    def neighbors(self, x):
        """Return a list of neighboring x values."""
        res = []
        if x > -self.span + 3 * self.delta:
            res += [x - i * self.delta for i in range(1, 4)]
        if x < self.span - 3 * self.delta:
            res += [x + i * self.delta for i in range(1, 4)]
        return res

    def random_state(self):
        """Return a random x within the search span."""
        return random.random() * self.span * 2 - self.span


class EightQueens(OptimizeMax):
    """
    Optimization problem: position 8 queens on an 8×8 chessboard so that
    no two queens threaten each other.
    """
    # indexy 0 - 63
    # Task #2: Implement the fitness, neighbours and random_state functions, so that the hill-climb could find a
    # solution for the 8 queens problem (at least for 1 in 10 runs)
    def fitness(self, x):
        non_attacking = 28  # total pairs
        for i in range(8):
            for j in range(i + 1, 8):
                if x[i] == x[j]:  
                    non_attacking -= 1
                elif abs(x[i] - x[j]) == abs(i - j):  
                    non_attacking -= 1
        return non_attacking

    def neighbors(self, x):
        neighbors = []
        for row in range(8):
            for col in range(8):
                if col != x[row]:
                    new_state = x.copy()
                    new_state[row] = col
                    neighbors.append(new_state)
        return neighbors
        
    def random_state(self):
        return [random.randint(0, 7) for _ in range(8)]


if __name__ == "__main__":
    #Task 1: Mystery function optimization
    # for _ in range(1):
    #     problem = MysteryFunction()
    #     max_x = problem.hillclimb()
    #     print(
    #         "Found maximum of Mystery function with hill climbing at "
    #         f"x={max_x}, f={problem.fitness(max_x)}\n"
    #     )
    #     sleep(2)

    # Task 2: Eight Queens problem
    n_attempts = 10
    for _ in range(n_attempts):
        problem = EightQueens()
        solution = problem.hillclimb(plot=False)
        print(
           "Found a solution (with fitness of {}) with hill climbing "
           "to 8 queens problem:\n{}\n".format(problem.fitness(solution), solution)
        )
