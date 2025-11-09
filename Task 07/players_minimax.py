import sys
import random
from games import TicTacToe, Gomoku
import time

infinity = 0
try:
    infinity = sys.maxint
except:
    infinity = sys.maxsize


class Player:
    def choose_move(self, game, state):
        raise NotImplementedError


class AskingPlayer(Player):
    def choose_move(self, game, state):
        # Asks user (human) which move to take. Useful for debug.
        actions = game.actions(state)
        action = None
        while True:
            print("Choose one of the following positions: {}".format(actions))
            game.display_state(state, True)
            inp = input('> ')
            try:
                action = int(inp)
            except ValueError:
                pass
            if action in actions:
                return action
            print('"{}" is not valid action!'.format(inp))


class RandomPlayer(Player):
    def choose_move(self, game, state):
        # Picks random move from list of possible ones.
        return random.choice(game.actions(state))


class MinimaxPlayer(Player):
    def choose_move(self, game, state):
        # # # Task 1 - implement the minimax algorithm # # #
        # Pseudocode: https://github.com/aimacode/aima-pseudocode/blob/master/md/Minimax-Decision.md

        my_player = game.player_at_turn(state)  # get 'X' or 'O'

        def max_value(state):
            # # # YOUR CODE GOES HERE # # #
            # <dummy code>
            v = -infinity
            best_action = game.actions(state)[0]
            return v, best_action
            # </dummy code>

        def min_value(state):
            # # # YOUR CODE GOES HERE # # #
            # <dummy code>
            v = -infinity
            best_action = game.actions(state)[0]
            return v, best_action
            # </dummy code>

        # # # YOUR CODE GOES HERE # # #
        # # # Examples:
        # my_player = game.player_at_turn(state)
        # opponent = game.other_player(my_player)
        # possible_actions = game.actions(state)
        # some_action = possible_actions[0]
        # new_state = game.state_after_move(current_state, some_action)
        # if game.is_terminal(some_state): ...
        # utility = game.utility(some_state, my_player)
        my_player = game.player_at_turn(state)
        opponent = game.other_player(my_player)
        possible_actions = game.actions(state)
        return game.actions(state)[0]


class AlphaBetaPlayer(Player):
    def choose_move(self, game, state):
        # # # Task 2 - Alpha-Beta pruning# # #
        # Pseudocode: https://github.com/aimacode/aima-pseudocode/blob/master/md/Alpha-Beta-Search.md

        # # # YOUR CODE GOES HERE # # #
        return game.actions(state)[0]


class AlphaBetaEvalPlayer(Player):
    def choose_move(self, game, state):
        # # # Optional task # # #
        # In addition to alpha-beta pruning, return a "possibly good move" instead of a random move

        def evaluate(state):
            return 0  # dummy return

        return game.actions(state)[0]


if __name__ == '__main__':
    show_moves = False

    start_t = time.perf_counter()
    TicTacToe().play_n_games([MinimaxPlayer(), MinimaxPlayer()], n=5)
    end_t = time.perf_counter()
    print(f'\nTotal time MiniMax: {end_t - start_t} seconds.')

    start_t = time.perf_counter()
    TicTacToe().play_n_games([AlphaBetaPlayer(), AlphaBetaPlayer()], n=5)
    end_t = time.perf_counter()
    print(f'\nTotal time AlphaBeta: {end_t - start_t} seconds.')

