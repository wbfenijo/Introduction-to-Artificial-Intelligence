import random, time
from games import TicTacToe, Gomoku


# # PLAYERS

class Player:
    def choose_move(self, game, state):
        raise NotImplementedError


class AskingPlayer(Player):
    def choose_move(self, game, state):
        # Asks user (human) which move to take. Useful for debug.
        actions = game.actions(state)
        print("Choose one of the following positions: {}".format(actions))
        game.display_state(state, True)
        return int(input('> '))


class RandomPlayer(Player):
    def choose_move(self, game, state):
        # Picks random move from list of possible ones.
        return random.choice(game.actions(state))


class MyPlayer(Player):
    def choose_move(self, game, state):
        # # # YOUR CODE GOES HERE # # #
        # # Examples:
        # my_player = game.player_at_turn(state)
        # opponent = game.other_player(my_player)
        # board = game.board_in_state(state)
        # if board[0][1] == my_player: ...
        # possible_actions = game.actions(state)
        # some_action = possible_actions[0]
        # new_state = game.state_after_move(current_state, some_action)
        # if game.is_terminal(some_state): ...
        # utility = game.utility(some_state, my_player)

        my_player = game.player_at_turn(state)
        opponent = game.other_player(my_player)
        possible_actions = game.actions(state)
        #zaklad, ked viem hned vyhrat, respektive hned musim blokovat
        for move in game.actions(state):
            new_state = game.state_after_move(state, move)
            if game.is_terminal(new_state) and game.utility(new_state, my_player) == 1:
                return move
            
        for move in possible_actions:
            new_state = game.state_after_move(state, move)
            for temp_move in game.actions(new_state):
                temp_state = game.state_after_move(new_state, temp_move)
                if game.is_terminal(temp_state) and game.utility(temp_state, opponent) == 1:
                    return temp_move #vyberiem ten tah, ktorym by super vyhral
        #aby sa prvy tah neopakoval        
        if len(possible_actions) == 9:
            return random.choice(possible_actions)
        
        scores = {}
        for move in possible_actions:
            scores[self.evaluate(game, state, move)] = move 
        return scores[max(scores.keys())] 
 
    def evaluate(self, game, state, move):
        #diag2 ked row = col je rozmer - 1, predpokladame len stvorcove siete
        #diag row = col
        #riadok iba menim col, stlpec naopak
        #row + col == len(temp_state["board"]) - 1:
        score = 0
        number_of_rows = len(state["board"])
        temp_state = game.state_after_move(state, move)
        row, col = game.idx_to_rc(move)
        for i in range(number_of_rows):                                                                             
            if (temp_state["board"][row][i] == " " or temp_state["board"][row][i] == game.player_at_turn(state)) and abs(col - i) < game.k: #na odstranenie zbytocnych tahov,                                                                                                                     
                score += 1                                                                                                                  #kedy su sice v jednom riadku 
                                                                                                                                            #ale daleko od seba                                                                                                             
        for i in range(number_of_rows):
            if (temp_state["board"][i][col] == " " or temp_state["board"][i][col] == game.player_at_turn(state)) and abs(row - i) < game.k:
                score += 1

        if row ==  col: #hlavna diagonala
            for i in range(number_of_rows):
                if (temp_state["board"][i][i] == " " or temp_state["board"][i][i] == game.player_at_turn(state)) and abs(row - i) < game.k:
                    score += 1

        if row + col == len(temp_state["board"]) - 1: #vedlajsia diagonala, ostatne diagonaly, ked je rozmer vacsi ako potrebne k neriesim
            for i in range(number_of_rows):
                if (temp_state["board"][i][number_of_rows - 1 - i] == " " or temp_state["board"][i][number_of_rows - 1 - i] == game.player_at_turn(state)):
                    score += 1
        return score


if __name__ == '__main__':
    # # Print all moves of the game? Useful for debugging, annoying if it`s already working.
    show_moves = True

    # # Play computer against human:
    # # a) with random player
    #TicTacToe().play([RandomPlayer(), AskingPlayer()], show_moves=show_moves)
    # # b) simple TicTacToe with MyPlayer
    # TicTacToe().play([MyPlayer(), AskingPlayer()], show_moves=show_moves)
    # # c) difficult Gomoku with MyPlayer
    # Gomoku().play([MyPlayer(), AskingPlayer()], show_moves=show_moves)

    # # Test MyPlayer
    # # a) play single game of TicTacToe
    # TicTacToe().play([MyPlayer(), RandomPlayer()], show_moves=show_moves)
    # # b) play single game of Gomoku
    # Gomoku().play([MyPlayer(), RandomPlayer()], show_moves=show_moves)
    # # c) play N games
    #TicTacToe().play_n_games([MyPlayer(), RandomPlayer()], n=1000)
    #start = time.time()
    Gomoku().play_n_games([MyPlayer(), RandomPlayer()], n=10)
    #end = time.time()
    #print(end - start , " s.")

