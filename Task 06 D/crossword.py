from collections import defaultdict
import time

class CrossWord:
    # Directions represent possible word placements on the grid:
    # Each entry maps direction name → (row offset, column offset)
    directions = {'down': (1, 0), 'right': (0, 1)}

    def __init__(self, grid):
        """
        Initialize a crossword puzzle from a given grid.

        Args:
            grid (list[list[str]]): 2D list representing crossword layout.
                                   '#' marks blocked cells,
                                   ' ' marks empty spaces.
        """
        self.grid = grid
        self.positions = self.get_positions(grid)
        self.counts = [[0 for _ in row] for row in grid] #aby som vedel, ci pismenko bolo pridane teraz alebo uz tam bolo

    @staticmethod
    def get_positions(grid):
        """
        Identify all valid positions in the grid where words can be placed.

        A valid position:
          - consists of two or more consecutive empty cells (' ')
          - runs either horizontally ('right') or vertically ('down')
          - is bounded by walls ('#') or grid edges

        Returns:
            list[tuple[int, int, int, str]]: Each tuple represents a valid word position:
                (start_row, start_col, word_length, direction)
        """
        def check_line(line):
            res = []
            start_i, was_space = 0, False
            for i in range(len(line)):
                if line[i] == '#' and was_space:
                    was_space = False
                    if i - start_i > 1:
                        res.append((start_i, i - start_i))
                elif line[i] == ' ' and not was_space:
                    start_i = i
                    was_space = True
            return res

        poss = []

        # Check all horizontal positions
        for r in range(len(grid)):
            row = grid[r]
            poss = poss + [(r, p[0], p[1], 'right') for p in check_line(row)]

        # Check all vertical positions
        for c in range(len(grid[0])):
            column = [row[c] for row in grid]
            poss = poss + [(p[0], c, p[1], 'down') for p in check_line(column)]
        return poss
    
    def get_pattern(self, pos): # pomocna metoda pre solve, forward checking, skontrolujem, ci pridanim slova mi nevznikne situacia,
                                #ze nemam ani jednu moznost pre nejaku poziciu pos
        r, c, length, direction = pos
        pattern = ""
        for i in range(length):
            if direction == 'right':
                ch = self.grid[r][c + i]
            else:
                ch = self.grid[r + i][c]
            pattern += ch if ch != ' ' else '_'
        return pattern

    def matches_pattern(self, word, pattern):
        return all(p == '_' or p == w for p, w in zip(pattern, word))
        
    def print_grid(self):
        """Pretty-print the current crossword grid."""
        for row in self.grid:
            print(''.join(row))

    def text_at_pos(self, position):
        """
        Retrieve the text currently written at a given position.

        Args:
            position (tuple): (start_row, start_col, length, direction)

        Returns:
            str: The string currently occupying the given slot.
        """
        dr, dc = self.directions[position[3]]
        r, c = position[0], position[1]
        return ''.join([self.grid[r + pos * dr][c + pos * dc] for pos in range(position[2])])

    def write_word(self, position, word):
        """
        Write a valid word into the grid at the specified position.

        Args:
            position (tuple): (start_row, start_col, length, direction)
            word (str): The word to insert.

        Returns:
            list[tuple[int, int]]: List of coordinates that were modified.
        """
        changed = []
        dr, dc = self.directions[position[3]]
        r, c = position[0], position[1]
        for i in range(position[2]):
            rr, cc = r + i * dr, c + i * dc
            self.grid[rr][cc] = word[i]
            changed.append((rr, cc))
            self.counts[rr][cc] += 1
        return changed

    def erase_word(self, coords):
        """
        Remove previously written letters from the grid.

        Args:
            coords (list[tuple[int, int]]): List of (row, col) positions to clear.
        """
        for r, c in coords:
            self.counts[r][c] -= 1
            if self.counts[r][c] == 0:# odsttranim iba ak bola usage 1, cize bol znak pridany tymto slovom 
                self.grid[r][c] = ' '

    def can_write_word(self, position, word):
        """
        Check if a word can be safely placed at the specified position.

        Args:
            position (tuple): (start_row, start_col, length, direction)
            word (str): Candidate word.

        Returns:
            bool: True if the word can be placed, False otherwise.
        """

        # # # YOUR CODE GOES HERE # # #
        row, col, length, direction = position
        # if length != len(word): #teoreticky nemusi byt, lebo v backtracku beriem len slova spravnej dlzky
        #     return False
        
        if direction == "right":
            for i in range(length):
                if self.grid[row][col + i] != ' ' and self.grid[row][col + i] != word[i]:
                    return False

        else:
            for i in range(length):
                if self.grid[row + i][col] != ' ' and self.grid[row + i][col] != word[i]:
                    return False
        return True


def load_words(path):
    """Load a list of words from a file (one per line)."""
    return open(path, 'r').read().splitlines()


def load_grids(path):
    """ Load crossword grid definitions from a file. """
    raw = open(path, 'r').read().split('\n\n')
    per_rows = [grid.rstrip().split('\n') for grid in raw]
    per_char = [[list(row) for row in grid] for grid in per_rows]
    return per_char


def solve(crossword, words):
    """ Solve a crossword by filling all valid positions with words. """

    # # # YOUR CODE GOES HERE # # #
    positions = crossword.positions

    words_by_len = {}  # slovnik podla dlzky slov
    for word in words:
        length = len(word)
        if length not in words_by_len:
            words_by_len[length] = []   
        words_by_len[length].append(word) 

    tried_words = {} #aby pri backtracku som neskusal to iste slovo, napr ak vlozim hello a potom najdem dead end a backtrackem sem,
                    #slovnik mi zaruci, ze uz slovo hello nepouzijem

    def forward_checking(test_pos, test_word):
        changed = crossword.write_word(test_pos, test_word) # ziskam pozicie doplneneho slovicka
        for pos in positions:
            length = pos[2]
            pattern = crossword.get_pattern(pos)  #zistim v akom stave je kazda volna pos - pattern
            candidates = [ # slova, ktore viem doplnit do crossword, ak doplnim test_word
                word for word in words_by_len.get(length, [])
                if crossword.matches_pattern(word, pattern)
            ]
            if not candidates: # ak nemam ziadneho kandidata, test_word nemozno pouzit
                crossword.erase_word(changed)
                return False
        crossword.erase_word(changed) #aby som obnovil stav grid-u
        return True

    def backtrack():
        if not positions:
            return True

        best_pos = None #  MRV
        best_candidates = []
        for pos in positions:
            length = pos[2]
            candidates = [
                word for word in words_by_len.get(length, [])
                if crossword.can_write_word(pos, word)
            ]
            if not candidates: #ak nema ziadnu moznost pre poziciu pos
                continue
            if best_pos is None or len(candidates) < len(best_candidates):
                best_pos = pos
                best_candidates = candidates

        if best_pos is None: #ak nie je ziadny kandidat pre sucasny stav grid-u
            return False

        if best_pos not in tried_words: #innit slovnika
            tried_words[best_pos] = set()

        positions.remove(best_pos)
        for word in best_candidates:
            if word in tried_words[best_pos]:
                continue
            tried_words[best_pos].add(word)

            if not forward_checking(best_pos, word): #ak sa nam pokazi grid vlozenim tohto word
                continue  
            
            changed = crossword.write_word(best_pos, word)


            if backtrack():
                return True

            crossword.erase_word(changed)


        positions.append(best_pos)
        tried_words[best_pos].clear() #aby ked backtrackem slovo, ktore krizuje moju poziciu mi neostali v tried words slova, ktore viem pouzit teraz
        return False

    backtrack()


if __name__ == "__main__":
    # Load data:
    words = load_words('words.txt')
    grids = load_grids('krizovky.txt')

    # Examples:
    # dummy_grid = [list(s) for s in ['########', '#      #', '#      #', '#      #', '###    #', '#      #', '########']]
    # cw = CrossWord(dummy_grid)
    # cw.print_grid()  # empty grid
    # print('Positions: ' + str(cw.positions))
    # print(cw.can_write_word((2, 1, 5, 'right'), 'hello'))
    # cw.write_word((2, 1, 5, 'right'), 'hello')
    # cw.write_word((1, 5, 5, 'down'), 'world')
    # print(cw.can_write_word((3, 1, 5, 'right'), 'hello'))
    # cw.write_word((4, 3, 4, 'right'), 'milk')
    # cw.print_grid()  # 3 words already filled in
    # print('Text at position (1,4) down: "' + cw.text_at_pos((1, 4, 5, 'down')) + '"\n\n\n')
    for crossword_id in range(len(grids)): #priemer tak 300s
        print('==== Crossword No.' + str(crossword_id + 1) + ' ====')
        cw = CrossWord(grids[crossword_id])
        solve(cw, words)
        cw.print_grid()

