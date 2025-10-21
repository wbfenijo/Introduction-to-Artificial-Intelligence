import sys
import time
from maps import *

class MapCSP():
    def __init__(self, states, neighbours):
        # List of available colors
        self.color_options = ['red', 'green', 'blue', 'yellow']
        self.states = states
        self.neighbours = neighbours
        self.colors = {s: None for s in self.states}

    def print_map(self):
        # Prints all states and their colors
        for s in sorted(self.states):
            print('{} has color: {}'.format(s, self.get_color(s)))
        print()


    def set_color(self, state, color):
        # Assign color to a state
        self.colors[state] = color

    def del_color(self, state):
        # Remove color from state - reset to None
        self.colors[state] = None

    def get_color(self, state):
        # Get color assigned to a state
        return self.colors[state]

    def has_color(self, state):
        # Returns True if state has already a color
        return self.colors[state] != None

    def same_colors(self, state1, state2):
        # Returns True if state1 and state2 are colored with the same color.
        return self.has_color(state1)  and  self.get_color(state1) == self.get_color(state2)

    def all_colored(self):
        # Returns True if all states of the map are already colored.
        return all([self.has_color(s) for s in self.states])

    def is_correct_coloring(self):
        # Returns True if coloring is all correct, False if not. Prints the result with found error (if any).
        print('Coloring is ', end='')
        for s1 in self.states:
            if self.get_color(s1) not in self.color_options:
                print('INCORRECT - {} has invalid color: {}\n'.format(s1, self.get_color(s1)))
                return False
            for s2 in self.neighbours[s1]:
                if self.same_colors(s1,s2):
                    print('INCORRECT - {} and {} have conflicting color {}\n'.format(s1, s2, self.get_color(s1)))
                    return False
        print('OK\n')
        return True


    def can_set_color(self, state, color):
        # Returns True if we can set color to a state without violating constrains - all neighbouring
        # states must have None or different color.
        # TODO Task 1: implement simple backtracking
        return color not in set(self.colors[stat] for stat in self.neighbours[state])

    def select_next_state(self, use_heuristic=True):
        # Selects next state that will be colored, or returns False if no such exists (all states are
        # colored). You can use heuristics or simply choose a state without color for start.
        if use_heuristic:
            best_state = []
            for state in self.states:
                if self.colors[state] == None:
                    poss_colors = []
                    for c in self.color_options:
                        if self.can_set_color(state, c):
                            poss_colors.append(c)
                    best_state.append((state,len(poss_colors)))
            best_state = sorted(best_state, key= lambda x: x[1])
            # TODO Task 2: use heuristics to speed up the program (should solve world map)
            return False if not best_state else best_state[0][0]
        else:
            # TODO Task 1: implement simple backtracking
            for state in self.states:
                if self.colors[state] == None:
                    return state
            return False

    def color_map(self):
        # Assign colors to all states on the map. (! Beware: 'map' is python`s reserved word - function)
        # TODO Task 1: implement simple backtracking
        state = self.select_next_state()
        if not state:
            return True
        for c in self.color_options:
            if self.can_set_color(state, c):
                self.set_color(state, c)
                if self.color_map():
                    return True
                self.del_color(state)
        return False




if __name__ == "__main__":
    maps = [('Australia', AustraliaMap()),
            ('USSR', USSRMap()),
            ('USA', USAMap()),
            ('World', WorldMap()),  # TODO uncomment for solution with heuristic

            ('Impossible Australia', ImpossibleMap(AustraliaMap())),
            ('Impossible USSR', ImpossibleMap(USSRMap())), # type: ignore
            # ('Impossible USA', ImpossibleMap(USAMap())),  # TODO uncomment for solution with heuristic
            # ('Impossible World', ImpossibleMap(WorldMap()))  # TODO uncomment for solution with heuristic
            ]

    for name, mapa in maps:
        print('==== {} ===='.format(name))
        t = time.time()
        has_result = mapa.color_map()    # Compute the colors for an empty map
        print('Time: {:.3f} ms'.format( (time.time() - t)*1000 ))
        if has_result:
            mapa.is_correct_coloring()  # Print whether coloring is correct
        else:
            print('Coloring does not exist\n')
        # mapa.print_map()    # Print whole coloring
