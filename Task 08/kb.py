from logic import *


class KB:
    #  Representation of whole knowledge base
    def __init__(self):
        self.clauses = []
        self.visited = set()

    def __str__(self):
        return 'Knowledge base:\n{}\n'.format('\n'.join(['\t' + str(c) for c in self.clauses]))

    def print_facts(self):
        print('Facts:\n{}\n'.format('\n'.join(['\t' + str(c) for c in self.clauses if len(c.premises) == 0])))

    def tell(self, clause):
        # Tell sentence (Implication/Fact) into KB
        if not isinstance(clause, Implication):
            raise TypeError('You can only tell clauses (Implication/Fact)! You passed: {}'
                            .format(type(clause).__name__))
        if clause not in self.clauses:
            self.clauses.append(clause)

    def find_clauses_with_conclusion(self, conclusion):
        # Find all clauses with given conclusion (Literal), sort them Facts first
        return sorted([c for c in self.clauses if c.conclusion == conclusion], key=lambda c: len(c.premises))

    def ask(self, goal_literal):
        # Backward-chaining
        if not isinstance(goal_literal, Literal):
            raise TypeError('You can only ask Literal! You asked: {}'
                            .format(type(goal_literal).__name__))
        # # # Task 1 & 2
        # # # YOUR CODE GOES HERE # # #

        def backward_chaining(goal, visited):
                for clause in self.clauses:
                    if len(clause.premises) == 0 and clause.conclusion == goal:
                        return True

                key = str(goal)
                if key in visited:  
                    return False
                visited.add(key)

                for clause in self.find_clauses_with_conclusion(goal):
                    if len(clause.premises) == 0:
                        continue

                    all_true = True
                    for premise in clause.premises:
                        if not backward_chaining(premise, visited):
                            all_true = False
                            break

                    if all_true:
                        self.tell(Fact(goal))  
                        return True

                return False

        result = backward_chaining(goal_literal, set())

        if result:
            self.tell(Fact(goal_literal))  

        return result



if __name__ == "__main__":
    # Task 1 - simple inference
    kb = KB()
    kb.tell(Implication(NOT('P'), L('Q')))  # -P => Q
    kb.tell(Implication(NOT('C'), L('P')))  # -C => P
    kb.tell(Implication(NOT('K'), L('M'), NOT('P')))  # -K ^ M => -P
    kb.tell(Implication(L('B'), NOT('K'), L('M')))  # B ^ -K => M
    kb.tell(Implication(L('A'), L('B'), NOT('K')))  # A ^ B => -K
    kb.tell(Fact(L('A')))
    kb.tell(Fact(L('B')))
    kb.tell(Fact(L('C')))
    print('Simple inference True: ', kb.ask(L('Q')))
    print('Simple inference False:', kb.ask(L('P')))

    # Task 2 - inference with cyclic rules
    kb_cyclic = KB()
    kb_cyclic.tell(Implication([L('A'), NOT('P')], NOT('K')))  # A ^ -P => -K
    for clause in kb.clauses:
        kb_cyclic.tell(clause)
    print('Cyclic rules True: ', kb_cyclic.ask(L('Q')))
    print('Cyclic rules False:', kb_cyclic.ask(L('P')))

    # # # Examples # # #
    # """
    # Literals:
    _wum_dead = Literal('wumpus_dead', L)  # Literal whose truth value is L
    wum_dead = L('wumpus_dead')  # The same thing, just faster to write
    not_wum_dead = -wum_dead  # Negation - truth value of not_wum_dead is False
    print(wum_dead, ',', not_wum_dead)

    _breeze_at_position_5_7 = ParamLiteral('breeze', 5, 7, L)  # Parametrized literal
    breeze_at_position_5_7 = L('breeze', 5, 7)  # The same param. literal

    if wum_dead:
        print('It`s safe to test literals in "if" - it checks its truth value')

    if not_wum_dead:
        pass
    else:
        print('Truth value of not_wum_dead is False')

    if L('abc') == -NOT('abc'):
        print('It`s safe to "==" and "!=" literals')
    if NOT('wumpus_at', 3, 5) == ParamLiteral('wumpus_at', 3, 5, False):
        print(
            '"L(...)" and "NOT(...)" are just convenience functions, you can use full constructors for Literal and ParamLiteral')

    # Clauses/sentences
    c1 = Fact(wum_dead)  # This is a simple fact, constructor takes one Literal
    c2 = Fact(L('breeze', 5, 7))  # Another simple fact
    c3 = Implication(
        L('wumpus_dead'), L('wumpus_at', 3, 5),  # Implication takes several premises,
        L('safe', 3, 5),  # and one conclusion (Literal)
    )

    _c4 = Implication(NOT('wumpus_at', 3, 5), NOT('pit', 3, 5),
                      L('safe', 3, 5))  # Another implication
    c4 = Implication([NOT('wumpus_at', 3, 5), NOT('pit', 3, 5)],
                     L('safe', 3, 5))  # The same thing

    list_of_premises = c4.premises  # You can access premises...
    conclusion = c4.conclusion  # ...or conclusion

    # Knowledge base
    kb = KB()
    kb.tell(c1)
    kb.tell(c2)
    kb.tell(c3)
    kb.tell(c4)  # Put clauses (Implications/Facts) to KB
    print(kb.ask(wum_dead))  # Ask the KB for Literals, not Facts or Implications
    print('\n', kb)
    print(kb.find_clauses_with_conclusion(L('safe', 3, 5)))  # Find appropriate clauses
    # """
