import numpy as np
import random

from problems import *

class GenAlgProblem:

    def __init__(self, population_size=12, n_crossover=3, mutation_prob=0.05):
        # Initialize the population - create population of 'size' individuals,
        # each individual is a bit string of length 'word_len'.
        self.population_size = population_size
        self.n_crossover = n_crossover
        self.mutation_prob = mutation_prob
        self.population = [self.generate_individual() for _ in range(self.population_size)]

    def generate_individual(self):
        # Generate random individual.
        # To be implemented in subclasses
        raise NotImplementedError

    def show_individual(self, x):
        # Show the given individual x, either to console or graphically.
        # To be implemented in subclasses
        raise NotImplementedError

    def show_population(self, title='Population:', limit=None, **kwargs):
        # Show whole population.
        # To be implemented in subclasses
        raise NotImplementedError


    def fitness(self, x):
        # Returns fitness of a given individual.
        # To be implemented in subclasses
        raise NotImplementedError

    def crossover(self, x, y, k):
        # Take two parents (x and y) and make two children by applying k-point
        # crossover. Positions for crossover are chosen randomly.
        ### YOUR CODE GOES HERE ###
        n = len(x)
        points = sorted(random.sample(range(1, n), k)) # toto mam z internetu - ako zvolit random cisla bez opakovania
        points.append(n) #aby sme mali aj index konca listu

        x_new, y_new = [], []
        last = 0
        swap = False

        for point in points:
            if  not swap:
                x_new.extend(x[last:point])
                y_new.extend(y[last:point])
            else:
                x_new.extend(y[last:point])
                y_new.extend(x[last:point])
            swap = not swap
            last = point

        return x_new, y_new

    def boolean_mutation(self, x, prob):
        # Elements of x are 0 or 1. Mutate (i.e. change) each element of x with given probability.
        ### YOUR CODE GOES HERE ###
        mutated = []
        for value in x:
            mutated.append((value + 1) % 2 if random.random() <= prob else value)
        return mutated
                                                                          
    def number_mutation(self, x, prob):
        # Elements of x are real numbers [0.0 .. 1.0]. Mutate (i.e. add/subtract random number)
        # each number in x with given probability.
        ### YOUR CODE GOES HERE ###
        mutated = []
        for value in x:
            mutated.append(min(1.0, max(0.0, value + random.gauss(0, 0.05))) if random.random() <= prob else value)#gauss - normalne rozdelenie
        return mutated

    def mutation(self, x, prob):
        # To be specified in subclasses, uses boolean_mutation or number_mutation functions
        raise NotImplementedError

    def solve(self, max_generations, goal_fitness=1):
        # Implementation of genetic algorithm. Produce generations until some
        # individual`s fitness reaches goal_fitness, or you exceed total number
        # of max_generations generations. Return best found individual.
        ### YOUR CODE GOES HERE ###
        for generation in range(max_generations):
                temp = [self.fitness(i) for i in self.population] # fitnesy
                sorted_population = [i for _, i in sorted(zip(temp, self.population), key=lambda x: x[0], reverse=True)] # zoradenie populacie podla fitnesov
                best_fitness = self.fitness(sorted_population[0])

                if best_fitness >= goal_fitness: #ciel
                    return sorted_population[0]

                half = self.population_size // 2 #polovica maju byt top rodicia
                new_population = sorted_population[:half]
                
                while len(new_population) < self.population_size:
                    parent1, parent2 = random.sample(new_population, 2) # random vyber rodicov
                    child1, child2 = self.crossover(parent1, parent2, self.n_crossover)
                    child1 = self.mutation(child1, self.mutation_prob)
                    child2 = self.mutation(child2, self.mutation_prob)                                                                  
                    new_population.extend([child1, child2])

                self.population = new_population[:] 
        
        fitnesses = [self.fitness(i) for i in self.population]
        best_index = np.argmax(fitnesses)
        return self.population[best_index]



if __name__ == "__main__":
    ## Choose problem
    ga = OnesString()
    ga = Smiley()
    ga = Painting(r'C:\Users\adamy\OneDrive\Plocha\ItAI\Task 04 D\painting.jpg', population_size=12, mutation_prob=0.25)
    ga.show_population('Initial population', limit=None)

    ## You can play with parameters
    # ga.n_crossover = 5
    # ga.mutation_prob = 0.1

    ## Solve to find optimal individual
    best = ga.solve(100) # you can also play with max. generations
    ga.show_population('Final population', limit=None)
    ga.show_individual(best, 'Best individual')


    ## Test your crossover function
    # ga = OnesString()
    #children = ga.crossover([0]*32, [1]*32, k=3)
    #print('{}\n{}'.format(*children))
