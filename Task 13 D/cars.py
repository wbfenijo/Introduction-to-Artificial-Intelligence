from genetic import *


class MyCar(Car):
    def __init__(self, chromosome, **kwargs):
        super(MyCar, self).__init__(**kwargs)

        self.input_dim = len(self.sensors) + 2  # two additional inputs for the current speed and the current turn
        self.output_dim = 2  # there are 2 options: 1, go left/right and 2, go forward/backward

        if chromosome is None:
            # # # YOUR CODE GOES HERE # # #
            # create the initial values for a chromosome -> list of random numbers
            # if you are using numpy to generate random numbers,
            # return python list of the result: list(generated_numbers)

            self.chromosome = [1., 2., 3., 4., 5., 6.]
        else:
            self.chromosome = chromosome

        # # # YOUR CODE GOES HERE # # #
        # create at least two numpy matrices from self.chromosome, which you will then use to control the car with
        # tip: try to use numpy.reshape(vector) to change a vector dimensions!

        self.W_in = np.array(0)
        self.W_out = np.array(0)

    def compute_output(self):
        sensor_norms = []
        # # # YOUR CODE GOES HERE # # #
        # create an array of sensor lengths. Afterwards, append to this array the speed and the turn of the car
        # (self.speed and self.turn). This array serves as an input to the network (but do not forget the bias)
        # the sensor lengths values are originally quite large - are in the interval <0, 200>. To prevent "explosion" in
        # the neural network, normalize the lengths to smaller interval, e.g. <0, 10>

        for point_1, point_2 in self.sensors:
            sensor_norms.append(9)

        # # # YOUR CODE GOES HERE # # #
        # use the input (inp) to calculate the instructions for the car at the given state
        # do not forget to add bias neuron before each projection!
        # for adding bias you can use np.concatenate(( state , [1]))
        # between two layers use an activation function. sigmoid(x) is already implemented, however, feel free to use
        # other, meaningful activation function
        # at the end of the network, use hyperbolic tangent -> to ensure that the change is in the range (-1, 1)

        out = np.zeros((2, 1))

        return out

    def update_parameters(self, instructions):
        # # # YOUR CODE GOES HERE # # #
        # use the values outputted from the neural network (instructions) to change the speed and turn of the car.
        # In order to be safe, the cars also have speed limit (self.speed_limit[]), to prevent accidents. You are
        # advised to incorporate it into the code. Similarly, self.turn_limit prevents huge turns at a given time.

        self.speed += np.random.randn()     # FIXME
        self.turn += np.random.randn()      # FIXME

        self.orientation += self.turn


class MyRaceCars(RaceCars):
    def __init__(self, *args, **kwargs):
        super(MyRaceCars, self).__init__(*args, **kwargs)

    def generate_individual(self, chromosome=None):
        initial_position = [500, 600 + random.randint(-25, 25)]
        orientation = 90
        new_car = MyCar(chromosome, position=initial_position, orientation=orientation,
                        additional_scale=self.scale_factor)
        sensors = []
        if self.show_sens:
            for _ in new_car.sensors:
                sensors.append(self.canvas.create_line(0, 0, 0, 0, dash=(2, 1)))
        return [new_car, self.canvas.create_image(initial_position), sensors]


if __name__ == '__main__':
    track = create_track()

    # useful parameters to play with:
    population_size = 16        # total population size used for training
    select_top = 4              # during selection, only the best select_top cars are chosen as parents

    show_training = True        # each generation is shown on canvas
    show_all_cars = False       # the code is faster if not all cars are always displayed
    displayed_cars = 8         # only the first few cars are displayed

    # show_training = False     # the training is done in the background
    show_every_n = 3            # the best cars are shown after every few generations (due to faster training)

    mutation_prob = 0.05        # mutation probability for number mutation
    deviation = 1               # this standard deviation used when mutating a chromosome

    RC = MyRaceCars(track, population_size=population_size, show_sensors=False, gen_steps=500, n_gens=100,
                    show_all_cars=show_all_cars, select_top=select_top, mutation_prob=mutation_prob,
                    show_training=show_training, displayed_cars=displayed_cars, vis_pause=10, show_every_n=show_every_n,
                    can_width=1100, deviation=deviation)

    RC.run_simulation()
