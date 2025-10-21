import random
import matplotlib.pyplot as plt # type: ignore
from room_env import Room


class MyRoom(Room):
    def __init__(self):
        Room.__init__(self)

    def take_action(self, agent='model_based'):
        current_action = "NOOP"

        # Useful info
        # Possible actions: {"UP", "DOWN", "LEFT", "RIGHT", "SUCK", "NOOP"}
        # self.valid_cells: set of tuples
        # self.percept: set containing 'dirt' when standing on a dirty tile; 'bump' if the previous action hit the wall
        # self.last_action: string with the last action
        # self.previous_position: previous robot position
        # self.robot_position: tuple (one from self.valid_cells)
        # self.has_cleaned: boolean, whether last action cleaned dirt

        # Tasks:
        # 1) implement a reflex agent
        # 2) implement a model-based agent
        # 3) compute performance_score of the agent
        # 4) make the environment non-deterministic (stochastic)
        # 5) experiment with the agents under different conditions

        # TODO Define a reflex agent
        if agent == 'reflex':
            if self.robot_position in dirt_positions:
                current_action = "SUCK"
            else:
                current_action = random.choice(["UP","DOWN","LEFT", "RIGHT"])

            

        # TODO Define a model-based agent
        elif agent == 'model_based':
            if self.robot_position in dirt_positions:
                current_action = "SUCK"
            else:
                current_action = random.choice(self.valid_moves[self.robot_position])

        self.last_action = current_action
        return current_action

    # returns a number
    def performance_measure(self):
        return 42


if __name__ == '__main__':
    my_room = MyRoom()
    dirt_positions = my_room.make_mess('as always')
    cleaning_probability = 1    # probability that the vacuum cleaner cleans the dirt after one attempt
    agent_type = 'model_based'       # 'reflex' or 'model_based'

    fig, ax = plt.subplots(figsize=(6, 6))

    # Simulation loop
    for step in range(200):
        my_room.draw(ax, step)
        plt.pause(0.1)
        action = my_room.take_action(agent=agent_type)
        my_room.update_state(action, cleaning_probability)

        # makes the room dirty again
        my_room.make_mess(mode='single_with_prob', dirt_probability=0)
        print(f"Step {step}: Action = {action}")

    # Final state
    my_room.draw(ax, "Final")
    plt.show()
