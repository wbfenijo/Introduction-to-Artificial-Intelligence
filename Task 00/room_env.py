import random
import matplotlib.patches as patches # type: ignore
import matplotlib.lines as mlines # type: ignore


class Room:
    def __init__(self):
        # Environment using "floor cells"
        self.valid_cells = {
            (0, 0), (1, 0), (2, 0), (3, 0),
            (0, 1), (3, 1),
            (0, 2), (1, 2),
            (3, 2),
            (0, 3), (1, 3),
            (3, 3)
            }
        self.valid_moves = dict()
        for x,y in self.valid_cells:
            self.valid_moves[(x,y)] = list()
        for x,y in self.valid_cells:
            if (x + 1, y) in self.valid_cells:
                self.valid_moves[(x,y)].append("RIGHT")
            if (x - 1, y) in self.valid_cells:
                self.valid_moves[(x,y)].append("LEFT")
            if (x, y + 1) in self.valid_cells:
                self.valid_moves[(x,y)].append("UP")
            if (x, y - 1) in self.valid_cells:
                self.valid_moves[(x,y)].append("DOWN")

        # Environment size: bounding box for drawing
        self.grid_width = max(x for x, y in self.valid_cells) + 1
        self.grid_height = max(y for x, y in self.valid_cells) + 1
        self.percept = set()

        self.robot_position = (1, 0)  # start in left corridor

        # Possible actions
        self.actions = ["UP", "DOWN", "LEFT", "RIGHT", "SUCK", "NOOP"]
        self.dirt_position = set()
        self.last_action = "NOOP"
        self.previous_position = self.robot_position

        self.performance = 0
        self.has_cleaned = False

    def make_mess(self, mode='fixed', dirt_probability=1.):
        """
        'fixed': adds 3 fixed dirt
        'as always': fills the room with dirt
        'single_with_prob': makes dirty a single tile with a given prob
        """
        dirt = self.dirt_position
        if mode == 'fixed':
            dirt.add((0, 0))
            dirt.add((3, 1))
            dirt.add((1, 3))
        elif mode == 'as always':
            for cell in self.valid_cells:
                dirt.add(cell)
        elif mode == 'single_with_prob':
            random_cell = random.choice(list(self.valid_cells))
            if random.random() < dirt_probability:
                dirt.add(random_cell)
        return dirt

    def update_state(self, act, clean_prob=1.):
        """Update environment state given action."""
        x, y = self.robot_position
        self.previous_position = self.robot_position
        self.percept = set()
        self.has_cleaned = False

        new_pos = (x, y)
        if act == "SUCK":
            if random.random() < clean_prob:
                self.dirt_position.discard(self.robot_position)
                self.has_cleaned = True
        elif act == "UP":
            new_pos = (x, y + 1)
        elif act == "DOWN":
            new_pos = (x, y - 1)
        elif act == "RIGHT":
            new_pos = (x + 1, y)
        elif act == "LEFT":
            new_pos = (x - 1, y)

        # Only move if new position is valid
        if new_pos in self.valid_cells:
            self.robot_position = new_pos
        else:
            self.percept.add("bump")    # the vacuum cleaner cannot move

        if self.robot_position in self.dirt_position:
            self.percept.add("dirt")    # the vacuum cleaner senses dirt

    def draw(self, axis, current_step):
        """Draw the irregular grid, dirt, and robot."""
        axis.clear()

        axis.set_xlim(0, self.grid_width)
        axis.set_ylim(0, self.grid_height)
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_aspect('equal')

        # Hide the frame lines (spines)
        for spine in axis.spines.values():
            spine.set_visible(False)

        # Draw valid floor cells
        for (x, y) in self.valid_cells:
            axis.add_patch(patches.Rectangle((x, y), 1, 1, fill=True, color="whitesmoke", ec="black"))

        # Draw dirt as gray stars
        for (x, y) in self.dirt_position:
            axis.plot(x + 0.5, y + 0.5, marker="*", markersize=20, color="dimgray")

        # Draw robot as a circle with arrow
        rx, ry = self.robot_position
        body = patches.Circle((rx + 0.5, ry + 0.5), 0.3, color="skyblue", ec="black")
        axis.add_patch(body)

        # generate arrows according to the last action
        action_to_arrow = {'DOWN': (0, -0.2), 'UP': (0, 0.2), 'LEFT': (-0.2, 0), 'RIGHT': (0.2, 0), 'NOOP': (0, 0),
                           'SUCK': (0, 0)}

        dx, dy = action_to_arrow[self.last_action]
        arrow = mlines.Line2D([rx + 0.5, rx + 0.5 + dx], [ry + 0.5, ry + 0.5 + dy],
                              color="black", linewidth=2)
        axis.add_line(arrow)

        axis.set_title(f"Step {current_step}  |  Performance: {self.performance_measure()}")

    def take_action(self):
        pass

    def performance_measure(self):
        pass
