class MappingAdapter:
    def __init__(self, adaptee):
        self.adaptee = adaptee

    def lighten(self, grid):
        lights = []
        obstacles = []
        for ind_i, i in enumerate(grid):
            for ind_j, j in enumerate(i):
                if j == 1:
                    lights.append((ind_j, ind_i))
                elif j == -1:
                    obstacles.append((ind_j, ind_i))

        self.adaptee.set_dim((len(grid[0]), len(grid)))
        self.adaptee.set_obstacles(obstacles)
        self.adaptee.set_lights(lights)
        return self.adaptee.generate_lights()
