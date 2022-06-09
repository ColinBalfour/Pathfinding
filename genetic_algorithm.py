from copy import deepcopy
import math

import numpy as np
from random import randint

import tkinter as tk

import a_star_algo as algo


### lol I gave up on this file we've now moved to just one file ###
# why? Because I suck at programming #

class Player:
    NUM_LAYERS = 4
    LAYER_SIZES = 5, 8, 8, 1

    def __init__(self, pop_size, app):
        self.pop_size = pop_size
        self.app = app
        self.app.algorithm = True

    def create_population(self):
        self.population = []
        for i in range(self.pop_size):
            net = Network(Player.NUM_LAYERS, Player.LAYER_SIZES, self.app)
            net.create_weights()
            self.population.append(net)
    
    def fitness(self):

        start = randint(0, self.app.cols), randint(0, self.app.rows)
        end = randint(0, self.app.cols), randint(0, self.app.rows)

        def setup():
            self.app.draw_start(*start)
            self.app.draw_end(*end)

            for x in range(10, 15):
                for y in range(10, 15):
                    self.app.draw_wall(x, y)

        for net in self.population:
            print('done:', net)
            setup()
            self.app.solver.heuristic = net.run_network
            self.app.run_program()
            self.app.reset_program()
        

class Network:

    def __init__(self, num_layers: int, layer_sizes: list[int], app):
        if len(layer_sizes) != num_layers or num_layers < 3:
            self.fuck_you()

        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        self.input_layer = None

        self.app = app

    def create_weights(self):

        self.hidden_weights = []
        for i in range(1, self.num_layers):
            size = self.layer_sizes[i], self.layer_sizes[i-1]
            weights = np.random.rand(*size).round(3)
            self.hidden_weights.append(weights)

        self.output_weights = self.hidden_weights.pop(-1)

    def create_input(self, pos):
        app = self.app

        dist_to_end = math.dist(pos, app.end)
        wall_top, wall_bottom, wall_left, wall_right = [False] * 4

        # Fuck me this code is shit please fix it
        for wall in app.walls:
            wall = wall[1] # get coords

            if wall[0] == pos[0]:
                if wall[1] > pos[1] and not wall_top:
                    wall_top = True
                if wall[1] < pos[1] and not wall_bottom:
                    wall_bottom = True

            if wall[1] == pos[1]:
                if wall[0] > pos[0] and not wall_right:
                    wall_right = True
                if wall[0] < pos[0] and not wall_left:
                    wall_left = True

        # fuck fuck fuck the code is so shit why
        wall_top = 1 if wall_top else 0
        wall_bottom = 1 if wall_bottom else 0
        wall_right = 1 if wall_right else 0
        wall_left = 1 if wall_left else 0

        self.input_layer = np.array([dist_to_end, wall_top, wall_bottom, wall_right, wall_left])

    def run_network(self, pos):
        self.create_input(pos)

        layer = self.input_layer
        for weights in self.hidden_weights:
            layer = np.dot(weights, layer)
        
        output = np.dot(self.output_weights, layer)
        return output

    def fuck_you(self):
        print('fuck you give me proper inputs')
        quit()


if __name__ == '__main__':
    root = tk.Tk()

    width, height = 25, 25
    box_size = 30
    
    app = algo.App(root, (width, height), box_size, True)
    app.pack(side="top", fill="both", expand=True)

    pop_size = 1
    player = Player(pop_size, app)
    player.create_population()
    player.fitness()

    root.mainloop()

