# wheeeeeee
import math
import time
import random

from copy import deepcopy
from random import randint, randrange

from multiprocessing import Process

import datetime
import heapq

import tkinter as tk

import numpy as np

DEBUG = False
DEBUG2 = True

COUNT = 0
COUNT2 = 0

class App(tk.Frame):

    def __init__(self, master: tk.Tk, size: tuple=(25, 25), box_size: int=30, algorithm: bool=False, pop_size: int=100):
        tk.Frame.__init__(self, master)

        #testing
        self.use_net = True

        self.master = master

        title = "A* With a Heuristic Trained by Genetic Algorithm" if algorithm else "A* Player"
        master.title(title)

        self.cols, self.rows = self.size = size

        self.width = size[0] * box_size
        self.height = size[1] * box_size
        self.box_size = box_size

        self.algorithm = algorithm

        self.canvas = tk.Canvas(self, width=self.width, height=self.height)
        self.canvas.grid(row=5, columnspan=1)

        self.canvas.bind('<Button-1>', self.on_left_click)
        self.canvas.bind('<B2-Motion>', self.on_scroll_hold)
        self.canvas.bind('<Button-3>', self.on_right_click)
        self.master.bind('<Key>', self.on_key_press)
        self.master.bind('<Delete>', self.on_delete)
        self.master.bind('<Motion>', self.motion)

        self.settings = Empty()

        self.init_UI()
        self.init_vars()
        
        self.solver = Solver(self, self.cols, self.rows)
        self.player = Player(pop_size, self)
        
    def init_UI(self):
    
        def remove_menu():
            key.destroy()
            hide.destroy()

        key_text = "Start Point: Left Click\tEnd Point: Right Click\tWalls: Mousewheel Click\t\n"+\
                   "Start Program: Space\tReset Program: Delete"
        key = tk.Label(self, text=key_text, font="Helvetica 16 bold", padx=10, pady=10)
        key.grid(row=0, columnspan=2)

        hide = tk.Button(self, text="Hide this menu", command=remove_menu)
        hide.grid(row=1, column=0, columnspan=2)

        if self.algorithm:
            remove_menu()

            self.settings = SettingsMenu(self)
            self.settings.init_UI()

            ### TRACKING GENERATION NUM ###
            
            self.generation_num = 0
            self.generation_str = tk.StringVar()
            self.generation_str.set(f"Generation: {self.generation_num}")
            gen_box = tk.Label(self, textvariable=self.generation_str, font="Helvetica 16 bold")
            gen_box.grid(row=0, columnspan=1)

            ### STARTING AND STOPPING ###
            self.algorithm_running = False
            self.in_progress = False

            self.running_status_text = tk.StringVar()
            self.running_status_text.set("Start Algorithm")
            start = tk.Button(self, textvariable=self.running_status_text, command=self.run_algorithm)
            start.grid(row=1, column=0)

        self.create_board()
    
    def init_vars(self):

        self.start = None
        self.end = None

        self.walls = set()
        self.drawn_walls = set()
        self.drawn_explorations = set()
        self.neighbors_explored = set()
        self.drawn_objects = set()
    
    def create_board(self):
        for x in range(self.cols):
            x = x * self.box_size
            self.canvas.create_line(x, 0, x, self.height)
            
        for y in range(self.rows):
            y = y * self.box_size
            self.canvas.create_line(0, y, self.width, y)

    def coords_to_grid(self, x, y):
        """Takes (x, y) coordinates and converts them to the grid coordinates (col, row)
        """

        # Get grid num
        col = x // self.box_size
        row = y // self.box_size

        return col, row

    def grid_to_box(self, col, row):
        """Takes (col, row) coordinates and converts them to the top left and bottom right corners of that box
        """

        # Get the range of x and y values for the box
        x_range = col * self.box_size, (col + 1) * (self.box_size)
        y_range = row * self.box_size, (row + 1) * (self.box_size)

        # Put it together (left x, top y, right x, bottom y)
        return x_range[0], y_range[0], x_range[1], y_range[1]

    def coords_to_box(self, x, y):
        """Takes (x, y) coordinates and converts them to the top left and bottom right corners of the nearest box
        """

        # Get grid num
        col, row = self.coords_to_grid(x, y)

        # Get box coords
        return self.grid_to_box(col, row)

    def motion(self, event):
        self.x, self.y = event.x, event.y

    def on_left_click(self, event):
        x, y = event.x, event.y
        self.draw_start(x, y)

    def on_right_click(self, event):
        x, y = event.x, event.y
        self.draw_end(x, y)

    def on_scroll_hold(self, event):
        x, y = event.x, event.y
        self.draw_wall(x, y)

    def on_key_press(self, event):
        # print(event.char)
        if event.char == ' ':
            self.on_space(event)
        if event.char == 'z':
            self.draw_wall(self.x, self.y)

        # Test cases
        if event.char == 'q':
            self.test_run()
        if event.char == 'a':
            self.use_net = not self.use_net

    def on_space(self, event):
        self.run_program()

    def on_delete(self, event):
        self.reset_program()

    def draw_start(self, x, y):
        # Remove old start
        if self.start:
            self.canvas.delete(self.start[0])

        color = 'red'
        box_coords = self.coords_to_box(x, y)
        grid_coords = self.coords_to_grid(x, y)

        self.start = self.canvas.create_rectangle(box_coords, fill=color), grid_coords
    
    def draw_end(self, x, y):
        # Remove old end
        if self.end:
            self.canvas.delete(self.end[0])

        color = 'blue'
        box_coords = self.coords_to_box(x, y)
        grid_coords = self.coords_to_grid(x, y)

        self.end = self.canvas.create_rectangle(box_coords, fill=color), grid_coords

    def draw_wall(self, x, y):
        color = 'purple'
        box_coords = self.coords_to_box(x, y)

        path = self.canvas.create_rectangle(box_coords, fill=color)
        self.walls.add(self.coords_to_grid(x, y))
        self.drawn_walls.add(path)

        self.update()
        return path, box_coords

    def draw_exploration(self, grid_pos):  # sourcery skip: move-assign
        show_steps = self.settings.show_steps.get()
        pos_drawn = grid_pos in self.drawn_explorations
        if not show_steps or pos_drawn:
            return

        color = 'yellow'
        if grid_pos != self.start[1]:
            coords = self.grid_to_box(*grid_pos)
            obj = self.canvas.create_rectangle(coords, fill=color)
            self.drawn_objects.add(obj)
            self.drawn_explorations.add(grid_pos)
            
        self.update()
    
    def draw_neighbors(self, neighbors):
        if not self.settings.show_steps.get():
            return

        color = 'lime'
        for grid_pos, _ in neighbors:
            coords = self.grid_to_box(*grid_pos)

            if grid_pos not in [*self.neighbors_explored, self.start[1], self.end[1]]:
                obj = self.canvas.create_rectangle(coords, fill=color)
                self.neighbors_explored.add(grid_pos)
                self.drawn_objects.add(obj)
        
        self.update()

    def draw_path(self, path):
        color = 'deep sky blue'
        for grid_pos in path:
            if grid_pos not in [self.start[1], self.end[1]]:
                coords = self.grid_to_box(*grid_pos)
                obj = self.canvas.create_rectangle(coords, fill=color)
                self.drawn_objects.add(obj)

    
    def run_program(self):
        def run():
            if not self.algorithm:
                window.destroy()

            self.solver.create_grid()
            self.solver.find_path(start, end)

        if self.start and self.end:
            start = self.start[1]
            end = self.end[1]
            
            

            if not self.algorithm:
                window = tk.Toplevel(self)
                window.geometry = ("200x200")

                self.settings.show_steps = tk.BooleanVar()
                steps_box = tk.Checkbutton(window, text="Show Steps", var=self.settings.show_steps)

                self.settings.sleep_time = tk.StringVar()
                self.settings.sleep_time.set('0')
                delay_input = tk.Entry(window, textvariable=self.settings.sleep_time)

                done = tk.Button(window, text="Done", command=run)

                steps_box.pack()
                delay_input.pack()
                done.pack() 

            # Don't wait for button press and run immidiately 
            if self.algorithm:
                run()
        
    def reset_program(self):
        self.canvas.delete(self.start[0])
        self.canvas.delete(self.end[0])
        
        for wall in self.drawn_walls:
            self.canvas.delete(wall)
        
        for obj in self.drawn_objects:
            self.canvas.delete(obj)
        
        self.init_vars()

    def run_algorithm(self):
        self.algorithm_running = not self.algorithm_running
        status = "Pause Algorithm" if self.algorithm_running else "Start Algorithm"
        self.running_status_text.set(status)

        while self.algorithm_running and not self.in_progress:
            self.in_progress = True
            t = datetime.datetime.now()

            # update generation number
            self.generation_num += 1
            self.generation_str.set(f"Generation: {self.generation_num}")
            print(self.generation_str.get())

            # Run Algorithm
            self.player.run()

            print('RUNTIME:', datetime.datetime.now() - t)
            print()
            self.in_progress = False
    
    def test_run(self):
        pop_size = 1
        player = Player(pop_size, self)
        player.create_population()
        print('VAR', self.use_net)
        player.test_run(self.use_net)
    
    @staticmethod
    def get_circle_coords(center, radius):

        x1 = center[0] - radius
        y1 = center[1] - radius

        x2 = center[0] + radius
        y2 = center[1] + radius

        return x1, y1, x2, y2

    def create_network_lines(self, net):
        
        # save line objects
        self.lines = set()

        current = []
        for i, layer in enumerate(self.neurons):
            # Create a list of prev layer positions
            prev = current
            current = []
            for j, (pos, circle_obj) in enumerate(layer):
                # No lines are drawn for the first layer
                if i == 0:
                    j = -1
                    current.append(pos)
                    continue
                
                # Loop through prev layer pos as starting points
                for k, start_pos in enumerate(prev):
                    # use i-1 since weights have 1 less layer (no input weights)
                    color = 'blue' if net.weights[i-1][j][k] >= 0 else 'red'
                    weight = abs(int(net.weights[i-1][j][k] * 5))
                    line = self.network_canvas.create_line(*start_pos, *pos, fill=color, width=weight)
                    self.lines.add(line)

                current.append(pos)

    def create_neurons(self, net):
        WINDOW_SIZE = 1920, 1080

        # Set padding/radius of nodes
        radius = 50
        x_padding = int(WINDOW_SIZE[0] // (1.5 * net.num_layers))
        y_padding = WINDOW_SIZE[1] // max(net.layer_sizes)


        # Create a list of all neurons: [layer[(pos, circle)]]
        # And the text created inside them
        self.neurons = []
        self.neuron_text = set()
        self.neuron_objects = set()

        # Loop through each layer
        x_offset = (WINDOW_SIZE[0] - (2 * x_padding)) // (net.num_layers - 1)
        for i, layer_size in enumerate(net.layer_sizes):

            # y_padding = WINDOW_SIZE[1] // layer_size
            
            layer = [] # stores neurons for each layer

            # Set to middle of screen when there is 1 node
            if layer_size == 1:
                center = (i * x_offset) + x_padding, (WINDOW_SIZE[1] // 2)
                coords = self.get_circle_coords(center, radius)
                neuron = self.network_canvas.create_oval(coords, fill='white')
                self.neurons.append([(center, neuron)])
                continue
            
            # Create a circle for each neuron in the layer
            y_offset = (WINDOW_SIZE[1] - (2 * y_padding)) // (layer_size - 1)
            
            for j in range(layer_size):
                center = (i * x_offset) + x_padding, (j * y_offset) + y_padding 
                
                coords = self.get_circle_coords(center, radius)
                neuron = self.network_canvas.create_oval(coords, fill='white')
                layer.append((center, neuron))
                self.neuron_objects.add(neuron)
            
            self.neurons.append(layer)
        
    def network_display_setup(self, net):
        # Create new window for net display
        WINDOW_SIZE = 1920, 1080
        self.network_window = tk.Toplevel(self)
        self.network_window.geometry(f"{WINDOW_SIZE[0]}x{WINDOW_SIZE[1]}")

        # Create canvas for display
        self.network_canvas = tk.Canvas(self.network_window, width=WINDOW_SIZE[0], height=WINDOW_SIZE[1])
        self.network_canvas.pack()
        
        # Create the neurons as done above
        self.create_neurons(net)
        
        # Create lines
        self.create_network_lines(net)

        # Run it again to overwrite the lines
        self.create_neurons(net)

    def destroy_network_display(self):
        if not self.settings.show_network.get() and self.network_window:
            self.network_window.destroy()
            self.network_window = None

    def display_network(self, net):
        print(type(net))

        # Create toplevel if not already created
        if not self.network_window:
            self.network_display_setup(net)
            self.prev_net = net

        # Update net specific displays
        if self.prev_net != net:
            # Redraw weight lines
            for line in self.lines:
                self.network_canvas.delete(line)
            self.create_network_lines(net)

            # Redraw neurons
            for neuron in self.neuron_objects:
                self.network_canvas.delete(neuron)
            self.create_neurons(net)

        # Delete old text
        for text in self.neuron_text:
            self.network_canvas.delete(text)
        self.neuron_text = set() # reset set
        
        # Create new text (neuron values)
        for display_layer, network_layer in zip(self.neurons, net.layers):
            for neuron, val in zip(display_layer, network_layer):
                coords, circle = neuron

                # neuron fill colors
                # color = int(val * 255)
                # color = "#%02x%02x%02x" % (color, color, color)
                # self.network_canvas.itemconfig(circle, fill=color)

                text = self.network_canvas.create_text(*coords, text=f"{val}")
                self.neuron_text.add(text)
        
        # set prev net so that display knows not to update 
        self.prev_net = net
        
        # For debug
        # print('---------------------------------------------------------')
        return None
    
    def run_toplevel(self):
        a_star = tk.Toplevel(self.app)
        

class Simulation(tk.Toplevel):

    def __init__(self, master: App, player, size, box_size, start, end, walls):
        tk.Toplevel.__init__(self, master)

        self.master = master

        self.cols, self.rows = size

        self.width = size[0] * box_size
        self.height = size[1] * box_size
        self.box_size = box_size

        self.canvas = tk.Canvas(self, width=self.width, height=self.height)
        self.canvas.grid(row=5, columnspan=2)

        self.create_board()
        self.init_funcs()
        self.init_vars()

        self.start = start[1]
        self.end = end[1]
        self.walls = walls

        self.settings = self.master.settings
        
        self.solver = Solver(self, self.cols, self.rows)
        self.player = player
    
    def init_funcs(self):
        self.coords_to_grid = self.master.coords_to_grid
        self.grid_to_box = self.master.grid_to_box
        self.coords_to_box = self.master.coords_to_box

    def init_vars(self):
        self.neighbors_explored = set()
        self.drawn_objects = set()

    def create_board(self):
        for x in range(self.cols):
            x = x * self.box_size
            self.canvas.create_line(x, 0, x, self.height)
            
        for y in range(self.rows):
            y = y * self.box_size
            self.canvas.create_line(0, y, self.width, y)
    
    def run_program(self):
        self.solver.create_grid()

        # Rewrites self.start/end with the 
        # (drawing, pos) format & actually draws it
        self.draw_start(self.start)
        self.draw_end(self.end)
        for wall in self.walls:
            self.draw_wall(wall)

        self.solver.find_path(self.start[1], self.end[1])

    def draw_start(self, grid_coords):
        color = 'red'
        box_coords = self.grid_to_box(*grid_coords)
        self.start = self.canvas.create_rectangle(box_coords, fill=color), grid_coords

    def draw_end(self, grid_coords):
        color = 'blue'
        box_coords = self.grid_to_box(*grid_coords)
        self.end = self.canvas.create_rectangle(box_coords, fill=color), grid_coords

    def draw_wall(self, grid_coords):
        color = 'purple'
        box_coords = self.grid_to_box(*grid_coords)
        self.canvas.create_rectangle(box_coords, fill=color)
        self.update()

    def draw_exploration(self, grid_pos):
        if not self.settings.show_steps.get():
            return

        color = 'yellow'
        if grid_pos != self.start[1]:
            coords = self.grid_to_box(*grid_pos)
            obj = self.canvas.create_rectangle(coords, fill=color)
            self.drawn_objects.add(obj)
        self.update()

    def draw_neighbors(self, neighbors):
        if not self.settings.show_steps.get():
            return

        color = 'lime'
        for grid_pos, _ in neighbors:
            coords = self.grid_to_box(*grid_pos)

            if grid_pos not in [*self.neighbors_explored, self.start[1], self.end[1]]:
                obj = self.canvas.create_rectangle(coords, fill=color)
                self.neighbors_explored.add(grid_pos)
                self.drawn_objects.add(obj)
        self.update()

    def draw_path(self, path):
        color = 'deep sky blue'
        for grid_pos in path:
            if grid_pos not in [self.start[1], self.end[1]]:
                coords = self.grid_to_box(*grid_pos)
                obj = self.canvas.create_rectangle(coords, fill=color)
                self.drawn_objects.add(obj)

class SettingsMenu(tk.Toplevel):

    def __init__(self, master: App):
        tk.Toplevel.__init__(self, master)

        self.master = master
        self.title('Settings')

    def init_UI(self):
        
            ### SHOWING STEPS ###
            self.show_steps = tk.BooleanVar()
            steps_box = tk.Checkbutton(self, text="Show Steps", var=self.show_steps)
            steps_box.grid(row=0, column=0)
            
            ### DISPLAYING NETWORK ###
            self.show_network = tk.BooleanVar()
            self.master.network_window = None
            network_box = tk.Checkbutton(self, text="Show Network", var=self.show_network, command=self.master.destroy_network_display)
            network_box.grid(row=1, column=0)
            
            ### DISPLAYING END OF GENERATION RESULTS ###
            self.show_end = tk.BooleanVar()
            show_end_box = tk.Checkbutton(self, text="Show End Results", var=self.show_end)
            show_end_box.grid(row=2, column=0)

            ### DISPLAYING END OF GENERATION VS A* PLAYER ###
            self.show_a_star = tk.BooleanVar()
            show_a_star = tk.Checkbutton(self, text="Compare With A*", var=self.show_a_star)
            show_a_star.grid(row=3, column=0)

            ### DELAYING PROGRAM ###
            delay_label = tk.Label(self, text="Delay:")
            delay_label.grid(row=4, column=0)

            self.sleep_time = tk.StringVar()
            self.sleep_time.set('0')
            delay_input = tk.Entry(self, textvariable=self.sleep_time)
            delay_input.grid(row=4, column=1)

class Node:
    DIAGONAL_LENGTH = 1.41421356237  # ~root 2
    ADJACENT_LENGTH = 1

    def __init__(self, position, GRID):

        self.grid = GRID
        self.position = position
        self.neighbors = {}
        self.neighbors_calculated = False

    def calculate_neighbors(self):

        x, y = self.position
        self.neighbors_calculated = True
        for new_x in [x-1, x, x+1]:
            for new_y in [y-1, y, y+1]:

                # If the node is itself: continue
                if new_x == x and new_y == y:
                    continue

                # If only one dir is shifted - must be adjacent
                length = Node.DIAGONAL_LENGTH
                if (new_x != x) != (new_y != y): # XOR
                    length = Node.ADJACENT_LENGTH

                # If the neighbor exists - add it to dict
                if (new_x, new_y) in self.grid.node_coords:
                    self.neighbors[(new_x, new_y)] = length

        return self.neighbors

    def calculate_distance(self, pos):
        x1, y1 = self.position
        x2, y2 = pos
        return math.dist((x1, y1), (x2, y2))

    # For heap comparison
    def __lt__(self, other):
        return False


class Solver:

    def __init__(self, app: App, width: int, height: int):

        self.app = app
        self.width = width
        self.height = height

    def create_grid(self):

        self.node_coords = set()
        for x in range(self.width):
            for y in range(self.height):

                # Check whether the position is blocked or not
                if (x, y) not in self.app.walls:
                    self.node_coords.add((x, y))
                    
        self.nodes = set()
        for pos in self.node_coords:
            self.nodes.add(Node(pos, self))
        
    def heuristic(self, pos):
        return math.dist(pos, self.end)

    def find_path(self, start, end):
        t = time.perf_counter()

        self.end = end # Make it an instance var for heuristic

        WEIGHT = {pos: math.inf for pos in self.node_coords}  # node weights
        WEIGHT[start] = 0  # Euclidean distance of path to node (0 for start)
        NODE_SCORE = {}  # Total score for each node (weight + heuristic[default: dist to end])
        NODES = []  # Priority queue of nodes
        NODE_HASH = {start}  # Set of all nodes in the queue

        PREV_NODE = {}  # Dict of nodes and which node they came from - used to generate path
        # ^^^ maybe just add this to the node class (self.prev or something)?
        path = []  # Used at end - represents each step it needs to take

        # Check if start & end are legitimate points
        start_check = start in self.node_coords
        end_check = end in self.node_coords

        if DEBUG:
            print('start neighbors: ', Node(start, self).calculate_neighbors())
            print('end neighbors: ', Node(end, self).calculate_neighbors())

        if not start or not end:
            print('One of the point given does not exist!')
            print('START:', start_check)
            print('END:', end_check)
            return
        if DEBUG:
            print('done check')


        # Setup vars for heappush
        iteration = 0
        NODE_SCORE[start] = 0

        # Push start node 
        # Ranked first by the score, then as a backup use iter num
        # Include the Node in the heap so when we pop(), we know which node it is
        heapq.heappush(NODES, (NODE_SCORE[start], iteration, Node(start, self)))

        # Run until queue is empty (explored all possibilities)
        while NODE_HASH:
            iteration += 1

            t2 = time.perf_counter()

            pop = heapq.heappop(NODES)
            min_node = pop[2]
            NODE_HASH.remove(min_node.position)

            if min_node.position == end:
                if DEBUG:
                    print("FOUND")
                break

            items = min_node.calculate_neighbors().items(
            ) if not min_node.neighbors_calculated else min_node.neighbors.items()
            
            self.app.draw_neighbors(items)

            for child, weight in items:
                # Create new path if path is shorter
                if weight + WEIGHT[min_node.position] < WEIGHT[child]:

                    WEIGHT[child] = weight + WEIGHT[min_node.position]
                    NODE_SCORE[child] = self.heuristic(child) + WEIGHT[child]
                    PREV_NODE[child] = min_node

                    # Add node if it doesn't already exist
                    # (even if it's already been explored, must re-explore if it has a shorter path)
                    if child not in NODE_HASH:
                        self.app.draw_exploration(min_node.position)
                        
                        heapq.heappush(
                            NODES, (NODE_SCORE[child], iteration, Node(child, self)))
                        NODE_HASH.add(child)

            # Get set delay
            sleep_time = float(self.app.settings.sleep_time.get())
            sleep_time -= float(time.perf_counter() - t2)
            
            # make sure its not below 0
            sleep_time = max(sleep_time, 0)

            # Sleep program for delay
            time.sleep(sleep_time)
            

        try:
            self.iterations = iteration
            node = end
            while True:
                path.append(node)
                node = PREV_NODE[node].position
        except KeyError:
            path_exists = True
            if node != start:
                path_exists = False
                print('There is no path')
                print('done printing results')
                return [start]
            path.reverse()

            if DEBUG:
                print('length:', NODE_SCORE[end])
                print('calculation done: ', iteration, time.perf_counter() - t)
                print('COUNT:', COUNT)
                print('COUNT2:', COUNT2)

            
            runtime = time.perf_counter() - t
            self.app.player.time += runtime

            self.app.draw_path(path)
            return path


class Player:
    NUM_LAYERS = 4
    LAYER_SIZES = 5, 8, 8, 1

    MUTATION_RATE = 0.1
    STEP_SIZE = 0.5

    def __init__(self, pop_size, app: App):
        self.pop_size = pop_size
        self.app = app
        self.settings = self.app.settings

        self.create_population()

    def run(self):
        self.fitness()
        winners = self.selection(self.nets)
        self.crossover(winners)

    def create_population(self):
        self.population = []
        for _ in range(self.pop_size):
            net = Network(Player.NUM_LAYERS, Player.LAYER_SIZES, self.app)
            net.create_weights()
            self.population.append(net)
    
    def fitness(self):
        
        middle_x = self.app.width // 2
        middle_y = self.app.height // 2

        # Define start at the top left
        self.start = randint(0, middle_x), randint(0, middle_y)

        # Define end at the bottom left
        self.end = randint(middle_x, self.app.width), randint(middle_y, self.app.height)

        def setup():
            self.WALLS = set()
            self.DRAWN_WALLS = set()
            
            ### Create diagonal wall ###
            for x in range(self.app.cols):
                for y in range(self.app.rows):
                    avg = (self.app.cols + self.app.rows) // 2
                    avg -= 1 # subtract one because grid starts at 0

                    # Check if its within one of the diagonal
                    if (avg - 1) <= (x + y) <= (avg + 1):
                        # Skip edges
                        if x in [0, self.app.cols - 1] or y in [0, self.app.rows - 1]:
                            continue
                        new_x = x * self.app.box_size
                        new_y = y * self.app.box_size 
                        wall = self.app.draw_wall(new_x, new_y)
                        self.WALLS.add((x, y))
                        self.DRAWN_WALLS.add(wall[0])                       
        
        def pos_in_wall():
            # Check if start is in a wall
            while self.app.coords_to_grid(*self.start) in self.WALLS:
                self.start = randrange(0, middle_x), randrange(0, middle_y)
            self.app.draw_start(*self.start)

            # Check if end is in a wall
            while self.app.coords_to_grid(*self.end) in self.WALLS:
                self.end = randrange(middle_x, self.app.width), randrange(middle_y, self.app.height)
            self.app.draw_end(*self.end)
            
            self.app.update()

        nets = []
        setup()
        self.time = 0
        self.t = 0
        for net in self.population:
            if self.settings.show_steps.get():
                setup()
            else:
                self.app.walls = self.WALLS
                self.app.drawn_walls = self.DRAWN_WALLS
                

            pos_in_wall()
            self.simulate(net)

            score = self.app.solver.iterations
            net.score = score
            nets.append(net)

        print('FITNESS2:', self.t)

        self.nets = nets
        nets.sort()

        ###  SHOW BEST  ###
        scores = [net.score for net in nets]
        print('HIGHEST VAL:', nets[-1].score)
        print('AVG VAL:', sum(scores) // len(scores))
        print('LOWEST VAL:', nets[0].score)
        print('ALGO RUNTIME:', self.time)

        settings = self.settings
        app = self.app
        if not settings.show_end.get():
            return
        
        prev_vals = settings.show_steps.get(), settings.sleep_time.get()
        settings.show_steps.set(True)
        settings.sleep_time.set('0.05')

        setup()
        pos_in_wall()
        app.solver.heuristic = nets[0].run_network
        sim = None
        if settings.show_a_star.get():
            sim = Simulation(app, self, app.size, app.box_size, app.start, app.end, app.walls)
            # Process(target=sim.run_program).start()
            # Process(target=app.run_program).start()
            sim.run_program()
            app.run_program()
        else:
            app.run_program()
        
        print(self.app.solver.iterations)
        print(nets[0].score)
        self.app.update()
        time.sleep(3)

        ### RESET UI ###
        settings.show_steps.set(prev_vals[0])
        settings.sleep_time.set(prev_vals[1])
        if sim:
            sim.destroy()

    def simulate(self, net):
        
        self.app.solver.heuristic = net.run_network
        self.app.run_program()
        self.app.reset_program()

    def test_run(self, use_net):
        for net in self.population:
            if use_net:
                self.app.solver.heuristic = net.run_network
            print('running net')
            self.app.run_program()
            print('done net')
            self.app.reset_program()

    def tournament(self, nets, probability):
        nets.sort()
        p = probability
        for i, net in enumerate(nets):
            prob_i = p * ((1 - p) ** i)
            selection = net if random.uniform(0, 1) < prob_i else None
            if selection:
                break
        else:
            selection = nets[-1]
        
        return selection

    def selection(self, pop):
        p = .95
        winners = []
        for _ in range(len(pop) // 2):
            selected = random.choices(pop, k=3)
            winner = self.tournament(selected, p)
            pop.remove(winner)
            winners.append(winner)
        
        return winners
    
    def crossover(self, winners):
        args = Player.NUM_LAYERS, Player.LAYER_SIZES, self.app

        offspring = []
        for _ in range(len(winners) // 2):
            # Pick 2 random parents
            net1, net2 = random.choices(winners, k=2)

            # Create 2 new blank children
            net_a = Network(*args)
            net_b = Network(*args)
            a_weights = []
            b_weights = []

            # Loop through each layer of weights
            for layer1, layer2 in zip(net1.weights, net2.weights):
                # Create a blank layer of weights
                a_layer = []
                b_layer = []
                for w1, w2 in zip(layer1, layer2):
                    # Split list of weights at random separator
                    # For each parent. Mix them for offspring and mutate 
                    separator = random.randrange(0, len(w1))
                    a, b = [], []
                    for item in w1[:separator]:
                        a.append(item)
                    for item in w2[separator:]:
                        a.append(item)
                    
                    for item in w2[:separator]:
                        b.append(item)
                    for item in w1[separator:]:
                        b.append(item)
                    
                    a = self.mutate(np.array(a))
                    b = self.mutate(np.array(b))

                    # print(a)
                    # print(b)

                    # Append new weights to the layer
                    a_layer.append(a)
                    b_layer.append(b)
                
                # Append the layer to the collection of weights
                a_weights.append(np.array(a_layer))
                b_weights.append(np.array(b_layer))
            
            # Set network weights and append to offspring to finalize
            net_a.weights = a_weights
            net_b.weights = b_weights
            offspring.append(net_a)
            offspring.append(net_b)

        self.population = winners + offspring

    @staticmethod
    def check_iterable(item):
        try:
            iter(item)
            is_iter = True
        except TypeError:
            is_iter = False
        return is_iter

    def mutate(self, weights):
        # Loop through items in weights
        for i, item in enumerate(weights):
            # If it's a nested list: 
            # Pass list back into the function
            if self.check_iterable(item):
                weights[i] = self.mutation(weights)
                continue
            
            # Mutate based on mutation chance
            if random.uniform(0, 1) < Player.MUTATION_RATE:
                # Get new mutated value
                range = -Player.STEP_SIZE, Player.STEP_SIZE
                offset = random.uniform(*range)
                new_val = round((item + offset), 3)

                # Keep vals between -1 and 1
                new_val = max(-1, new_val)
                new_val = min(1, new_val)

                # Update weight
                weights[i] = new_val

        return weights


class Network:

    def __init__(self, num_layers: int, layer_sizes: list[int], app):
        if len(layer_sizes) != num_layers or num_layers < 3:
            self.die()

        self.num_layers = num_layers
        self.layer_sizes = layer_sizes
        self.input_layer = None

        self.score = 0

        self.app = app

    def create_weights(self):

        self.weights = []
        for i in range(1, self.num_layers):
            size = self.layer_sizes[i], self.layer_sizes[i-1]
            weights = np.random.uniform(-1, 1, size).round(3)
            self.weights.append(weights)
        
        # self.weights = np.array(self.weights)


    def create_input(self, pos):
        app = self.app
        
        ### Fuck me this code is shit please fix it ###
        x, y = pos
        dist_to_end = math.dist(pos, app.end[1])
        dist_to_end = round(dist_to_end, 3)

        # Dist to closest wall on the right
        new_x = x + 1
        while (new_x, y) in app.solver.node_coords:
            new_x += 1
        wall_right = math.dist((x, y), (new_x, y))

        # Dist to closest wall on the left
        new_x = x - 1
        while (new_x, y) in app.solver.node_coords:
            new_x -= 1
        wall_left = math.dist((x, y), (new_x, y))

        # Dist to closest wall above
        new_y = y + 1
        while (x, new_y) in app.solver.node_coords:
            new_y += 1
        wall_top = math.dist((x, y), (x, new_y))

        # Dist to closest wall below
        new_y = y - 1
        while (x, new_y) in app.solver.node_coords:
            new_y -= 1
        wall_bottom = math.dist((x, y), (x, new_y))

        self.input_layer = np.array([dist_to_end, wall_top, wall_bottom, wall_right, wall_left])

    # def create_input(self, pos):
    #     app = self.app
        
    #     dist_to_end = math.dist(pos, app.end[1])
    #     dist_to_end = round(dist_to_end, 3)

    #     # Dist to closest wall
    #     nearest_wall = min(self.app.walls, key=lambda wall: math.dist(pos, (wall[0], wall[1])))
    #     dist_to_wall = round(math.dist(pos, nearest_wall), 3)

    #     self.input_layer = np.array([dist_to_end, dist_to_wall])

    def create_layers(self):
        
        self.layers = [self.input_layer]

        layer = self.input_layer
        for weights in self.weights:
            layer = np.dot(weights, layer).round(3)
            self.layers.append(layer)

    def run_network(self, pos):
        t3 = time.perf_counter()
        self.create_input(pos)
        self.create_layers()
        self.app.player.t += time.perf_counter() - t3

        if self.app.settings.show_network.get():
            self.app.display_network(self)
        
        return self.layers[-1]

    def die(self):
        print('give me proper inputs *skullemoji*')
        quit()
    
    def __lt__(self, other):
        try:
            return self.score < other.score
        except AttributeError:
            return False

class Empty:

    def __init__(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()

    width, height = 25, 25
    box_size = 30
    app = App(root, (width, height), box_size, True)

    app.pack(side="top", fill="both", expand=True)
    root.mainloop()
