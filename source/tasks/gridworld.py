import numpy as np
import random

from tasks.task import Task


class Shapes(Task):
    """
    A discretized version of the gridworld environment introduced in [1]. Here, an agent learns to 
    collect shapes with positive reward, while avoid those with negative reward, and then travel to a fixed goal.
    The gridworld is split into four rooms separated by walls with passage-ways.
    
    References
    ----------
    [1] Barreto, André, et al. "Successor Features for Transfer in Reinforcement Learning." NIPS. 2017.
    """

    LEFT, UP, RIGHT, DOWN = 0, 1, 2, 3
 
    def __init__(self, maze, shape_rewards):
        """
        Creates a new instance of the shapes environment.
        
        Parameters
        ----------
        maze : np.ndarray
            an array of string values representing the type of each cell in the environment:
                G indicates a goal state (terminal state)
                _ indicates an initial state (there can be multiple, and one is selected at random
                    at the start of each episode)
                X indicates a barrier 
                0, 1, .... 9 indicates the type of shape to be placed in the corresponding cell
                entries containing other characters are treated as regular empty cells
        shape_rewards : dict
            a dictionary mapping the type of shape (0, 1, ... ) to a corresponding reward to provide
            to the agent for collecting an object of that type
        """
        self.height, self.width = maze.shape
        self.maze = maze
        self.shape_rewards = shape_rewards
        shape_types = sorted(list(shape_rewards.keys()))
        self.all_shapes = dict(zip(shape_types, range(len(shape_types))))
        
        self.goal = None
        self.initial = []
        self.occupied = set()
        self.shape_ids = dict()
        for c in range(self.width):
            for r in range(self.height):
                if maze[r, c] == 'G':
                    self.goal = (r, c)
                elif maze[r, c] == '_':
                    self.initial.append((r, c))
                elif maze[r, c] == 'X':
                    self.occupied.add((r, c))
                elif maze[r, c] in {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9'}:
                    self.shape_ids[(r, c)] = len(self.shape_ids)
        
    def clone(self):
        return Shapes(self.maze, self.shape_rewards)

    def initialize(self):
        self.state = (random.choice(self.initial), tuple(0 for _ in range(len(self.shape_ids))))
        return self.state
    
    def action_count(self):
        return 4

    def transition(self, action): 
        (row, col), collected = self.state
        
        # perform the movement
        if action == Shapes.LEFT: 
            col -= 1
        elif action == Shapes.UP: 
            row -= 1
        elif action == Shapes.RIGHT: 
            col += 1
        elif action == Shapes.DOWN: 
            row += 1
        else:
            raise Exception('bad action {}'.format(action))
        
        # out of bounds, cannot move
        if col < 0 or col >= self.width or row < 0 or row >= self.height:
            return self.state, 0., False

        # into a blocked cell, cannot move
        s1 = (row, col)
        if s1 in self.occupied:
            return self.state, 0., False
        
        # can now move
        self.state = (s1, collected)
        
        # into a goal cell
        if s1 == self.goal:
            return self.state, 1., True
        
        # into a shape cell
        if s1 in self.shape_ids:
            shape_id = self.shape_ids[s1]
            if collected[shape_id] == 1:
                
                # already collected this flag
                return self.state, 0., False
            else:
                
                # collect the new flag
                collected = list(collected)
                collected[shape_id] = 1
                collected = tuple(collected)
                self.state = (s1, collected)
                reward = self.shape_rewards[self.maze[row, col]]
                return self.state, reward, False
        
        # into an empty cell
        return self.state, 0., False

    # ===========================================================================
    # STATE ENCODING FOR DEEP LEARNING
    # ===========================================================================
    def encode(self, state):
        (y, x), coll = state
        n_state = self.width + self.height
        result = np.zeros((n_state + len(coll),))
        result[y] = 1
        result[self.height + x] = 1
        result[n_state:] = np.array(coll)
        result = result.reshape((1, -1))
        return result
    
    def encode_dim(self):
        return self.width + self.height + len(self.shape_ids)
        
    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def features(self, state, action, next_state):
        s1, _ = next_state
        _, collected = state
        nc = len(self.all_shapes)
        phi = np.zeros((nc + 1,))
        if s1 in self.shape_ids:
            if collected[self.shape_ids[s1]] != 1:
                y, x = s1
                shape_index = self.all_shapes[self.maze[y, x]]
                phi[shape_index] = 1.
        elif s1 == self.goal:
            phi[nc] = 1.
        return phi
    
    def feature_dim(self):
        return len(self.all_shapes) + 1
    
    def get_w(self):
        ns = len(self.all_shapes)
        w = np.zeros((ns + 1, 1))
        for shape, shape_index in self.all_shapes.items():
            w[shape_index, 0] = self.shape_rewards[shape]
        w[ns, 0] = 1.
        return w
            
