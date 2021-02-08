from itertools import tee
import numpy as np
import pygame

from tasks.task import Task
import math

__copyright__ = "Copyright 2013, RLPy http://acl.mit.edu/RLPy"
__credits__ = ["Alborz Geramifard", "Robert H. Klein", "Christoph Dann",
               "William Dabney", "Jonathan P. How"]
__license__ = "BSD 3-Clause"
__author__ = ["Pierre-Luc Bacon",  # author of the original version
              "Austin Hays"]  # adapted for RLPy and TKinter


class Pinball(Task):
    
    """ This class is a self-contained model of the pinball
    domain for reinforcement learning.
    It can be used either over RL-Glue through the :class:`PinballRLGlue`
    adapter or interactively with :class:`PinballView`.
    """

    ACC_X = 0
    ACC_Y = 1
    DEC_X = 2
    DEC_Y = 3
    ACC_NONE = 4
            
    def __init__(self, config, targets, active_target):
        self.action_effects = {self.ACC_X:(1, 0), self.ACC_Y:(0, 1),
                               self.DEC_X:(-1, 0), self.DEC_Y:(0, -1), self.ACC_NONE:(0, 0)}

        self.config = config
        self.targets = targets
        self.active_target = active_target
        self.target_pos = targets[active_target]
        
        # Set up the environment according to the configuration
        self.obstacles = []
        self.target_rad = 0.04
        
        with open(config) as fp:
            for line in fp.readlines():
                tokens = line.strip().split()
                if not len(tokens):
                    continue
                elif tokens[0] == 'polygon':
                    self.obstacles.append(PinballObstacle(list(zip(*[iter(map(float, tokens[1:]))] * 2)), False))
                elif tokens[0] == 'start':
                    self.start_pos = list(zip(*[iter(map(float, tokens[1:]))] * 2))[0]
                elif tokens[0] == 'ball':
                    self.ball_rad = float(tokens[1])
            
    def clone(self):
        return Pinball(self.config, self.targets, self.active_target)
 
    def initialize(self):
        self.ball = BallModel(np.array(self.start_pos), self.ball_rad)
        return self._get_state()
    
    def action_count(self):
        return 5
    
    def transition(self, action):
         
        for i in range(20):
            if i == 0:
                self.ball.add_impulse(*self.action_effects[action])
            self.ball.step()

            # Detect collisions
            ncollision = 0
            dxdy = np.array([0, 0])

            for obs in self.obstacles:
                if obs.collision(self.ball):
                    dxdy = dxdy + obs.collision_effect(self.ball)
                    ncollision += 1
            if ncollision == 1:
                self.ball.xdot = dxdy[0]
                self.ball.ydot = dxdy[1]
                if i == 19:
                    self.ball.step()
            elif ncollision > 1:
                self.ball.xdot = -self.ball.xdot
                self.ball.ydot = -self.ball.ydot
            
        self.ball.add_drag()
        self._check_bounds()
        
        # reward
        d = self._distance(self.ball.position, self.target_pos)
        reward = math.exp(-4. * d)
        
        return self._get_state(), reward, False
    
    def _get_state(self):
        return (self.ball.position[0], self.ball.position[1], self.ball.xdot, self.ball.ydot)
    
    def _distance(self, p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))
        
    def _check_bounds(self):
        if self.ball.position[0] > 1.0:
            self.ball.position[0] = 0.95
        if self.ball.position[0] < 0.0:
            self.ball.position[0] = 0.05
        if self.ball.position[1] > 1.0:
            self.ball.position[1] = 0.95
        if self.ball.position[1] < 0.0:
            self.ball.position[1] = 0.05
    
    # ===========================================================================
    # STATE ENCODING FOR DEEP LEARNING
    # ===========================================================================
    def encode(self, state):
        return np.reshape(state, (1, -1))
    
    def encode_dim(self):
        return 4
    
    # ===========================================================================
    # SUCCESSOR FEATURES
    # ===========================================================================
    def features(self, state, action, next_state):
        phi = np.zeros((len(self.targets),))
        for i, target in enumerate(self.targets):
            d = self._distance(next_state[:2], target)
            phi[i] = math.exp(-4. * d)
        return phi
    
    def feature_dim(self):
        return len(self.targets)
    
    def get_w(self):
        w = np.zeros((len(self.targets), 1))
        w[self.active_target, 0] = 1.
        return w
    
 
class PinballView:
    """ This class displays a :class:`PinballModel`
    This class is used in conjunction with the :func:`run_pinballview`
    function, acting as a *controller*.
    We use `pygame <http://www.pygame.org/>` to draw the environment.
    """
 
    def __init__(self, model):
        self.screen = pygame.display.set_mode([600, 600])
        self.model = model
 
        self.DARK_GRAY = [48, 48, 48]
        self.LIGHT_GRAY = [128, 126, 128]
        self.BALL_COLOR = [0, 0, 255]
        self.TARGET_COLOR = [255, 0, 0]
        self.BLUE = [214, 236, 239]
        self.KEY_COLOR = [128, 255, 128]
 
        # Draw the background
        pygame.init()
        
        self.background_surface = pygame.Surface(self.screen.get_size())
        self.background_surface.fill(self.LIGHT_GRAY)
        for obs in model.obstacles:
            pygame.draw.polygon(self.background_surface, self.DARK_GRAY,
                                list(map(self._to_pixels, obs.points)), 0)
            
        pygame.draw.circle(
            self.background_surface, self.TARGET_COLOR, self._to_pixels(self.model.target_pos),
            int(self.model.ball_rad * self.screen.get_width()))

    def _to_pixels(self, pt):
        return (int(pt[0] * self.screen.get_width()), int(pt[1] * self.screen.get_height()))
 
    def update(self):
        pygame.event.pump()       
        self.screen.blit(self.background_surface, (0, 0))
        pygame.draw.circle(self.screen, self.BALL_COLOR,
                           self._to_pixels(self.model.ball.position),
                           int(self.model.ball.radius * self.screen.get_width()))
        pygame.display.update()
        

class BallModel:
    """ This class maintains the state of the ball
    in the pinball domain. It takes care of moving
    it according to the current velocity and drag coefficient.
    """
    DRAG = 0.995

    def __init__(self, start_position, radius):
        self.position = start_position
        self.radius = radius
        self.xdot = 0.0
        self.ydot = 0.0

    def add_impulse(self, delta_xdot, delta_ydot):
        self.xdot += delta_xdot / 5.0
        self.ydot += delta_ydot / 5.0
        self._clip(self.xdot)
        self._clip(self.ydot)

    def add_drag(self):
        self.xdot *= self.DRAG
        self.ydot *= self.DRAG

    def step(self):
        self.position[0] += self.xdot * self.radius / 20.0
        self.position[1] += self.ydot * self.radius / 20.0

    def _clip(self, val, low=-1, high=1):
        if val > high:
            val = high
        if val < low:
            val = low
        return val


class PinballObstacle:
    """ This class represents a single polygon obstacle in the
    pinball domain and detects when a :class:`BallModel` hits it.
    When a collision is detected, it also provides a way to
    compute the appropriate effect to apply on the ball.
    """

    def __init__(self, points, istrap):
        self.points = points
        self.min_x = min(self.points, key=lambda pt: pt[0])[0]
        self.max_x = max(self.points, key=lambda pt: pt[0])[0]
        self.min_y = min(self.points, key=lambda pt: pt[1])[1]
        self.max_y = max(self.points, key=lambda pt: pt[1])[1]

        self._double_collision = False
        self._intercept = None
        self.istrap = istrap

    def collision(self, ball):
        self._double_collision = False

        if ball.position[0] - ball.radius > self.max_x:
            return False
        if ball.position[0] + ball.radius < self.min_x:
            return False
        if ball.position[1] - ball.radius > self.max_y:
            return False
        if ball.position[1] + ball.radius < self.min_y:
            return False

        a, b = tee(np.vstack([np.array(self.points), self.points[0]]))
        next(b, None)
        intercept_found = False
        for pt_pair in zip(a, b):
            if self._intercept_edge(pt_pair, ball):
                if intercept_found:
                    # Ball has hit a corner
                    self._intercept = self._select_edge(pt_pair, self._intercept, ball)
                    self._double_collision = True
                else:
                    self._intercept = pt_pair
                    intercept_found = True

        return intercept_found

    def collision_effect(self, ball):
        if self._double_collision:
            return [-ball.xdot, -ball.ydot]

        # Normalize direction
        obstacle_vector = self._intercept[1] - self._intercept[0]
        if obstacle_vector[0] < 0:
            obstacle_vector = self._intercept[0] - self._intercept[1]

        velocity_vector = np.array([ball.xdot, ball.ydot])
        theta = self._angle(velocity_vector, obstacle_vector) - np.pi
        if theta < 0:
            theta += 2 * np.pi

        intercept_theta = self._angle([-1, 0], obstacle_vector)
        theta += intercept_theta

        if theta > 2 * np.pi:
            theta -= 2 * np.pi

        velocity = np.linalg.norm([ball.xdot, ball.ydot])

        return [velocity * np.cos(theta), velocity * np.sin(theta)]

    def _select_edge(self, intersect1, intersect2, ball):
        velocity = np.array([ball.xdot, ball.ydot])
        obstacle_vector1 = intersect1[1] - intersect1[0]
        obstacle_vector2 = intersect2[1] - intersect2[0]

        angle1 = self._angle(velocity, obstacle_vector1)
        if angle1 > np.pi:
            angle1 -= np.pi

        angle2 = self._angle(velocity, obstacle_vector2)
        if angle1 > np.pi:
            angle2 -= np.pi

        if np.abs(angle1 - (np.pi / 2.0)) < np.abs(angle2 - (np.pi / 2.0)):
            return intersect1
        return intersect2

    def _angle(self, v1, v2):
        angle_diff = np.arctan2(v1[0], v1[1]) - np.arctan2(v2[0], v2[1])
        if angle_diff < 0:
            angle_diff += 2 * np.pi
        return angle_diff

    def _intercept_edge(self, pt_pair, ball):
    
        # Find the projection on an edge
        obstacle_edge = pt_pair[1] - pt_pair[0]
        difference = ball.position - pt_pair[0]

        scalar_proj = difference.dot(obstacle_edge) / obstacle_edge.dot(obstacle_edge)
        if scalar_proj > 1.0:
            scalar_proj = 1.0
        elif scalar_proj < 0.0:
            scalar_proj = 0.0

        # Compute the distance to the closest point
        closest_pt = pt_pair[0] + obstacle_edge * scalar_proj
        obstacle_to_ball = ball.position - closest_pt
        distance = obstacle_to_ball.dot(obstacle_to_ball)

        if distance <= ball.radius * ball.radius:
            # A collision only if the ball is not already moving away
            velocity = np.array([ball.xdot, ball.ydot])
            ball_to_obstacle = closest_pt - ball.position

            angle = self._angle(ball_to_obstacle, velocity)
            if angle > np.pi:
                angle = 2 * np.pi - angle

            if angle > np.pi / 1.99:
                return False

            return True
        else:
            return False

