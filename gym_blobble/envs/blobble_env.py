import io
from PIL import Image
import gym
import numpy as np
import plotly.graph_objects as go
from copy import deepcopy

from gym import spaces, error, utils
from gym.utils import seeding

TASTE = True
SMELL = True


class BlobbleEnv(gym.Env):
    """
    Description:


    Observations:

        Type: Box(3)
        Num     Observation                             Min             Max
        0       Blobble Northerly Location              MIN_LOC(-10)    MAX_LOC (10)
        1       Blobble Easterly location               MIN_LOC(-10)    MAX_LOC (10)
        2       Health                                  0               MAX_HEALTH (10)
        3       Taste (Food nutritional value at loc)   -5              +5     (set to zero if no food)
        The smell is the average of the four closest locations in the relevant direction
        4       Food smell N                            -5              +5
        5       Food smell S                            -5              +5
        6       Food smell E                            -5              +5
        7       Food smell W                            -5              +5


    Actions:

        Type: Discrete(10)
        Num     Action
        0       eat
        1       move N
        2       move S
        3       move E
        4       move W
        5       eat, move N
        6       eat, move S
        7       eat, move E
        8       eat, move W

    An Episode Ends When:

        - The blobble's health goes below zero

    Rewards:

        - 0.5 for every step
        - extra 0.5 for being over average health (encourage healthy eating)

    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self):

        super(BlobbleEnv, self).__init__()

        self.action_space = spaces.Discrete(9)

        self._SMELL = SMELL
        self._TASTE = TASTE

        self._MAX_HEALTH = 10
        self._MAX_LOC = 10
        self._MIN_LOC = -self._MAX_LOC
        high_values = np.array([self._MAX_LOC,
                                self._MAX_LOC,
                                self._MAX_HEALTH,
                                5,
                                5,
                                5,
                                5,
                                5],
                               dtype=np.float32)
        low_values = np.array([-self._MAX_LOC,
                               -self._MAX_LOC,
                               0,
                               -5,
                               -5,
                               -5,
                               -5,
                               -5],
                              dtype=np.float32)
        self.observation_space = spaces.Box(low_values, high_values, dtype=np.float32)

        # Initialise aspects of the envs that never change
        max_bubble_size = 50
        self._SIZE_REF = 2. * self._MAX_HEALTH / (max_bubble_size ** 2)  # For scaling blobbles and food

        self._MAX_FOOD = 100
        self._HEALTH_COLOURS = ['rgb(243, 224, 247)',
                                'rgb(228, 199, 241)',
                                'rgb(209, 175, 232)',
                                'rgb(185, 152, 221)',
                                'rgb(159, 130, 206)',
                                'rgb(130, 109, 186)',
                                'rgb(99, 85, 159)']
        self._episode = 0

        # Create New Blobble
        # This is the starting state for the blobble every reset [northerly, easterly, health]
        self._initial_state = (0, 0, 5)
        # Allocate the Blobble State array - this is returned as the observation following each step
        self._blobble_state = np.zeros(8, dtype=np.float)

        self.seed()
        # Reset the env to its start position
        self.reset()
        self._best_episode = 0

    def reset(self):
        """
        Resets the Blobble to its initial state

        :return:
        observation
        """

        self._blobble_state[0:3] = self._initial_state
        self._blobble_state[3:8] = np.array((0., 0., 0., 0., 0.), dtype=float)

        # Reset Blobble food locations
        self._food = self.np_random.randint(low=-5,
                                            high=5,
                                            size=(self._MAX_LOC - self._MIN_LOC + 1, self._MAX_LOC - self._MIN_LOC + 1))
        self._hunger = 0
        self._rewards_so_far = 0

        return self._blobble_state

    def seed(self, seed=None):
        """
        Provided to enable deterministic behaviour

        :param seed:
        :return:
        seed
        """
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def _eat(self):
        """
        Internal (private) method called when the blobble eats
        :return:
        """

        # Eat any food at the current location
        nutrition = self._blobble_state[3]
        if nutrition != 0:
            # print('yum')
            self._hunger = 0
        else:
            self._hunger += 1
        # Add health but clip it between 0 and max_health
        self._blobble_state[2] = max(0, min(self._blobble_state[2] + nutrition, self._MAX_HEALTH))
        # Delete the food
        row, column = self._position_in_food_array(self._blobble_state[0], self._blobble_state[1])
        self._food[row, column] = 0

        return

    def _position_in_food_array(self, northerly, easterly):
        # The row in self._food corresponding to the blobble's northerly position
        food_array_row = -(int(northerly) + self._MIN_LOC)
        # The column in self._food corresponding to the blobble's easterly location
        food_array_column = int(easterly) - self._MIN_LOC
        return food_array_row, food_array_column

    def _clip(self, value, minimum_value, maximum_value):
        return min(maximum_value, max(minimum_value, value))

    def _sniff(self, sniff_row, sniff_column):

        # If we are at the edge of the land, nothing to smell
        if (sniff_row < 0) or (sniff_column < 0):
            return 0
        if (sniff_row > self._MAX_LOC-self._MIN_LOC) or (sniff_column > self._MAX_LOC-self._MIN_LOC):
            return 0
        # Else return the food's value
        return self._food[sniff_row, sniff_column]

    def _smell(self):

        if self._SMELL is False:
            return np.zeros(4, dtype=np.float)  # Non-smelling blobble

        food_array_row, food_array_column = self._position_in_food_array(self._blobble_state[0],
                                                                         self._blobble_state[1])

        # Take a sniff north, south, east and west

        # Sniff North
        # [ ., ., ., ., .],
        # [ ., o, o, o, .],
        # [ ., ., B, ., .]
        smell_north = self._blobble_state[4] = np.average(np.array([
                                                          self._sniff(food_array_row - 1, food_array_column - 1),
                                                          self._sniff(food_array_row - 1, food_array_column),
                                                          self._sniff(food_array_row - 1, food_array_column + 1)]
                                                            ))

        # Sniff South
        # [ ., ., B, ., .],
        # [ ., o, o, o, .],
        # [ ., ., ., ., .]
        smell_south = self._blobble_state[5] = np.average(np.array([
                                                          self._sniff(food_array_row + 1, food_array_column - 1),
                                                          self._sniff(food_array_row + 1, food_array_column),
                                                          self._sniff(food_array_row + 1, food_array_column + 1)]
                                                                   ))
        # Sniff East
        # [ ., ., ., o, .],
        # [ ., ., B, o, .],
        # [ ., ., ., o, .]
        smell_east = self._blobble_state[6] = np.average(np.array([
                                                         self._sniff(food_array_row - 1, food_array_column + 1),
                                                         self._sniff(food_array_row, food_array_column + 1),
                                                         self._sniff(food_array_row + 1, food_array_column + 1)]
                                                                  ))
        # Sniff West
        # [ ., o, ., ., .],
        # [ ., o, B, ., .],
        # [ ., o, ., ., .]
        smell_west = self._blobble_state[7] = np.average(np.array([
                                                         self._sniff(food_array_row - 1, food_array_column - 1),
                                                         self._sniff(food_array_row, food_array_column - 1),
                                                         self._sniff(food_array_row + 1, food_array_column - 1)]
                                                                  ))

        smell = np.array((smell_north,
                          smell_south,
                          smell_east,
                          smell_west),
                         dtype=np.float32)

        return smell

    def _taste(self):

        if self._TASTE is False:
            return 0  # Non-tasting blobble

        food_array_row, food_array_column = self._position_in_food_array(self._blobble_state[0],
                                                                         self._blobble_state[1])
        return self._food[food_array_row, food_array_column]

    def step(self, action):
        """
        Perform an action

        :param action:
        Action to be performed
        :return:
        observation, reward, done, {}
        """

        if action > 4 or action == 0:  # Eat (if there is food)
            self._eat()
        else:
            self._hunger += 0.5

        if (action == 1) or (action == 5):  # Move North
            self._blobble_state[0] = min(self._blobble_state[0]+1, self._MAX_LOC)
        if (action == 2) or (action == 6):  # Move South
            self._blobble_state[0] = max(self._blobble_state[0]-1, self._MIN_LOC)
        if (action == 3) or (action == 7):  # Move East
            self._blobble_state[1] = min(self._blobble_state[1]+1, self._MAX_LOC)
        if (action == 4) or (action == 8):  # Move West
            self._blobble_state[1] = max(self._blobble_state[1]-1, self._MIN_LOC)

        self._blobble_state[3] = self._taste()
        self._blobble_state[4:8] = self._smell()

        # Set game as done if the health of the blobble has gone to zero
        done = bool(self._blobble_state[2] <= 0)

        # Allocate a reward
        reward = 0.0
        if not done:
            reward = 0.5
            if self._blobble_state[2] > self._MAX_HEALTH / 2:  # Encourage healthy blobble living
                reward = 1.0
            self._rewards_so_far = self._rewards_so_far + reward
        else:
            self._best_episode = max(self._rewards_so_far, self._best_episode)

        if self._hunger > 5:  # Reduce health if the blobble is getting hungry
            self._blobble_state[2] = max(0, self._blobble_state[2] - 1)

        return self._blobble_state, reward, done, {}

    def render(self, mode='rgb_array', close=False):
        """
        Renders the blobble env in a human readable format or in an RGB array.

        :param mode: 'human' or 'rgb_array'
        :param close:
        :return:
        """

        # Create the x and y location lists for rendering the food positions
        # Note that here we:
        #  - Select all the values which adhere to a specific criteria (such as self._food >3)
        #    This gives a list of values [[row, column]...] such as [[0, 5], [0, 7] ... [20, 15] ]
        #  - Transpose the resulting array.
        #    This gives a list of all the row values and a list of all the column values, such as:
        #    [[0, 0, ... 20]
        #     [5, 7, ... 15]]
        #  - shift the values by self.MIN_lOC to move the x, y coordinate space so that position [0, 0] appears
        #    in the centre of blobble world. Like this (if MIN_LOC = -10):
        #    [[-10, -10, ... 10]
        #     [-5,  -2,  ... 5]]
        super_food = np.argwhere(self._food > 3).T + self._MIN_LOC
        good_food = np.argwhere((self._food > 0) & (self._food <= 3)).T + self._MIN_LOC
        unhealthy_food = np.argwhere((self._food < 0) & (self._food >= -3)).T + self._MIN_LOC
        bad_weed = np.argwhere(self._food < -3).T + self._MIN_LOC

        # Depict the food locations
        fig = go.Figure(data=go.Scatter(
            x=super_food[1],   # x corresponds to columns
            y=-super_food[0],  # y corresponds to rows
            name='Blobble Superfood',
            mode='markers',
            marker=dict(
                sizeref=self._SIZE_REF,
                size=15,
                symbol='hexagram',
                color='green'))
        )

        fig.add_trace(go.Scatter(
            x=good_food[1],   # x corresponds to columns
            y=-good_food[0],  # y corresponds to rows
            name='Good Food',
            mode='markers',
            marker=dict(
                sizeref=self._SIZE_REF,
                size=15,
                symbol='hexagram',
                color='lightgreen'))
        )

        fig.add_trace(go.Scatter(
            x=unhealthy_food[1],   # x corresponds to columns
            y=-unhealthy_food[0],  # y corresponds to rows
            name='Junk Food',
            mode='markers',
            marker=dict(
                sizeref=self._SIZE_REF,
                size=15,
                symbol='asterisk-open',
                color='orange'))
        )

        fig.add_trace(go.Scatter(
            x=bad_weed[1],   # x corresponds to columns
            y=-bad_weed[0],  # y corresponds to rows
            name='Do Not Eat!',
            mode='markers',
            marker=dict(
                sizeref=self._SIZE_REF,
                size=15,
                symbol='asterisk-open',
                color='red'))
        )

        # Now add the blobble.
        # The blobble is depicted by an asterisk-open marker overlaid with a default (circle) marker
        fig.add_trace(go.Scatter(
            x=[self._blobble_state[1]],  # x corresponds to columns
            y=[self._blobble_state[0]],  # y corresponds to rows
            showlegend=False,
            mode='markers',
            marker=dict(
                sizeref=self._SIZE_REF,
                size=self._blobble_state[2] * 5,
                symbol='asterisk-open',
                color='gray',
                line=dict(width=2, color='gray')
            )
        )
        )

        text_blobble = "Blobble (N="+str(self._blobble_state[0])+", E="+str(self._blobble_state[1])+")"

        fig.add_trace(go.Scatter(
            x=[self._blobble_state[1]],  # x corresponds to columns
            y=[self._blobble_state[0]],  # y corresponds to rows
            text=text_blobble,
            showlegend=False,
            mode='markers',
            marker=dict(
                sizeref=self._SIZE_REF,
                size=self._blobble_state[2] * 4,
                color=self._HEALTH_COLOURS[
                    int(self._blobble_state[2] / self._MAX_HEALTH * (len(self._HEALTH_COLOURS) - 1))],
                line=dict(width=2, color='gray')
            )
        )
        )

        fig.update_yaxes(automargin=True, range=[self._MIN_LOC - 0.5, self._MAX_LOC + 0.5], nticks=40)
        fig.update_xaxes(automargin=True, range=[self._MIN_LOC - 0.5, self._MAX_LOC + 0.5], nticks=40)

        fig.update_layout(
            title={
                'text': str(self._episode - 1) + '-score:' + str(self._rewards_so_far) + ' (' + str(
                    self._best_episode) + ')',
                'y': 0.5,
                'x': 0.9,
                'xanchor': 'center',
                'yanchor': 'top'})

        fig.update_layout(
            width=912,
            height=608,
            margin=dict(r=10, l=10, b=10, t=10))

        if mode == 'human':
            fig.show()

        elif mode == 'rgb_array':
            img_as_bytes = fig.to_image(format='png')
            np_img = np.array(Image.open(io.BytesIO(img_as_bytes)))
            return np_img

    def close(self):
        """

        :return:
        """
        print('Closing Blobble Environment')

    def render_print(self):
        """
        For testing
        :return:
        """
        print('Blobble details are:')
        print('  Health             : ', self._blobble_state[2])

        print('  Position northerly : ', self._blobble_state[0])
        print('  Position easterly  : ', self._blobble_state[1])

        print('  Local nutrition    : ', self._blobble_state[3])
        print('  Remaining food     :', len(self._food[0]))
        print('  Blobble hunger     : ', self._hunger)

    def reset_test(self, reset_state, reset_food, taste = True, smell = True):
        """
        For testing:
        Allows reset of the food matrix and the location/health of the blobble

                '''
        Initialise

        :param reset_state:
        A numpy array depicting the reset state of the blobble. [northerly, easterly, health]

        :param reset_food:
        If reset_food==None, the food will be allocated randomly on each reset. This is the defauly.
        Alternatively, a numpy array can be specified. This is especially useful for testing.

        :return:
        """
        self._blobble_state[0:3] = deepcopy(reset_state)
        self._food = deepcopy(reset_food)
        self._TASTE = taste
        self._SMELL = smell
        self._blobble_state[3] = self._taste()
        self._blobble_state[4:8] = self._smell()

        return self._blobble_state
