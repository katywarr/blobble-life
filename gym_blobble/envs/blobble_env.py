import io
from PIL import Image
import gym
import numpy as np
import plotly.graph_objects as go

from gym import spaces, error, utils
from gym.utils import seeding


class BlobbleEnv(gym.Env):
    """
    Description:


    Observations:

        Type: Box(3)
        Num     Observation                             Min             Max
        0       Blobble X location                      MIN_LOC(-10)    MAX_LOC (10)
        1       Blobble Y location                      MIN_LOC(-10)    MAX_LOC (10)
        2       Health                                  0               MAX_HEALTH (10)
        3       Taste (Food nutritional value at loc)   -5              +5     (set to zero if no food)
        4       Food smell north                        -10             +10
        5       Food smell south                        -10             +10
        6       Food smell east                         -10             +10
        7       Food smell west                         -10             +10


    Actions:

        Type: Discrete(10)
        Num     Action
        0       eat
        1       eat, move E
        2       eat, move S
        3       eat, move W
        4       eat, move N
        5       move E
        6       move S
        7       move W
        8       move N

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

        self.seed()
        # Reset the env to its start position
        self.reset()
        self._best_episode = 0

    def reset(self, initial_state=None, test_food_at_location=None):
        """
        Resets the Blobble to its initial state

        :param initial_state:
        Initial state array [x, y, health]. If not specified, it will default.
        :param test_food_at_location:
        For test purposes only - forces food to be placed at the initial location
        :return:
        observation
        """

        if initial_state is None:
            initial_state = ([0, 0, 5])
        self._episode += 1

        if test_food_at_location is not None:  # Simply a way of forcing food at current loc for testing purposes
            self._food = np.zeros((self._MAX_LOC-self._MIN_LOC+1, self._MAX_LOC-self._MIN_LOC+1), dtype=int)
            self._food[initial_state[0] - self._MIN_LOC][initial_state[1]-self._MIN_LOC]=test_food_at_location

        else:  # This is the proper code path
            # Reset Blobble food locations
            self._food = self.np_random.randint(low=-5,
                                                high=5,
                                                size=(self._MAX_LOC-self._MIN_LOC+1, self._MAX_LOC-self._MIN_LOC+1))

        self._hunger = 0

        # Create New Blobble
        self._blobble_state = np.array(initial_state)
        local_nutrition = self._food[self._blobble_state[0] - self._MIN_LOC][self._blobble_state[1] - self._MIN_LOC]
        self._blobble_state = np.append(self._blobble_state, [0, 0, 0, 0, 0])

        self._rewards_so_far = 0

        self._taste()
        self._smell()

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
        self._food[self._blobble_state[0] - self._MIN_LOC][self._blobble_state[1] - self._MIN_LOC] = 0

        return

    def _smell(self):

        x = self._blobble_state[0] - self._MIN_LOC
        y = self._blobble_state[1] - self._MIN_LOC
        smell_east, smell_north, smell_south, smell_west = [0, 0, 0, 0]

        smell_strength_fraction = 1

        for i in range(1, 5):
            if y+i < (self._MAX_LOC-self._MIN_LOC + 1):  # Check not at edge of blobble env
                smell_east = smell_east + (self._food[x, y+i]/smell_strength_fraction)
            if y-i >= 0:                                  # Check not at edge of blobble env
                smell_west = smell_west + (self._food[x, y-i]/smell_strength_fraction)
            if x+i < (self._MAX_LOC-self._MIN_LOC + 1):  # Check not at edge of blobble env
                smell_north = smell_north + (self._food[x+i, y]/smell_strength_fraction)
            if x-i >= 0:                                  # Check not at edge of blobble env
                smell_south = smell_south + (self._food[x-i, y]/smell_strength_fraction)

            smell_strength_fraction = smell_strength_fraction * 2  # Increase fraction - food further away smells less

        self._blobble_state[4] = smell_north
        self._blobble_state[5] = smell_south
        self._blobble_state[6] = smell_east
        self._blobble_state[7] = smell_west


    def _taste(self):
        self._blobble_state[3] = self._food[self._blobble_state[0] - self._MIN_LOC][
            self._blobble_state[1] - self._MIN_LOC]

    def step(self, action):
        """
        Perform an action

        :param action:
        Action to be performed
        :return:
        observation, reward, done, {}
        """

        if action < 5:  # Eat (if there is food)
            self._eat()
        else:
            self._hunger += 1

        if (action == 1) or (action == 5):  # Move East
            self._blobble_state[0] = min(self._blobble_state[0] + 1, self._MAX_LOC)
        if (action == 2) or (action == 6):  # Move South
            self._blobble_state[1] = max(self._blobble_state[1] - 1, self._MIN_LOC)
        if (action == 3) or (action == 7):  # Move West
            self._blobble_state[0] = max(self._blobble_state[0] - 1, self._MIN_LOC)
        if (action == 4) or (action == 8):  # Move North
            self._blobble_state[1] = min(self._blobble_state[1] + 1, self._MAX_LOC)

        self._taste()
        self._smell()

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

        if self._hunger > 5:
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
        super_food = np.argwhere(self._food > 3).T + self._MIN_LOC
        good_food = np.argwhere((self._food > 0) & (self._food <= 3)).T + self._MIN_LOC
        unhealthy_food = np.argwhere((self._food < 0) & (self._food >= -3)).T + self._MIN_LOC
        bad_weed = np.argwhere(self._food < -3).T + self._MIN_LOC

        # Depict the food locations
        fig = go.Figure(data=go.Scatter(
            x=super_food[0],
            y=super_food[1],
            name='Blobble Superfood',
            mode='markers',
            marker=dict(
                sizeref=self._SIZE_REF,
                size=15,
                symbol='hexagram',
                color='green'))
        )

        fig.add_trace(go.Scatter(
            x=good_food[0],
            y=good_food[1],
            name='Good Food',
            mode='markers',
            marker=dict(
                sizeref=self._SIZE_REF,
                size=15,
                symbol='hexagram',
                color='lightgreen'))
        )

        fig.add_trace(go.Scatter(
            x=unhealthy_food[0],
            y=unhealthy_food[1],
            name='Junk Food',
            mode='markers',
            marker=dict(
                sizeref=self._SIZE_REF,
                size=15,
                symbol='asterisk-open',
                color='orange'))
        )

        fig.add_trace(go.Scatter(
            x=bad_weed[0],
            y=bad_weed[1],
            name='Do Not Eat!',
            mode='markers',
            marker=dict(
                sizeref=self._SIZE_REF,
                size=15,
                symbol='asterisk-open',
                color='red'))
        )

        fig.add_trace(go.Scatter(
            x=[self._blobble_state[0]],
            y=[self._blobble_state[1]],
            showlegend=False,
            mode='markers',
            marker=dict(
                sizeref=self._SIZE_REF,
                size=self._blobble_state[2]*5,
                symbol='asterisk-open',
                color='gray',
                line=dict(width=2, color='gray')
                )
            )
        )

        fig.add_trace(go.Scatter(
            x=[self._blobble_state[0]],
            y=[self._blobble_state[1]],
            text='Blobble',
            showlegend=False,
            mode='markers',
            marker=dict(
                sizeref=self._SIZE_REF,
                size=self._blobble_state[2]*4,
                color=self._HEALTH_COLOURS[
                    int(self._blobble_state[2] / self._MAX_HEALTH * (len(self._HEALTH_COLOURS) - 1))],
                line=dict(width=2, color='gray')
                )
            )
        )

        fig.update_yaxes(automargin=True, range=[self._MIN_LOC-0.5, self._MAX_LOC+0.5], nticks=40)
        fig.update_xaxes(automargin=True, range=[self._MIN_LOC-0.5, self._MAX_LOC+0.5], nticks=40)

        fig.update_layout(
            title={
                'text': str(self._episode-1)+'-score:'+str(self._rewards_so_far)+' ('+str(self._best_episode)+')',
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
        print('  Health: ', self._blobble_state[2])
        print('  Position x : ', self._blobble_state[0])
        print('  Local nutrition : ', self._blobble_state[3])
        print('  Remaining food :', len(self._food[0]))
        print('  Blobble hunger: ', self._hunger)
