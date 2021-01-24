import unittest
from gym_blobble.envs.blobble_env import BlobbleEnv
import numpy as np


TEST_FOOD = np.array([
             # -10                                      0                                      +10
             # 0                                                                               20
             [-5, -4, -3, -2, -1,  0,  1,  2,  3,  4,  5,  0,  4, -1,  0,  4,  1,  2, -4, -4,  1],   # +10   0
             [4,   3,  3, -3, -4,  4,  4, -2,  4,  2,  0, -1, -3, -3,  4, -5,  3, -4, -1,  0, -2],
             [-4, -2, -3,  3,  1,  1, -5,  1, -5, -2,  1,  1,  2, -3, -2, -4,  4, -1, -3, -3,  0],
             [1,   0, -1, -4, -5,  1, -2, -3, -1,  4,  1, -5,  4,  2, -5,  5,  3, -2, -3, -4, -3],
             [-3,  2,  3,  5, -4, -3, -4, -4, -5,  5,  4, -1,  1,  4, -1, -2,  3, -5, -3, -1, -5],
             [-4, -4,  3,  4,  5, -4, -2, -1, -1, -4,  0, -1, -3, -1, -3, -2, -4,  4, -4, -5,  0],
             [-2,  4, -3, -2, -2, -5, -5,  3, -2,  4, -5, -4, -1, -5, -4, -1,  3, -4, -5,  0,  0],
             [-5, -5,  3,  0, -1,  4,  5, -5, -5, -4, -5,  3,  2,  1, -2,  1,  1, -2,  3, -1, -5],
             [-3,  4, -5,  1, -5, -2,  0,  1,  1,  4,  5,  2,  1, -1, -2,  2,  2,  0,  1, -1, -2],
             [-4, -2, -4,  4, -1, -4,  1,  0,  2,  -1,  5,  5, -5, -1, -5,  3, -4,  2, -3, -5, -1],
             #                                         \/ Position 0,0
             [1,   2, -3, -2,  1,  4, -4,  4,  2,  -1,  0,  0, -5,  0,  2, -3, -3,  3, -2, -3,  0],
             [2,   2,  0, -2, -5,  3, -2,  2,  2,  2,  -3, -5, -5, -2, -3,  3,  3,  1,  5,  0, -1],
             [-1,  3, -2, -1,  3,  0, -5,  4,  1,  1,  0,  2,  1, -3, -3, -4,  0, -2,  5, -1,  2],
             [3,   2, -1,  4, -5, -2,  0, -3,  2,  2, -3, -2, -1,  5,  5,  3,  0, -1,  3, -3, -4],
             [4,   2,  5, -3,  5,  1,  3,  4,  2, -5, -5, -2,  4, -1,  3, -4, -2,  1,  0, -5, -1],
             [1,  -3,  2,  4,  3, -3,  1,  1, -2, -3, -4,  3, -3, -3,  1,  0,  3, -1,  2,  4,  3],
             [2,  -4,  1,  0, -5, -5, -3,  5,  0,  2, -5, -1, -2, -3, -3, -4,  3,  4, -4,  2, -3],
             [-4,  4,  5, -2, -4,  1, -3, -3,  5, -3,  1,  2,  2, -3, -3, -4,  5, -2, -4,  1,  4],
             [-2,  0, -4, -5,  3,  0,  2, -1, -4,  2, -3, -3,  2,  2,  0, -2, -4, -1, -5, -3, -1],
             [-3,  1,  3, -4, -3, -5,  0, -5, -5, -5,  5, -5,  0,  3, -1, -1,  1,  4, -2,  2, -3],
             [1,  -3,  0, -2, -2,  5,  5,  2, -3, -1, -5, -4,  4,  4, -5,  1,  4,  5, -5,  4,  5]])  # -10   20


class TestBlobbleEnv(unittest.TestCase):

    def test_reset_observation(self):
        blobble_env = BlobbleEnv()
        observation = blobble_env.reset()
        expected = np.array((0, 0, 5), dtype=np.float)
        np.testing.assert_array_equal(expected, observation[0:3], 'Observation not as expected after reset')

        observation = blobble_env.reset_test([1, 2, 3], TEST_FOOD)
        expected = np.array((1, 2, 3), dtype=np.float)
        np.testing.assert_array_equal(expected, observation[0:3], 'Observation not as expected after reset test')

    def test_eat_no_step(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([0, 0, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(0)  # No step
        self.assertEqual(0, observation[0], 'Should be northerly=0')
        self.assertEqual(0, observation[1], 'Should be easterly=0')

    def test_move_north(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([0, 0, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(1)  # Step North
        self.assertEqual(1, observation[0], 'Should be northerly=1')
        self.assertEqual(0, observation[1], 'Should be easterly=0')

    def test_move_south(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([0, 0, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(0)  # No step
        observation, reward, done, _ = blobble_env.step(1)  # Step North
        observation, reward, done, _ = blobble_env.step(2)  # Step South
        self.assertEqual(0, observation[0], 'Should be northerly=0')
        self.assertEqual(0, observation[1], 'Should be easterly=0')

    def test_move_east(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([0, 0, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(3)  # Step East
        self.assertEqual(0, observation[0], 'Should be northerly=0')
        self.assertEqual(1, observation[1], 'Should be easterly=1')

    def test_move_west(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([0, 0, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(3)  # Step East
        self.assertEqual(0, observation[0], 'Should be northerly=0')
        self.assertEqual(1, observation[1], 'Should be easterly=1')
        observation, reward, done, _ = blobble_env.step(4)  # Step West
        self.assertEqual(0, observation[0], 'Should be northerly=0')
        self.assertEqual(0, observation[1], 'Should be easterly=0')

    def test_taste(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        observation = blobble_env.reset_test([0, 0, 5], TEST_FOOD)
        self.assertEqual(0, observation[3], 'Should be taste=0')
        observation = blobble_env.reset_test([-1, -4, 5], TEST_FOOD)
        self.assertEqual(-2, observation[3], 'Should be taste=-2')
        observation = blobble_env.reset_test([3, 3, 5], TEST_FOOD)
        self.assertEqual(1, observation[3], 'Should be taste=1')
        observation = blobble_env.reset_test([3, 4, 5], TEST_FOOD)
        self.assertEqual(-2, observation[3], 'Should be taste=-2')

    '''
    smell_tests
    
    These tests start at the centre of the TEST_FOOD where the matrix is as follows:
        [ ., ., 1, .,  .],
        [ ., 4, 2, 5,  .],
        [ 2, 4, B, 5, -5],
        [ ., 2, 0, -5, .],
        [ ., ., 0, .,  .],
    
    '''
    def test_smell_north(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([0, 0, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(0)  # Eat food at current location
        '''
        [ ., ., ., ., .],
        [ ., 4, 2, 3, .],
        [ ., ., B, ., .]
        '''
        self.assertEqual(3, observation[4], 'Should be sniff north=4')

    def test_smell_south(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([0, 0, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(0)  # Eat food at current location
        '''
        [ ., ., B, ., .]
        [ ., 2, 0, -5 .],
        [ ., ., 0, ., .],
        '''
        self.assertEqual(-2, observation[5], 'Should be sniff south=0.75')

    def test_smell_east(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([0, 0, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(0)  # Eat food at current location
        '''
        [ ., ., ., 5,.],
        [ ., ., B, 0, .],
        [ ., ., ., -5,.],
        '''
        self.assertEqual(0, observation[6], 'Should be sniff east=0')

    def test_smell_west(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([0, 0, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(0)  # Eat food at current location
        '''
        [ ., 4, ., ., .],
        [ 2, 4, B, ., .],
        [ ., 2, ., ., .],
        '''
        self.assertEqual(0, observation[7], 'Should be sniff west=3')

    def test_reward_eat_no_nutritional_value(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([0, 0, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(0)  # Eat food=0
        self.assertEqual(5, observation[2], 'Should be health=5')
        self.assertEqual(0.5, reward, 'Should be reward=0.5')

    def test_reward_eat_nutritional_value(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([0, 0, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(3)  # Move East
        observation, reward, done, _ = blobble_env.step(1)  # Move North
        observation, reward, done, _ = blobble_env.step(0)  # Eat food=5

        self.assertEqual(10, observation[2], 'Should be health=10')
        self.assertEqual(1.0, reward, 'Should be reward=1.0')

    def test_move_too_far_north(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([10, 0, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(1)  # Step North
        self.assertEqual(10, observation[0], 'Should be northerly=10')
        self.assertEqual(0, observation[1], 'Should be easterly=0')

    def test_move_too_far_south(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([-10, 0, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(2)  # Step South
        self.assertEqual(-10, observation[0], 'Should be northerly=-10')
        self.assertEqual(0, observation[1], 'Should be easterly=0')

    def test_move_too_far_east(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([0, 10, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(3)  # Step East
        self.assertEqual(0, observation[0], 'Should be northerly=0')
        self.assertEqual(10, observation[1], 'Should be easterly=10')

    def test_move_too_far_west(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([0, -10, 5], TEST_FOOD)
        observation, reward, done, _ = blobble_env.step(4)  # Step West
        self.assertEqual(0, observation[0], 'Should be northerly=0')
        self.assertEqual(-10, observation[1], 'Should be easterly=-10')

    def test_no_taste(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        observation = blobble_env.reset_test([3, 3, 5], TEST_FOOD, taste=False)
        self.assertEqual(0, observation[3], 'Should be no taste 0')
        observation, reward, done, _ = blobble_env.step(4)  # Step West
        self.assertEqual(0, observation[3], 'Should be no taste 0')

    def test_no_smell(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        observation = blobble_env.reset_test([3, 3, 5], TEST_FOOD, smell=False)
        self.assertEqual(0, observation[4], 'Should be no smell north')
        self.assertEqual(0, observation[5], 'Should be no smell south')
        self.assertEqual(0, observation[6], 'Should be no smell east')
        self.assertEqual(0, observation[7], 'Should be no smell west')

        observation, reward, done, _ = blobble_env.step(4)  # Step West
        self.assertEqual(0, observation[4], 'Should be no smell north after step')
        self.assertEqual(0, observation[5], 'Should be no smell south after step')
        self.assertEqual(0, observation[6], 'Should be no smell east after step')
        self.assertEqual(0, observation[7], 'Should be no smell west after step')

    def test_eat(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        observation = blobble_env.reset_test([3, 3, 5], TEST_FOOD)
        self.assertEqual(1, observation[3], 'Should be food of value 1')
        observation, reward, done, _ = blobble_env.step(0)  # Eat food
        self.assertEqual(0, observation[3], 'Should be no food')

    def test_eat_and_move_east(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        observation = blobble_env.reset_test([3, 3, 5], TEST_FOOD)
        print(observation)
        self.assertEqual(1, observation[3], 'Should be food of value 1')
        observation, reward, done, _ = blobble_env.step(7)  # Eat food, Go East
        print(observation)
        observation, reward, done, _ = blobble_env.step(4)  # Go West back to original (no food) spot
        print(observation)
        self.assertEqual(0, observation[3], 'Should be no food')
        observation, reward, done, _ = blobble_env.step(4)
        blobble_env.render(mode='human')

    def test_eat_and_move_west(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        observation = blobble_env.reset_test([3, 3, 5], TEST_FOOD)
        print(observation)
        self.assertEqual(1, observation[3], 'Should be food of value 1')
        observation, reward, done, _ = blobble_env.step(8)  # Eat food, Go West
        print(observation)
        observation, reward, done, _ = blobble_env.step(3)  # Go East back to original (no food) spot
        print(observation)
        self.assertEqual(0, observation[3], 'Should be no food')

    def test_eat_and_move_west(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        observation = blobble_env.reset_test([3, 3, 5], TEST_FOOD)
        print(observation)
        self.assertEqual(1, observation[3], 'Should be food of value 1')
        observation, reward, done, _ = blobble_env.step(8)  # Eat food, Go West
        print(observation)
        observation, reward, done, _ = blobble_env.step(3)  # Go East back to original (no food) spot
        print(observation)
        self.assertEqual(0, observation[3], 'Should be no food')

    def test_render_human(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset()
        blobble_env.reset_test([-2, 5, 5], TEST_FOOD)
        blobble_env.render(mode='human')

    def test_food(self):
        print(TEST_FOOD)
        super_food = np.argwhere(TEST_FOOD == -30).T -10
        print(super_food)


    # def test_render_rgb(self):
    #   blobble_env = BlobbleEnv()
    #    blobble_env.reset()
    #    blobble_env.reset_test([0, 0, 5], TEST_FOOD)
    #    blobble_env.render(mode='rgb_array')


if __name__ == '__main__':
    unittest.main()
