import unittest
from .context import environment
from environment.BlobbleEnv_v1 import BlobbleEnv


class TestBlobbleEnv(unittest.TestCase):

    def test_no_move_no_food_to_eat(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset(initial_state=[0, 0, 5], test_food_at_location=0)
        observation, reward, done, _ = blobble_env.step(0)
        self.assertEqual(0, observation[0], 'Should be x=0')
        self.assertEqual(0, observation[1], 'Should be y=0')
        self.assertEqual(5, observation[2], 'Should be health=5')
        self.assertEqual(0.5, reward, 'Should be reward=0.5')

    def test_no_move_food_to_eat(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset(initial_state=[0, 0, 5], test_food_at_location=5)
        observation, reward, done, _ = blobble_env.step(0)
        self.assertEqual(0, observation[0], 'Should be x=0')
        self.assertEqual(0, observation[1], 'Should be y=0')
        self.assertEqual(10, observation[2], 'Should be health=10')
        self.assertEqual(1.0, reward, 'Should be reward=1.0')

    def test_move_east(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset(initial_state=[0, 0, 5])
        observation, reward, done, _ = blobble_env.step(5)
        self.assertEqual(1, observation[0], 'Should be x=1')
        self.assertEqual(0, observation[1], 'Should be y=0')
        self.assertEqual(5, observation[2], 'Should be health=5')
        self.assertEqual(0.5, reward, 'Should be reward=0.5')

    def test_move_west(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset(initial_state=[0, 0, 5])
        observation, reward, done, _ = blobble_env.step(6)
        self.assertEqual(0, observation[0], 'Should be x=0')
        self.assertEqual(-1, observation[1], 'Should be y=-1')
        self.assertEqual(5, observation[2], 'Should be health=5')
        self.assertEqual(0.5, reward, 'Should be reward=0.5')

    def test_move_south(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset(initial_state=[0, 0, 5])
        observation, reward, done, _ = blobble_env.step(7)
        self.assertEqual(-1, observation[0], 'Should be x=-1')
        self.assertEqual(0, observation[1], 'Should be y=0')
        self.assertEqual(5, observation[2], 'Should be health=5')
        self.assertEqual(0.5, reward, 'Should be reward=0.5')

    def test_move_north(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset(initial_state=[0, 0, 5])
        observation, reward, done, _ = blobble_env.step(8)
        self.assertEqual(0, observation[0], 'Should be x=0')
        self.assertEqual(1, observation[1], 'Should be y=1')
        self.assertEqual(5, observation[2], 'Should be health=5')
        self.assertEqual(0.5, reward, 'Should be reward=0.5')

    def test_move_east_eat(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset(initial_state=[0, 0, 5], test_food_at_location=3)
        observation, reward, done, _ = blobble_env.step(1)
        self.assertEqual(1, observation[0], 'Should be x=1')
        self.assertEqual(0, observation[1], 'Should be y=0')
        self.assertEqual(8, observation[2], 'Should be health=8')
        self.assertEqual(1.0, reward, 'Should be reward=1.0')

    def test_move_west_eat(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset(initial_state=[0, 0, 5], test_food_at_location=3)
        observation, reward, done, _ = blobble_env.step(2)
        self.assertEqual(0, observation[0], 'Should be x=0')
        self.assertEqual(-1, observation[1], 'Should be y=-1')
        self.assertEqual(8, observation[2], 'Should be health=8')
        self.assertEqual(1.0, reward, 'Should be reward=1.0')

    def test_move_south_eat(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset(initial_state=[0, 0, 5], test_food_at_location=-3)
        observation, reward, done, _ = blobble_env.step(3)
        self.assertEqual(-1, observation[0], 'Should be x=-1')
        self.assertEqual(0, observation[1], 'Should be y=0')
        self.assertEqual(2, observation[2], 'Should be health=2')
        self.assertEqual(0.5, reward, 'Should be reward=0.5')

    def test_move_north_eat(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset(initial_state=[0, 0, 5], test_food_at_location=-0)
        observation, reward, done, _ = blobble_env.step(4)
        self.assertEqual(0, observation[0], 'Should be x=0')
        self.assertEqual(1, observation[1], 'Should be y=1')
        self.assertEqual(5, observation[2], 'Should be health=5')
        self.assertEqual(0.5, reward, 'Should be reward=0.5')

    def test_move_north_eat_max(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset(initial_state=[0, 0, 8], test_food_at_location=5)
        observation, reward, done, _ = blobble_env.step(4)
        self.assertEqual(0, observation[0], 'Should be x=0')
        self.assertEqual(1, observation[1], 'Should be y=1')
        self.assertEqual(10, observation[2], 'Should be health=10')
        self.assertEqual(1.0, reward, 'Should be reward=1.0')

    def test_render_human(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset(initial_state=[0, 0, 5])
        blobble_env.render()

    def test_render_rgb(self):
        blobble_env = BlobbleEnv()
        blobble_env.reset(initial_state=[0, 0, 5])
        rgb_array = blobble_env.render(mode='rgb_array')


if __name__ == '__main__':
    unittest.main()
