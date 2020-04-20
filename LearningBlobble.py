from gym_blobble.envs import BlobbleEnv
import imageio as imageio
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.agents.dqn import dqn_agent


def collect_step(environment, policy, buffer):
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def create_blobble_video(video_filename, num_episodes=1, fps=30):
    filename = video_filename + ".mp4"

    blobble_env = BlobbleEnv()

    print('Observation space: ' + str(blobble_env.observation_space))
    print('Action space:      ' + str(blobble_env.action_space))
    num_observations = blobble_env.observation_space.shape[0]
    num_actions = blobble_env.action_space.n

    print(num_observations)
    print(num_actions)

    with imageio.get_writer(filename, fps=fps) as video:
        for i in range(num_episodes):
            print('Episode: ' + str(i))
            blobble_env.reset([0, 0, 10])
            done = False
            while not done:
                action = np.random.randint(8)
                print('  Next Action: ', action)
                observation, reward, done, _ = blobble_env.step(action)
                # blobble_env.render_print()
                # video.append_data(blobble_env.render(mode='rgb_array'))

    blobble_env.close()


def compute_avg_return(environment, policy, num_episodes=10):
    total_return = 0.0
    for i in range(num_episodes):
        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes

    return avg_return.numpy()[0]


def train_neural_network(agent,
                         train_env,
                         eval_env,
                         num_eval_episodes=10,
                         log_interval=200,
                         eval_interval=1000,
                         collect_steps_per_iteration=1):

    print('Train the Network')
    replay_buffer_max_length = 100000
    num_iterations = 10000

    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]
    print('Average return (initial): ', avg_return)

    '''
    collect_data_spec returns a trajectory spec
    A trajectory spec is a tuple that contains:
    - the state of the environment in some time step (observation)
    - the action that the agent should take in that state (action)
    - the state in which the environment will be after the action is taken
    '''
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    '''
    The agent needs access to the replay buffer. 
    This is provided by creating an iterable tf.data.Dataset pipeline which will feed data to the agent.

    Each row of the replay buffer only stores a single observation step. 
    But since the DQN Agent needs both the current and next observation to compute the loss, 
    the dataset pipeline will sample two adjacent rows for each item in the batch (num_steps=2).
    '''
    dataset = replay_buffer.as_dataset(
        num_parallel_calls=3,
        sample_batch_size=64,
        num_steps=2).prefetch(3)

    dataset_iterator = iter(dataset)

    for i in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
        collect_steps_per_iteration = 2
        for _ in range(collect_steps_per_iteration):
            collect_step(train_env, agent.collect_policy, replay_buffer)

        # Sample a batch of data from the buffer and update the agent's network.
        experience, unused_info = next(dataset_iterator)
        train_loss = agent.train(experience).loss
        step = agent.train_step_counter.numpy()

        if step % log_interval == 0:
            print('step = {0}: loss = {1}'.format(step, train_loss))

        if step % eval_interval == 0:
            avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
            print('step = {0}: Average Return = {1}'.format(step, avg_return))
            returns.append(avg_return)

    iterations = range(0, num_iterations + 1, eval_interval)
    plt.plot(iterations, returns)
    plt.ylabel('Average Return')
    plt.xlabel('Iterations')
    # plt.ylim(top=250)
    # plt.ion()  # Turn on interactive mode so the following line is non-blocking
    plt.show()


def create_neural_network_agent(env):
    # Create a neural network that can learn to predict the QValues (expected returns) for all the possible
    # Blobble actions, given a specific observation
    # fc_layer_params = (100,)
    fc_layer_params = (75, 40)

    q_net = q_network.QNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=fc_layer_params)

    # Now create the DqnAgent
    learning_rate = 1e-3  # @param {type:"number"}

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate)

    train_step_counter = tf.Variable(0)

    agent = dqn_agent.DqnAgent(
        env.time_step_spec(),
        env.action_spec(),
        q_network=q_net,
        optimizer=optimizer,
        td_errors_loss_fn=common.element_wise_squared_loss,
        train_step_counter=train_step_counter)

    agent.initialize()

    return agent


class QNetworkAgent():
    """
    Wrapper class to provide a Deep Neural Network agent for any provided tf-agent environment.
    """

    def __init__(self, env_name='blobble-world-v0'):
        """
        Initalise the agent by training a neural network for the passed tf-agent environment

        :param env_name:
        Name of environment for the agent so solve
        """
        self._env_name = env_name

        # Create training and evaluation environments
        self._train_py_env = suite_gym.load(self._env_name)
        self._eval_py_env = suite_gym.load(self._env_name)

        # Convert the training and test environments to Tensors
        self._train_env = tf_py_environment.TFPyEnvironment(self._train_py_env)
        self._eval_env = tf_py_environment.TFPyEnvironment(self._eval_py_env)
        print('=====================================================')
        print('Environments created for : ', self._env_name)
        print('Training Environment')
        print('  Observation Spec:')
        print('    ', self._train_env.time_step_spec().observation)
        print('  Reward Spec:')
        print('    ', self._train_env.time_step_spec().reward)
        print('  Action Spec:')
        print('    ', self._train_env.action_spec())
        print('Evaluation Environment')
        print('  Observation Spec:')
        print('    ', self._eval_env.time_step_spec().observation)
        print('  Reward Spec:')
        print('    ', self._eval_env.time_step_spec().reward)
        print('  Action Spec:')
        print('    ', self._eval_env.action_spec())
        print('=====================================================')

        print('Create and train a neural network agent')
        self._neural_network_agent = create_neural_network_agent(self._train_env)

        train_neural_network(self._neural_network_agent,
                             self._train_env,
                             self._eval_env,
                             num_eval_episodes=10,
                             log_interval=200,
                             eval_interval=1000,
                             collect_steps_per_iteration=1)


    def get_random_baseline_performance(self, iterations=10):
        """
        Establish a baseline performance based on random behaviour
        :param iterations:
        :return:
        """
        random_policy = random_tf_policy.RandomTFPolicy(self._train_env.time_step_spec(),
                                                        self._train_env.action_spec())

        return compute_avg_return(self._train_env, random_policy, iterations)

    def run_agent(self, video_filename=None, num_episodes=50, fps=2, random=False):
        """
        Run iterations.
        :param video_filename:
        If specified, create a video of the iterations in the filename
        :param num_episodes:
        Number of episodes to run
        :param fps:
        Frames per second for video
        :param policy:
        For random behaviour, set policy as follows
        :return:
        """
        run_py_env = suite_gym.load(self._env_name)
        run_env = tf_py_environment.TFPyEnvironment(run_py_env)

        filename = video_filename + ".mp4"

        if not random:
            policy = self._neural_network_agent.policy
        else:
            policy = random_tf_policy.RandomTFPolicy(run_env.time_step_spec(),
                                                     run_env.action_spec())

        if video_filename is not None:
            with imageio.get_writer(filename, fps=fps) as video:
                for episode in range(num_episodes):
                    print('Episode: ', episode)
                    # Reset the evaluation environment
                    time_step = run_env.reset()
                    while not time_step.is_last():
                        action_step = policy.action(time_step)
                        time_step = run_env.step(action_step.action)
                        tf.print(action_step.action, time_step)
                        video.append_data(run_py_env.render())
        else:
            for episode in range(num_episodes):
                print('Episode: ', episode)
                # Reset the evaluation environment
                time_step = run_env.reset()
                while not time_step.is_last():
                    action_step = policy.action(time_step)
                    time_step = run_env.step(action_step.action)
                    tf.print(action_step.action, time_step)


def main():
    # agent = QNetworkAgent('CartPole-v0')
    agent = QNetworkAgent('blobble-life-v0')

    agent.run_agent('BlobbleVideo_taste_smell', num_episodes=10)


if __name__ == "__main__":
    main()
