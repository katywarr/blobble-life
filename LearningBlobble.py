from gym_blobble.BlobbleConfig import BlobbleConfig

import imageio as imageio
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime
import os

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

    if num_episodes > 0:
        avg_return = total_return / num_episodes
        return avg_return.numpy()[0]
    else:
        return 0.0


def train_neural_network(agent,
                         train_env,
                         eval_env,
                         num_train_iterations,
                         log_interval,
                         eval_interval,
                         num_eval_episodes,
                         replay_buffer_max_length,
                         collect_steps_per_iteration,
                         output_folder,
                         timestamp):
    print('Train the Network')
    replay_buffer_max_length = replay_buffer_max_length
    num_iterations = num_train_iterations

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
        collect_steps_per_iteration = collect_steps_per_iteration
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
    filename = os.path.join(output_folder, timestamp + "-training_returns" + ".png")
    plt.savefig(filename)


def create_neural_network_agent(env, learning_rate, fc_layer_params):
    # Create a neural network that can learn to predict the QValues (expected returns) for all the possible
    # Blobble actions, given a specific observation
    print('Creating Neural Network (layers={:s}, learning rate={:e})'.format(str(fc_layer_params), learning_rate))
    q_net = q_network.QNetwork(
        env.observation_spec(),
        env.action_spec(),
        fc_layer_params=fc_layer_params)

    # Now create the DqnAgent

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

    # Let's take a look at the network
    q_net.summary()

    return agent


class QNetworkAgent:
    """
    Wrapper class to provide a Deep Neural Network agent for any provided tf-agent environment.
    """

    def __init__(self,
                 env_name='blobble-world-v0'
                 ):
        """
        Initalise the agent by training a neural network for the passed tf-agent environment

        :param env_name:
        Name of environment for the agent so solve
        """
        self._env_name = env_name

        # Take a timestamp. This will be used for any output files created in the output folder
        self._timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")

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

        self._config = BlobbleConfig('blobble_config.ini')
        self._config.print_config()

        # Get the demonstration parameters and output folder. We don't need these just yet but it's
        # good to do now in case there is an error in the config file (exception will be thrown)
        self._output_folder = (self._config.get_output_params()['output_folder'])

        self._num_demo_episodes = int(self._config.get_output_params()['num_demonstration_episodes'])
        demo_video = (self._config.get_output_params()['demonstration_video'])
        if demo_video == 'True':
            self._demo_video = True
        else:
            self._demo_video = False

        # Get and check the advanced learning parameters
        self._learning_rate = float(self._config.get_learning_adv_params()['learning_rate'])
        self._fc_layer_params = tuple(self._config.get_learning_adv_params()['fc_layer_params'].split(','))

        print('Create and train a neural network agent')
        self._neural_network_agent = create_neural_network_agent(self._train_env,
                                                                 self._learning_rate,
                                                                 self._fc_layer_params)

        learning_params = self._config.get_learning_params()
        train_neural_network(self._neural_network_agent,
                             self._train_env,
                             self._eval_env,
                             num_train_iterations=learning_params['training_iterations'],
                             log_interval=learning_params['training_log_interval'],
                             eval_interval=learning_params['eval_interval'],
                             num_eval_episodes=learning_params['num_eval_episodes'],
                             replay_buffer_max_length=learning_params['replay_buffer_max_length'],
                             collect_steps_per_iteration=learning_params['collect_steps_per_iteration'],
                             output_folder=self._output_folder,
                             timestamp=self._timestamp)

    def get_random_baseline_performance(self, iterations=10):
        """
        Establish a baseline performance based on random behaviour
        :param iterations:
        :return:
        """
        random_policy = random_tf_policy.RandomTFPolicy(self._train_env.time_step_spec(),
                                                        self._train_env.action_spec())

        return compute_avg_return(self._train_env, random_policy, iterations)

    def run_agent(self, fps=2, random=False):
        """
        Run iterations.
        :param fps:
        Frames per second for video
        :param random:
        For random behaviour
        :return:
        """
        run_py_env = suite_gym.load(self._env_name)
        run_env = tf_py_environment.TFPyEnvironment(run_py_env)

        if not random:
            policy = self._neural_network_agent.policy
        else:
            policy = random_tf_policy.RandomTFPolicy(run_env.time_step_spec(),
                                                     run_env.action_spec())

        if self._num_demo_episodes > 0:
            if self._demo_video:
                filename = os.path.join(self._output_folder, self._timestamp + "-demonstration" + ".mp4")
                with imageio.get_writer(filename, fps=fps) as video:
                    for episode in range(self._num_demo_episodes):
                        print('Demonstration Episode: ', episode+1)
                        # Reset the evaluation environment
                        time_step = run_env.reset()
                        while not time_step.is_last():
                            action_step = policy.action(time_step)
                            time_step = run_env.step(action_step.action)
                            tf.print('ACTION: ', action_step.action, time_step)
                            video.append_data(run_py_env.render())
                print('Demonstration video is in: '+filename)
            else:
                for episode in range(self._num_demo_episodes):
                    print('Demonstration Episode: ', episode+1)
                    # Reset the evaluation environment
                    time_step = run_env.reset()
                    while not time_step.is_last():
                        action_step = policy.action(time_step)
                        time_step = run_env.step(action_step.action)
                        tf.print('ACTION: ', action_step.action, time_step)


def main():
    agent = QNetworkAgent('blobble-life-v0')
    agent.run_agent()


if __name__ == "__main__":
    main()
