from gym_blobble.envs import BlobbleEnv
import imageio as imageio
import numpy as np
import tensorflow as tf

from tf_agents.environments import suite_gym
from tf_agents.environments import tf_py_environment
from tf_agents.eval import metric_utils
from tf_agents.metrics import tf_metrics
from tf_agents.networks import q_network
from tf_agents.policies import random_tf_policy
from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.trajectories import trajectory
from tf_agents.utils import common
from tf_agents.agents.dqn import dqn_agent
from tf_agents.drivers import dynamic_step_driver

from PIL import Image

num_iterations = 100


def collect_step(environment, policy, buffer):
    print('collect step')
    time_step = environment.current_time_step()
    action_step = policy.action(time_step)
    next_time_step = environment.step(action_step.action)
    traj = trajectory.from_transition(time_step, action_step, next_time_step)

    print(traj)
    # Add trajectory to the replay buffer
    buffer.add_batch(traj)


def collect_data(env, policy, buffer, steps):
    print('collecting data for ', steps, ' steps')
    for _ in range(steps):
        collect_step(env, policy, buffer)


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
        for i in range(num_episodes - 1):
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
    for _ in range(num_episodes):

        time_step = environment.reset()
        episode_return = 0.0

        while not time_step.is_last():
            action_step = policy.action(time_step)
            time_step = environment.step(action_step.action)
            episode_return += time_step.reward
        total_return += episode_return

    avg_return = total_return / num_episodes
    return avg_return.numpy()[0]


def train_agent(agent, train_env, eval_env,
                dataset_iterator,
                replay_buffer,
                num_eval_episodes=10,
                log_interval=200,
                eval_interval=1000,
                collect_steps_per_iteration=1):
    # (Optional) Optimize by wrapping some of the code in a graph using TF function.
    agent.train = common.function(agent.train)

    # Reset the train step
    agent.train_step_counter.assign(0)

    # Evaluate the agent's policy once before training.
    avg_return = compute_avg_return(eval_env, agent.policy, num_eval_episodes)
    returns = [avg_return]

    for _ in range(num_iterations):

        # Collect a few steps using collect_policy and save to the replay buffer.
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


def create_neural_network_agent(env):
    # Create a neural network that can learn to predict the QValues (expected returns) for all the possible
    # Blobble actions, given a specific observation
    fc_layer_params = (100,)

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


'''
Each policy returns an action (0...8).
    When the policy uses a learned model, the policy will depend on the predictions from the model.
    Create two policies:
    - One is for the evaluation and deployment (the agent.policy)
    - One is for the data collction (the agent.collect_policy)
'''


def get_policies_for_agent(self, agent):
    eval_policy = agent.policy
    collect_policy = agent.collect_policy
    return eval_policy, collect_policy


class BlobbleAgent():

    def __init__(self, env_name='blobble-world-v0'):
        self._env_name = env_name

        # Create training and evaluation environments
        train_py_env = suite_gym.load(self._env_name)
        eval_py_env = suite_gym.load(self._env_name)

        # Convert the training and test environments to Tensors
        self._train_env = tf_py_environment.TFPyEnvironment(train_py_env)
        self._eval_env = tf_py_environment.TFPyEnvironment(eval_py_env)
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

        self._training_agent = create_neural_network_agent(self._train_env)

    '''
    get_baseline_performance
    
    Establish a baseline performance based on a random agent
    '''

    def get_baseline_performance(self, iterations=100):
        random_policy = random_tf_policy.RandomTFPolicy(self._eval_env.time_step_spec(),
                                                        self._eval_env.action_spec())

        return compute_avg_return(self._eval_env, random_policy, iterations)

    def execute_random_policy(self, steps, replay_buffer_max_length=10):
        random_policy = random_tf_policy.RandomTFPolicy(self._eval_env.time_step_spec(),
                                                        self._eval_env.action_spec())

        '''
        collect_data_spec returns a trajectory spec
        A trajectory spec is a tuple that contains:
        - the state of the environment in some time step (observation)
        - the action that the agent should take in that state (action)
        - the state in which the environment will be after the action is taken
        '''
        replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
            data_spec=self._training_agent.collect_data_spec,
            batch_size=self._train_env.batch_size,
            max_length=replay_buffer_max_length)

        collect_data(self._train_env, random_policy, replay_buffer, steps=steps)

        '''
        The agent requires access to the replay buffer.
        
        '''
        buffer_it = iter(replay_buffer.as_dataset())
        for i in buffer_it:
            print(i)


def main():
    blobble_agent = BlobbleAgent('CartPole-v0')

    print('Baseline Performance is: ', blobble_agent.get_baseline_performance())

    blobble_agent.execute_random_policy(5)



    '''
    time_step = env.reset()
    # Image.fromarray(env.render(mode='rgb_array')).show()

    print('Time step:')
    print(time_step)
    action = np.array(2, dtype=np.int32)
    next_time_step = env.step(action)
    print('Next time step:')
    print(next_time_step)
    action = np.array(1, dtype=np.int32)
    next_time_step = env.step(action)
    print('Next time step:')
    print(next_time_step)
    env.render()
    # create_blobble_video('blobble_random', num_episodes=0, fps=30)


    '''

    '''
    Let's test the baseline performance of the environment
  
    random_policy = random_tf_policy.RandomTFPolicy(train_env.time_step_spec(),
                                                    train_env.action_spec())
    #example_environment = tf_py_environment.TFPyEnvironment(
    #    suite_gym.load('env_name'))
    #time_step = example_environment.reset()
 #   random_policy.action(time_step)
#

    # compute_avg_return(eval_env, random_policy, 10)

    replay_buffer_max_length = 100000
    replay_buffer = tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=train_env.batch_size,
        max_length=replay_buffer_max_length)

    #agent.collect_data_spec
    #agent.collect_data_spec._fields
  '''


if __name__ == "__main__":
    main()
