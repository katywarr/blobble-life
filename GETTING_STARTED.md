# Running blobble-life 

## Setting up the environment

From within an Anaconda command prompt, navigate to this cloned repository to create 
a virtual environment.

```
conda env create -f blobble-life-gpu.yml
conda activate blobble-life
```

For non-gpu execution, reinstall tensorflow:

```
conda install tensorflow
```

The AI Gym environment must be built and installed.

From the cloned repo folder:

```
pip install -e .
```

Then, within python (or notebook):

```
import gym
gym.make('gym_blobble:blobble-life-v0')
```

See details on creating AI Gym environments [here](https://github.com/openai/gym/blob/master/docs/creating-environments.md).

## Configuration

The training environment is configured through a single configuration file ```blobble_config.ini```. This file enables 
configuration of the training hyper-parameters, how the output is presented, and the fully-connected neural network 
layers. There are comments in the example configuration to explain.

The config file must be located in the same folder as you run the agent.

(To switch on/off taste and smell, edit the global boolean values at the top of ```envs/blobble_env.py```).

## Running the code

Just run ```LearningBlobble.py```.



 





