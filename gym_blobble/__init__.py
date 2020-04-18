from gym.envs.registration import register

register(
    id='blobble-world-v0',
    entry_point='gym_blobble.envs:BlobbleEnv',
)