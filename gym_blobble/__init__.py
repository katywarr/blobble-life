from gym.envs.registration import register

register(
    id='blobble-life-v0',
    entry_point='gym_blobble.envs:BlobbleEnv',
)