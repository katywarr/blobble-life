from environment.BlobbleEnv_v1 import BlobbleEnv
import imageio as imageio
import numpy as np


def create_blobble_video(video_filename, num_episodes=1, fps=30):
    filename = video_filename + ".mp4"

    blobble_env = BlobbleEnv()

    with imageio.get_writer(filename, fps=fps) as video:
        for i in range(num_episodes-1):
            print(i)
            blobble_env.reset([0, 0, 10])
            done = False
            while not done:
                action = np.random.randint(8)
                print('Next Action: ', action)
                observation, reward, done, _ = blobble_env.step(action)
                # blobble_env.render_print()
                video.append_data(blobble_env.render(mode='rgb_array'))


def main():
    create_blobble_video('blobble_random', num_episodes=10, fps=30)


if __name__ == "__main__":
    main()
