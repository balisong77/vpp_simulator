import gymnasium as gym
import envs.VPP

if __name__ == "__main__":
    env = gym.make('vpp-simulator', max_episode_steps=500)
    env.reset()
    state, reward, terminated, truncated, info = env.step(32)
    env.render()