"""
Credits to https://github.com/openai/baselines/blob/master/baselines/common/atari_wrappers.py
"""

from typing import Tuple, List

import gym
from gym import wrappers
import numpy as np
from PIL import Image
import carla_gym
from mmengine import Config


def make_atari(id, size=64, max_episode_steps=None, noop_max=30, frame_skip=4, done_on_life_loss=False, clip_reward=False):
    env = gym.make(id)
    assert 'NoFrameskip' in env.spec.id or 'Frameskip' not in env.spec
    env = ResizeObsWrapper(env, (size, size))
    if clip_reward:
        env = RewardClippingWrapper(env)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    if noop_max is not None:
        env = NoopResetEnv(env, noop_max=noop_max)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    if done_on_life_loss:
        env = EpisodicLifeEnv(env)
    return env

def make_carla(id, size=64, max_episode_steps=None, noop_max=30, frame_skip=4, done_on_life_loss=False, clip_reward=False):
    # NOTE noop_max and done_on_life_loss are ignored for carla env
    cfg = Config.fromfile("src/envs/carla_config.py")
    params = cfg.env
    params["connection"] = dict(
        host = 'localhost',
        port = 2100 + id * 10,
        tm_port = 8100 + id * 10
    )
    env = gym.make("carla-v0", **params)
    env = CarlaObsWrapper(env)
    env = ResizeObsWrapper(env, (size, size))
    if clip_reward:
        env = RewardClippingWrapper(env)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = MaxAndSkipEnv(env, skip=frame_skip)
    return env


def make_multi_agent_carla(id, size=64, max_episode_steps=None, noop_max=30, frame_skip=2, done_on_life_loss=False, clip_reward=False):
    # NOTE noop_max and done_on_life_loss are ignored for multi agent carla env
    cfg = Config.fromfile("src/envs/carla_config.py")
    params = cfg.env
    params["connection"] = dict(
        host = 'localhost',
        port = 2100 + id * 10,
        tm_port = 8100 + id * 10
    )
    # env = gym.make("carla-ma-cbv-v0", **params)
    env = gym.make("carla-ma-v0", **params)
    env = CarlaMultiAgentObsWrapper(env)
    env = MultiAgentResizeObsWrapper(env, (size, size))
    if clip_reward:
        env = MultiAgentRewardClippingWrapper(env)
    if max_episode_steps is not None:
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_episode_steps)
    env = MultiAgentMaxAndSkipEnv(env, skip=frame_skip)
    return env


class CarlaObsWrapper(gym.ObservationWrapper):
    # Extracts the BEV array from the dict of observations returned by carla_gym
    def __init__(self, env: gym.Env) -> None:
        gym.ObservationWrapper.__init__(self, env)
        assert(len(env.observation_space) == 1)
        self.observation_space = env.observation_space[0]

    def observation(self, observation):
        return observation["bev"][0]
    
    
class CarlaMultiAgentObsWrapper(gym.ObservationWrapper):
    # Extracts the BEV array from the dict of observations returned by carla_gym
    def __init__(self, env: gym.Env) -> None:
        gym.ObservationWrapper.__init__(self, env)
        assert(len(env.observation_space) == 1)
        self.observation_space = env.observation_space[0]

    def observation(self, observation):
        return np.array([obs["bev"][0] for obs in observation])


class ResizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size: Tuple[int, int]):
        gym.ObservationWrapper.__init__(self, env)
        self.size = tuple(size)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)
        self.unwrapped.original_obs = None

    def resize(self, obs: np.ndarray):
        img = Image.fromarray(obs)
        img = img.resize(self.size, Image.BILINEAR)
        return np.array(img)

    def observation(self, observation: np.ndarray) -> np.ndarray:
        self.unwrapped.original_obs = observation
        return self.resize(observation)
    
    
class MultiAgentResizeObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, size: Tuple[int, int]):
        gym.ObservationWrapper.__init__(self, env)
        self.size = tuple(size)
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(size[0], size[1], 3), dtype=np.uint8)
        self.unwrapped.original_obs = None
    
    def resize(self, obs: List[np.ndarray]):
        imgs = [Image.fromarray(ego_obs) for ego_obs in obs]
        imgs = [np.asarray(img.resize(self.size, Image.BILINEAR)) for img in imgs]
        return np.array(imgs)

    def observation(self, observation: List[np.ndarray]):
        self.unwrapped.original_obs = observation
        return self.resize(observation)


class RewardClippingWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return np.sign(reward)

    
class MultiAgentRewardClippingWrapper(gym.RewardWrapper):
    def reward(self, reward):
        return [np.sign(agent_reward) for agent_reward in reward]


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        assert skip > 0
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        self.max_frame = np.zeros(env.observation_space.shape, dtype=np.uint8)

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None
        for i in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            if i == self._skip - 2:
                self._obs_buffer[0] = obs
            if i == self._skip - 1:
                self._obs_buffer[1] = obs
            total_reward += reward
            if done:
                break
        # Note that the observation on the done=True frame
        # doesn't matter
        self.max_frame = self._obs_buffer.max(axis=0)

        return self.max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    

class MultiAgentMaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=2):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        assert skip > 0
        # most recent raw observations (for max pooling across time steps)
        # self._obs_buffer = np.zeros((2, env.n_agents) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip
        # self.max_frame = np.zeros((env.n_agents, *env.observation_space.shape), dtype=np.uint8)

    def step(self, action):  # TODO: Maxing observations in combination with respawning may not make sense.
        """Repeat action, sum reward, and max over last 2 observations."""
        total_rewards = np.zeros(self.env.n_agents)
        dones = [None] * self.env.n_agents 
        
        for _ in range(self._skip - 1):
            _, rs, dones, info = self.env.step_without_obs(action)  # TODO: May be a problem on env done
            total_rewards += rs
            
        ego_obs, rs, dones, info = self.env.step(action)  # Just step, combining/maxing frames happens inside sensor.
        total_rewards += rs
        
        
        # for i in range(steps_without_obs, self._skip):
        #     ego_obs, rs, dones, info = self.env.step(action)
        #     if i == self._skip - 2:
        #         self._obs_buffer[0] = ego_obs
        #     if i == self._skip - 1:
        #         self._obs_buffer[1] = ego_obs
        #     total_rewards += rs
        #     if np.all(dones):
        #         break
        # Note that the observation on the done=True frame
        # doesn't matter
        # self.max_frame = self._obs_buffer.max(axis=0)

        return ego_obs, total_rewards, dones, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
