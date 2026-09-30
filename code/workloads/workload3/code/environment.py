from __future__ import division
import gym
import numpy as np

if not hasattr(np, "bool8"):
    np.bool8 = np.bool_

from collections import deque
from gym.spaces.box import Box
from cv2 import resize, INTER_AREA


def process_frame(frame, conf):
    frame = frame[conf["crop1"]:conf["crop2"] + 160, :160]
    frame = resize(frame, (80, 80), interpolation=INTER_AREA)
    frame = (
        0.2989 * frame[:, :, 0]
        + 0.5870 * frame[:, :, 1]
        + 0.1140 * frame[:, :, 2]
    )
    frame = np.reshape(frame, [1, 80, 80]).astype(np.float32)
    return frame


class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super().__init__(env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == "NOOP"

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            rng = self.unwrapped.np_random
            if hasattr(rng, "integers"):
                noops = rng.integers(1, self.noop_max + 1)
            else:
                noops = rng.randint(1, self.noop_max + 1)

        assert noops > 0
        for _ in range(noops):
            obs, _, terminated, truncated, info = self.env.step(self.noop_action)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)

        return obs, info

    def step(self, action):
        return self.env.step(action)


class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        assert env.unwrapped.get_action_meanings()[1] == "FIRE"
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)

        obs, _, terminated, truncated, info = self.env.step(1)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        obs, _, terminated, truncated, info = self.env.step(2)
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)

        return obs, info

    def step(self, action):
        return self.env.step(action)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        real_done = terminated or truncated
        self.was_real_done = real_done

        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            truncated = True
        self.lives = lives

        return obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        if self.was_real_done:
            obs, info = self.env.reset(**kwargs)
        else:
            obs, _, terminated, truncated, info = self.env.step(0)
            if terminated or truncated:
                obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}

        for _ in range(self._skip):
            obs, reward, term, trunc, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if term or trunc:
                terminated = term
                truncated = trunc
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, terminated, truncated, info

    def reset(self, **kwargs):
        self._obs_buffer.clear()
        obs, info = self.env.reset(**kwargs)
        self._obs_buffer.append(obs)
        return obs, info


class AtariRescale(gym.ObservationWrapper):
    def __init__(self, env, env_conf):
        super().__init__(env)
        self.observation_space = Box(0.0, 1.0, [1, 80, 80], dtype=np.uint8)
        self.conf = env_conf

    def observation(self, observation):
        return process_frame(observation, self.conf)


class NormalizedEnv(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.state_mean = 0.0
        self.state_std = 0.0
        self.alpha = 0.9999
        self.num_steps = 0

    def observation(self, observation):
        self.num_steps += 1
        self.state_mean = self.state_mean * self.alpha + observation.mean() * (1.0 - self.alpha)
        self.state_std = self.state_std * self.alpha + observation.std() * (1.0 - self.alpha)

        unbiased_mean = self.state_mean / (1.0 - (self.alpha ** self.num_steps))
        unbiased_std = self.state_std / (1.0 - (self.alpha ** self.num_steps))

        return (observation - unbiased_mean) / (unbiased_std + 1e-8)


class OldGymAdapter(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            obs, info = result
            return obs
        return result

    def step(self, action):
        result = self.env.step(action)
        if isinstance(result, tuple):
            if len(result) == 4:
                return result
            if len(result) == 5:
                obs, reward, terminated, truncated, info = result
                done = bool(terminated) or bool(truncated)
                return obs, reward, done, info
        # fallback
        return result


def atari_env(env_id, env_conf, args):
    env = gym.make(env_id)

    if "NoFrameskip" in env_id:
        assert "NoFrameskip" in env.spec.id
        env._max_episode_steps = args.max_episode_length * args.skip_rate
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=args.skip_rate)
    else:
        env._max_episode_steps = args.max_episode_length

    env = EpisodicLifeEnv(env)

    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)

    env._max_episode_steps = args.max_episode_length
    env = AtariRescale(env, env_conf)
    env = NormalizedEnv(env)

    env = OldGymAdapter(env)

    return env

