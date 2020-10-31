#!/usr/bin/env python3

import gym
import numpy as np

from .base import Wrapper


class ActionSpaceScaler(Wrapper):

    """
    Scales the action space to be in the range (-clip, clip).

    Adapted from Vitchyr Pong's RLkit:
    https://github.com/vitchyr/rlkit/blob/master/rlkit/envs/wrappers.py#L41
    """

    def __init__(self, env, clip=1.0):
        print("im updated version")
        super(ActionSpaceScaler, self).__init__(env)
        self.env = env
        self.clip = clip
        ub = np.ones(self.env.action_space.shape) * clip
        self.action_space = gym.spaces.Box(-1 * ub, ub, dtype=np.float32)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)

    def _normalize(self, action):
        #lb = self.env.action_space.low
        #ub = self.env.action_space.high
        lb = 0
        ub = 1
        scaled_action = lb + (action + self.clip) * 0.5 * (ub - lb)
        scaled_action = np.clip(scaled_action, lb, ub)
        return scaled_action

    def step(self, action):
        if self.is_vectorized:
            action = [self._normalize(a) for a in action]
        else:
            action = self._normalize(action)
        print("actual action: ", action)
        #return self.env.step(action)
        state, reward, done, _ = self.env.step(action)
        print("actual reward: ", reward)
        if done:
            print("done")
        return state, reward, done, _
