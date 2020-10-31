#!/usr/bin/env python3

import cherry as ch
from cherry._utils import _min_size, _istensorable
from .base import Wrapper
from .utils import is_vectorized

from collections.abc import Iterable
from skimage.color import rgb2gray
from skimage.transform import rescale, resize, downscale_local_mean

import torch

import sys
sys.path.append("/content/cs285_f2020/cs285-final-project") # Adds higher directory to python modules path.
from infrastructure import pytorch_util as ptu

def flatten_episodes(replay, episodes, num_workers):
    """
    TODO: This implementation is not efficient.

    NOTE: Additional info (other than a transition's default fields) is simply copied.
    To know from which worker the data was gathered, you can access sars.runner_id
    TODO: This is not great. What is the best behaviour with infos here ?
    """
    flat_replay = ch.ExperienceReplay()
    worker_replays = [ch.ExperienceReplay() for w in range(num_workers)]
    flat_episodes = 0
    for sars in replay:
        state = sars.state.view(_min_size(sars.state))
        action = sars.action.view(_min_size(sars.action))
        reward = sars.reward.view(_min_size(sars.reward))
        next_state = sars.next_state.view(_min_size(sars.next_state))
        done = sars.done.view(_min_size(sars.done))
        fields = set(sars._Transition__fields) - {'state', 'action', 'reward', 'next_state', 'done'}
        infos = {f: getattr(sars, f) for f in fields}
        for worker in range(num_workers):
            infos['runner_id'] = worker
            # The following attemps to split additional infos. (WIP. Remove ?)
            # infos = {}
            # for f in fields:
            #     value = getattr(sars, f)
            #     if isinstance(value, Iterable) and len(value) == num_workers:
            #         value = value[worker]
            #     elif _istensorable(value):
            #         tvalue = ch.totensor(value)
            #         tvalue = tvalue.view(_min_size(tvalue))
            #         if tvalue.size(0) == num_workers:
            #             value = tvalue[worker]
            #     infos[f] = value
            worker_replays[worker].append(state[worker],
                                          action[worker],
                                          reward[worker],
                                          next_state[worker],
                                          done[worker],
                                          **infos,
                                          )
            if bool(done[worker]):
                flat_replay += worker_replays[worker]
                worker_replays[worker] = ch.ExperienceReplay()
                flat_episodes += 1
            if flat_episodes >= episodes:
                break
        if flat_episodes >= episodes:
            break
    return flat_replay

# device = None


# def init_gpu(use_gpu=True, gpu_id=0):
#     global device
#     if torch.cuda.is_available() and use_gpu:
#         device = torch.device("cuda:" + str(gpu_id))
#         print("Using GPU id {}".format(gpu_id))
#     else:
#         device = torch.device("cpu")
#         print("GPU not detected. Defaulting to CPU.")


# def set_device(gpu_id):
#     torch.cuda.set_device(gpu_id)


# def from_numpy(*args, **kwargs):
#     print("moving to device: ", device)
#     return torch.from_numpy(*args, **kwargs).float().to(device)


# def to_numpy(tensor):
#     return tensor.to('cpu').detach().numpy()


class Runner(Wrapper):

    """
    Runner wrapper.

    TODO: When is_vectorized and using episodes=n, use the parallel
    environmnents to sample n episodes, and stack them inside a flat replay.
    """

    def __init__(self, env):
        super(Runner, self).__init__(env)
        self.env = env
        self._needs_reset = True
        self._current_state = None
        #init_gpu()

    def full_obs_to_smol_boi(self, state):
        state = ptu.to_numpy(state)
        R = state[0,:,:,0]
        G = state[0,:,:,1]
        B = state[0,:,:,2]
        gray = 0.2125 * R + 0.7154 * G + 0.0721 * B
        image_rescaled = rescale(gray, 0.125, anti_aliasing=False)
        image_rescaled = ptu.from_numpy(image_rescaled)
        image_rescaled = image_rescaled.float().unsqueeze(0)
        #image_rescaled = image_rescaled.add(1.0)
       # print("image_rescaled", image_rescaled)
        print("image_rescaled.max", image_rescaled.max())
        print("image_rescaled.mean", image_rescaled.mean())
        #print("image_rescaled.shape", image_rescaled.shape)
        return image_rescaled
        
    def reset(self, *args, **kwargs):
        self._current_state = self.full_obs_to_smol_boi(self.env.reset(*args, **kwargs))
        self._needs_reset = False
        return self._current_state

    def step(self, action, *args, **kwargs):
        # TODO: Implement it to be compatible with .run()
        raise NotImplementedError('Runner does not currently support step.')

    def run(self,
            get_action,
            steps=None,
            episodes=None,
            render=False):
        """
        Runner wrapper's run method.
        """
        
        if steps is None:
            steps = float('inf')
            if self.is_vectorized:
                self._needs_reset = True
        elif episodes is None:
            episodes = float('inf')
        else:
            msg = 'Either steps or episodes should be set.'
            raise Exception(msg)

        steps = 1000

        replay = ch.ExperienceReplay()
        collected_episodes = 0
        collected_steps = 0
        while True:
            print("collected_steps", collected_steps)
            if collected_steps >= steps or collected_episodes >= episodes:
                if self.is_vectorized and collected_episodes >= episodes:
                    replay = flatten_episodes(replay, episodes, self.num_envs)
                    self._needs_reset = True
                return replay
            if self._needs_reset:
                self.reset()
            info = {}
            action = get_action(self._current_state)
            print("action", action)

            if isinstance(action, tuple):
                skip_unpack = False
                if self.is_vectorized:
                    if len(action) > 2:
                        skip_unpack = True
                    elif len(action) == 2 and \
                            self.env.num_envs == 2 and \
                            not isinstance(action[1], dict):
                        # action[1] is not info but an action
                        action = (action, )

                if not skip_unpack:
                    if len(action) == 2:
                        info = action[1]
                        action = action[0]
                    elif len(action) == 1:
                        action = action[0]
                    else:
                        msg = 'get_action should return 1 or 2 values.'
                        raise NotImplementedError(msg)
            old_state = self._current_state
            state, reward, done, _ = self.env.step(action)
            #print("reward: ", reward)
            #print("state.shape", state.shape)
            #print("state", state)
            #state = rgb2gray(state)
            state = self.full_obs_to_smol_boi(state)
            #reward = reward.to(ptu.get_device())
            #print("INNER LOOPS")
            #print(state)
            #print(reward)
            # print("gray.shape", gray.shape)
            # print("gray", gray)
            # if collected_steps >= 0:
            #     collected_episodes += 1
            #     self._needs_reset = True
            if not self.is_vectorized and done:
                collected_episodes += 1
                self._needs_reset = True
            elif self.is_vectorized:
                collected_episodes += sum(done)
            replay.append(old_state, action, reward, state, done, **info)
            self._current_state = state
            if render:
                self.env.render()
            collected_steps += 1
