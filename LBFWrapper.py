import numpy as np


class _StackedState:
    def __init__(self, keep_frame, axis, lstm=False):
        self.keep_frame = keep_frame
        self.axis = axis
        self.lstm = lstm
        self.stack = []

    def initiate(self, obj):
        self.stack = [obj] * self.keep_frame
        if self.lstm:
            return np.stack(self.stack, axis=self.axis)
        return np.concatenate(self.stack, axis=self.axis)


class LBFWrapper:
    def __init__(self, env, numFramesObs=3, lstm=False, time_limit=None, **kwargs):
        self.env = env
        self.lstm = lstm

        self.n_agents = getattr(self.env.unwrapped, 'n_agents', None)
        if self.n_agents is None:
            obss, _ = self.env.reset()
            self.n_agents = len(obss) if isinstance(obss, (list, tuple)) else 1

        action_spaces = getattr(self.env.action_space, 'spaces', self.env.action_space)
        try:
            if isinstance(action_spaces, (list, tuple)):
                self.num_action = int(max([space.n for space in action_spaces]))
            else:
                self.num_action = int(action_spaces.n)
        except Exception:
            self.num_action = 6

        obss, _ = self.env.reset()
        if not isinstance(obss, (list, tuple)):
            obss = [obss] * self.n_agents
        obs_dim = int(np.asarray(obss[0], dtype=np.float32).size)
        state_dim = obs_dim * self.n_agents

        ep_limit = time_limit
        if ep_limit is None:
            ep_limit = getattr(self.env.unwrapped, 'max_episode_steps', None)
        if ep_limit is None:
            spec = getattr(self.env, 'spec', None)
            ep_limit = getattr(spec, 'max_episode_steps', None) if spec is not None else None
        if ep_limit is None:
            ep_limit = 50

        self.env_info = {
            'n_agents': int(self.n_agents),
            'n_actions': int(self.num_action),
            'obs_shape': obs_dim,
            'state_shape': state_dim,
            'episode_limit': int(ep_limit),
        }
        self.stackedStates_obs = _StackedState(numFramesObs, 1, lstm)

    def reset(self):
        obss, _ = self.env.reset()
        if not isinstance(obss, (list, tuple)):
            obss = [obss] * self.n_agents
        obs_mat = np.vstack([np.asarray(o, dtype=np.float32).ravel() for o in obss])
        state = obs_mat.reshape(-1).astype(np.float32)

        action_id_oh = np.zeros([self.n_agents, self.num_action], dtype=np.float32)
        agent_id_oh = np.eye(self.n_agents, dtype=np.float32)
        obs = np.concatenate([obs_mat, action_id_oh, agent_id_oh], axis=1)
        return self.stackedStates_obs.initiate(obs), state

    def step(self, action, *args, **kwargs):
        action_arr = np.asarray(action, dtype=np.int64).reshape(-1)
        if action_arr.size == 1 and self.n_agents > 1:
            action_arr = np.repeat(action_arr, self.n_agents)
        if action_arr.size != self.n_agents:
            raise ValueError(
                "Expected {} actions, got {}.".format(self.n_agents, action_arr.size)
            )
        action_arr = np.clip(action_arr, 0, self.num_action - 1)

        step_result = self.env.step(action_arr.tolist())
        if len(step_result) == 5:
            obss, reward, terminated, truncated, info = step_result
        elif len(step_result) == 4:
            obss, reward, terminated, info = step_result
            truncated = False
        else:
            raise ValueError("Unexpected step() return format from LBF env.")

        if isinstance(reward, (list, tuple, np.ndarray)):
            reward = float(np.asarray(reward, dtype=np.float32).sum())
        else:
            reward = float(reward)

        term_arr = np.asarray(terminated, dtype=np.bool_).reshape(-1)
        trunc_arr = np.asarray(truncated, dtype=np.bool_).reshape(-1)
        if term_arr.size == 1:
            term_arr = np.repeat(term_arr, self.n_agents)
        if trunc_arr.size == 1:
            trunc_arr = np.repeat(trunc_arr, self.n_agents)
        if term_arr.size != self.n_agents or trunc_arr.size != self.n_agents:
            done_flag = bool(np.all(term_arr) or np.all(trunc_arr))
            done = np.asarray([done_flag] * self.n_agents)
        else:
            done = np.logical_or(term_arr, trunc_arr)

        if not isinstance(obss, (list, tuple)):
            obss = [obss] * self.n_agents
        obs_mat = np.vstack([np.asarray(o, dtype=np.float32).ravel() for o in obss])
        state = obs_mat.reshape(-1).astype(np.float32)

        action_id_oh = np.zeros([self.n_agents, self.num_action], np.float32)
        action_id_oh[np.arange(self.n_agents), action_arr] = 1
        agent_id_oh = np.eye(self.n_agents, dtype=np.float32)
        obs = np.concatenate([obs_mat, action_id_oh, agent_id_oh], axis=1)
        obss_stacked = self.stackedStates_obs.initiate(obs)

        if not isinstance(info, dict):
            info = {}
        info.setdefault('battle_won', False)
        info['valid_action'] = self.get_avail_actions()
        info['terminated'] = bool(np.all(done))

        return (obss_stacked, state), reward, done, info

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.num_action), dtype=np.float32)
