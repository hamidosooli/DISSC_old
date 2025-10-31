import numpy as np

class SMACWrapper:
    def __init__(self, env, numFramesObs=3, numFramesState=1, lstm=False, **kwargs):
        self.env = env
        self.env_info = env.get_env_info()

        self.num_agent = self.env_info['n_agents']
        self.num_action = self.env_info['n_actions']

        self.lstm=lstm
        self.stackedStates_obs = Stacked_state(numFramesObs, 1, lstm)
        #self.stackedStates_states = Stacked_state(numFramesState, 1, False)

    def reset(self):
        self.env.reset()

        # Initiate
        state = self.env.get_state().astype(np.float32)
        obs = np.vstack(self.env.get_obs()).astype(np.float32)

        # Add action one-hot and agent one-hot
        action_id_oh = np.zeros([self.num_agent, self.num_action], dtype=np.float32)
        agent_id_oh = np.eye(self.num_agent, dtype=np.float32)
        obs = np.concatenate([obs, action_id_oh, agent_id_oh], axis=1)

        return self.stackedStates_obs.initiate(obs), state#self.stackedStates_states.initiate(state)

    def step(self,action,*args,**kwargs):
        reward, terminated, info = self.env.step(action)
        validActions = self.get_avail_actions()

        # Find 'done' forr each agent
        if terminated:
            done = np.asarray([terminated]*self.env_info['n_agents'])
        else:
            done=[]
            for validAction in validActions:
                if validAction[0] == 1 and np.all(validAction[1:] == 0):
                    done.append(True)
                else:
                    done.append(False)
            done = np.asarray(done)

        # Update observation and state
        states = self.env.get_state().astype(np.float32)
        #states = self.stackedStates_states(state)

        obs = np.vstack(self.env.get_obs()).astype(np.float32)
        action_id_oh = np.zeros([self.num_agent, self.num_action], np.float32)
        action_id_oh[np.arange(self.num_agent), np.asarray(action)] = 1
        agent_id_oh = np.eye(self.num_agent, dtype=np.float32)
        obs = np.concatenate([obs, action_id_oh, agent_id_oh], axis=1)
        obss = self.stackedStates_obs.initiate(obs)

        # Update Information
        if "battle_won" not in info:
            info["battle_won"]=False
        info['valid_action'] = validActions
        info['terminated'] = terminated

        return (obss, states), reward, done, info

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.env_info['n_agents']):
            avail_actions.append(self.env.get_avail_agent_actions(agent_id))
        return np.vstack(avail_actions)

    def save_replay(self):
        self.env.save_replay()


class Stacked_state:
    def __init__(self, keep_frame, axis,lstm=False):
        self.keep_frame = keep_frame
        self.axis = axis
        self.lstm=lstm
        self.stack = []

    def initiate(self, obj):
        self.stack = [obj] * self.keep_frame
        if self.lstm:
            return np.stack(self.stack, axis=self.axis)
        else:
            return np.concatenate(self.stack, axis=self.axis)


class SMACV2Wrapper:
    def __init__(self, env, numFramesObs=3, numFramesState=1, lstm=False, **kwargs):
        self.env = env
        # SMACv2 provides get_env_info() with similar keys; fall back if needed
        self.env_info = self.env.get_env_info()

        self.num_agent = self.env_info['n_agents']
        self.num_action = self.env_info['n_actions']

        self.lstm = lstm
        self.stackedStates_obs = Stacked_state(numFramesObs, 1, lstm)

    def reset(self):
        self.env.reset()

        state = self.env.get_state().astype(np.float32)
        obs = np.vstack(self.env.get_obs()).astype(np.float32)

        action_id_oh = np.zeros([self.num_agent, self.num_action], dtype=np.float32)
        agent_id_oh = np.eye(self.num_agent, dtype=np.float32)
        obs = np.concatenate([obs, action_id_oh, agent_id_oh], axis=1)

        return self.stackedStates_obs.initiate(obs), state

    def step(self, action, *args, **kwargs):
        # SMACv2 step returns (obs, reward, terminated, truncated, info)
        result = self.env.step(action)
        if len(result) == 5:
            _, reward, terminated, truncated, info = result
            terminated = bool(terminated or truncated)
        else:
            # fallback to old API just in case
            reward, terminated, info = result

        # Aggregate per-agent rewards if provided
        if isinstance(reward, (list, tuple, np.ndarray)):
            reward = float(np.sum(reward))
        else:
            reward = float(reward)

        validActions = self.get_avail_actions()

        if terminated:
            done = np.asarray([terminated] * self.env_info['n_agents'])
        else:
            done = []
            for validAction in validActions:
                if validAction[0] == 1 and np.all(validAction[1:] == 0):
                    done.append(True)
                else:
                    done.append(False)
            done = np.asarray(done)

        states = self.env.get_state().astype(np.float32)
        obs = np.vstack(self.env.get_obs()).astype(np.float32)
        action_id_oh = np.zeros([self.num_agent, self.num_action], np.float32)
        action_id_oh[np.arange(self.num_agent), np.asarray(action)] = 1
        agent_id_oh = np.eye(self.num_agent, dtype=np.float32)
        obs = np.concatenate([obs, action_id_oh, agent_id_oh], axis=1)
        obss = self.stackedStates_obs.initiate(obs)

        if "battle_won" not in info:
            info["battle_won"] = False
        info['valid_action'] = validActions
        info['terminated'] = terminated

        return (obss, states), reward, done, info

    def get_avail_actions(self):
        # Prefer SMACv2's batch API if present; else per-agent API
        if hasattr(self.env, 'get_avail_actions'):
            avail = self.env.get_avail_actions()
            return np.asarray(avail)
        avail_actions = []
        for agent_id in range(self.env_info['n_agents']):
            avail_actions.append(self.env.get_avail_agent_actions(agent_id))
        return np.vstack(avail_actions)

    def save_replay(self):
        self.env.save_replay()


class VMASWrapper:
    def __init__(self, env, numFramesObs=3, lstm=False, **kwargs):
        self.env = env
        self.lstm = lstm

        # Infer basic info
        try:
            self.n_agents = self.env.unwrapped.n_agents
        except Exception:
            # Fallback: infer from reset
            obss, _ = self.env.reset()
            self.n_agents = len(obss) if isinstance(obss, (list, tuple)) else 1

        # We discretize to 5 cardinal actions as common in VMAS setups
        self.num_action = 5

        # One-step probe to get obs dim
        try:
            obss, _ = self.env.reset()
        except Exception:
            obss = [np.zeros((1,), dtype=np.float32) for _ in range(self.n_agents)]
        if isinstance(obss, (list, tuple)):
            obs_dim = int(np.asarray(obss[0], dtype=np.float32).size)
        else:
            obs_dim = int(np.asarray(obss, dtype=np.float32).size)

        # Compose env_info to match expected keys
        self.env_info = {
            'n_agents': self.n_agents,
            'n_actions': self.num_action,
            'obs_shape': obs_dim + self.num_action + self.n_agents,
            'state_shape': obs_dim * self.n_agents,
            'episode_limit': getattr(self.env.unwrapped, 'max_steps', 200),
        }

        self.stackedStates_obs = Stacked_state(numFramesObs, 1, lstm)

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
        obss, rews, term, trunc, info = self.env.step(action)
        terminated = bool(term or trunc)

        # Scalarize/aggregate reward to a team reward
        if isinstance(rews, (list, tuple, np.ndarray)):
            reward = float(np.asarray(rews, dtype=np.float32).sum())
        else:
            reward = float(rews)

        # Build availability mask: all actions available
        validActions = np.ones((self.n_agents, self.num_action), dtype=np.float32)

        # Per-agent done flags
        if terminated:
            done = np.asarray([True] * self.n_agents)
        else:
            done = np.asarray([False] * self.n_agents)

        # Next obs/state
        if not isinstance(obss, (list, tuple)):
            obss = [obss] * self.n_agents
        obs_mat = np.vstack([np.asarray(o, dtype=np.float32).ravel() for o in obss])
        state = obs_mat.reshape(-1).astype(np.float32)

        action_id_oh = np.zeros([self.n_agents, self.num_action], np.float32)
        action_id_oh[np.arange(self.n_agents), np.asarray(action)] = 1
        agent_id_oh = np.eye(self.n_agents, dtype=np.float32)
        obs = np.concatenate([obs_mat, action_id_oh, agent_id_oh], axis=1)
        obss_stacked = self.stackedStates_obs(obs)

        if not isinstance(info, dict):
            info = {}
        if 'battle_won' not in info:
            info['battle_won'] = False
        info['valid_action'] = validActions
        info['terminated'] = terminated

        return (obss_stacked, state), reward, done, info

    def get_avail_actions(self):
        return np.ones((self.n_agents, self.num_action), dtype=np.float32)


class SMACliteWrapper:
    def __init__(self, env, numFramesObs=3, lstm=False, time_limit=None, **kwargs):
        self.env = env
        self.lstm = lstm

        # Basic info from smaclite
        self.n_agents = getattr(self.env.unwrapped, 'n_agents', None)
        if self.n_agents is None:
            obss, _ = self.env.reset()
            self.n_agents = len(obss) if isinstance(obss, (list, tuple)) else 1

        # Determine max action space size across agents
        try:
            action_spaces = getattr(self.env, 'action_space', None)
            if isinstance(action_spaces, (list, tuple)):
                self.num_action = int(max([sp.n for sp in action_spaces]))
            else:
                self.num_action = int(action_spaces.n)
        except Exception:
            self.num_action = 10

        # Probe obs/state dims
        try:
            obss = self.env.unwrapped.get_obs()
        except Exception:
            obss, _ = self.env.reset()
        if isinstance(obss, (list, tuple)):
            obs_dim = int(np.asarray(obss[0], dtype=np.float32).size)
        else:
            obs_dim = int(np.asarray(obss, dtype=np.float32).size)
        try:
            state = self.env.unwrapped.get_state()
            state_dim = int(np.asarray(state, dtype=np.float32).size)
        except Exception:
            state_dim = obs_dim * self.n_agents

        ep_limit = getattr(self.env.unwrapped, 'time_limit', None)
        if ep_limit is None:
            ep_limit = getattr(self.env, 'spec', None).max_episode_steps if getattr(self.env, 'spec', None) is not None else 200
        self.env_info = {
            'n_agents': self.n_agents,
            'n_actions': self.num_action,
            'obs_shape': obs_dim + self.num_action + self.n_agents,
            'state_shape': state_dim,
            'episode_limit': int(ep_limit),
        }

        self.stackedStates_obs = Stacked_state(numFramesObs, 1, lstm)

    def reset(self):
        obss, _ = self.env.reset()
        if not isinstance(obss, (list, tuple)):
            obss = [obss] * self.n_agents
        obs_mat = np.vstack([np.asarray(o, dtype=np.float32).ravel() for o in obss])
        try:
            state = np.asarray(self.env.unwrapped.get_state(), dtype=np.float32).reshape(-1)
        except Exception:
            state = obs_mat.reshape(-1).astype(np.float32)

        action_id_oh = np.zeros([self.n_agents, self.num_action], dtype=np.float32)
        agent_id_oh = np.eye(self.n_agents, dtype=np.float32)
        obs = np.concatenate([obs_mat, action_id_oh, agent_id_oh], axis=1)
        return self.stackedStates_obs.initiate(obs), state

    def step(self, action, *args, **kwargs):
        obss, reward, terminated, truncated, info = self.env.step(action)
        done_flag = bool(terminated or truncated)

        # Team reward scalar
        if isinstance(reward, (list, tuple, np.ndarray)):
            reward = float(np.asarray(reward, dtype=np.float32).sum())
        else:
            reward = float(reward)

        # Per-agent done
        if done_flag:
            done = np.asarray([True] * self.n_agents)
        else:
            # Derive from avail actions if needed
            done = []
            validActions = self.get_avail_actions()
            for validAction in validActions:
                if validAction[0] == 1 and np.all(validAction[1:] == 0):
                    done.append(True)
                else:
                    done.append(False)
            done = np.asarray(done)

        # Next obs/state
        if not isinstance(obss, (list, tuple)):
            obss = [obss] * self.n_agents
        obs_mat = np.vstack([np.asarray(o, dtype=np.float32).ravel() for o in obss])
        try:
            state = np.asarray(self.env.unwrapped.get_state(), dtype=np.float32).reshape(-1)
        except Exception:
            state = obs_mat.reshape(-1).astype(np.float32)

        action_id_oh = np.zeros([self.n_agents, self.num_action], np.float32)
        action_id_oh[np.arange(self.n_agents), np.asarray(action)] = 1
        agent_id_oh = np.eye(self.n_agents, dtype=np.float32)
        obs = np.concatenate([obs_mat, action_id_oh, agent_id_oh], axis=1)
        obss_stacked = self.stackedStates_obs(obs)

        # Info
        if not isinstance(info, dict):
            info = {}
        if 'battle_won' not in info:
            info['battle_won'] = False
        info['valid_action'] = self.get_avail_actions()
        info['terminated'] = done_flag

        return (obss_stacked, state), reward, done, info

    def get_avail_actions(self):
        try:
            avail = self.env.unwrapped.get_avail_actions()
            return np.asarray(avail, dtype=np.float32)
        except Exception:
            return np.ones((self.n_agents, self.num_action), dtype=np.float32)

    def __call__(self, obj=None):
        if obj is None:
            if self.lstm:
                return np.stack(self.stack, axis=self.axis)
            else:
                return np.concatenate(self.stack, axis=self.axis)
        self.stack.append(obj)
        self.stack.pop(0)
        if self.lstm:
            return np.stack(self.stack, axis=self.axis)
        else:
            return np.concatenate(self.stack, axis=self.axis)
