import json
import time

import yaml
from gymnasium import Wrapper
from pogema_toolbox.run_episode import run_episode

from pogema import pogema_v0, GridConfig

from env.create_env import ProvideGlobalObstacles
from follower.inference import FollowerInferenceConfig, FollowerInference
from follower.preprocessing import follower_preprocessor

# from pypibt.inference import PIBTInferenceConfig, PIBTInference
# from show_speed import num_agents

with open('pogema-sps-speed-maps.yaml', 'r') as f:
    maps = yaml.safe_load(f)


class RuntimeMetricWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self._start_time = None
        self._env_step_time = None
        self._current_step = None

    def step(self, actions):
        env_step_start = time.monotonic()
        observations, rewards, terminated, truncated, infos = self.env.step(actions)
        env_step_end = time.monotonic()
        self._env_step_time += env_step_end - env_step_start
        self._current_step += 1
        if all(terminated) or all(truncated):
            final_time = time.monotonic() - self._start_time - self._env_step_time
            if 'metrics' not in infos[0]:
                infos[0]['metrics'] = {}
            infos[0]['metrics'].update(runtime=final_time)
            infos[0]['metrics'].update(EnvOPS=self._current_step * len(observations) / self._env_step_time)
            infos[0]['metrics'].update(EnvSPS=self._current_step / self._env_step_time)
            infos[0]['metrics'].update(AlgoOPS=self._current_step * len(observations) / final_time)
            infos[0]['metrics'].update(AlgoSPS=self._current_step / final_time)

        return observations, rewards, terminated, truncated, infos

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        self._start_time = time.monotonic()
        self._env_step_time = 0.0
        self._current_step = 0
        return obs


results = []
for num_agents in reversed([64, 128, 256, 512, 1024, 2048]):
    # for map_name in maps:
    for _ in range(3):
        map_name = "large-validation-mazes-seed-9"
        grid = maps[map_name]
        follower_cfg = FollowerInferenceConfig()
        algo = FollowerInference(follower_cfg)
        algo.reset_states()
        env = pogema_v0(
            GridConfig(map=grid, max_episode_steps=1024, map_name=map_name, num_agents=num_agents, on_target='restart',
                       observation_type='MAPF'))
        env = RuntimeMetricWrapper(env)
        env = ProvideGlobalObstacles(env)
        env = follower_preprocessor(env, follower_cfg)
        env.reset()
        # algo_cfg = PIBTInferenceConfig(device='cpu', num_process=8, centralized=True)
        # algo = PIBTInference(algo_cfg)
        metrics = run_episode(env, algo)
        print(metrics)
        # env.render()
        results.append({"num_agents": num_agents, "metrics": metrics})
with open('sps-results.json', 'w') as f:
    json.dump(results, f)
