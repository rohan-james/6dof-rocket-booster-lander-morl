from gymnasium.envs.registration import register

from environments.morl_lander import RocketLanding6DOFEnv

register(id="RocketBooster-capql-v0", entry_point="envs:RocketLanding6DOFEnv")
