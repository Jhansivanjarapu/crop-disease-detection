from env import CropEnv


class EasyTask:
    """
    Easy Task: Binary classification (healthy vs diseased).
    """

    def run(self, agent):
        env = CropEnv()
        obs = env.reset()
        return agent(obs)


class MediumTask:
    """
    Medium Task: Multi-class disease classification.
    """

    def run(self, agent):
        env = CropEnv()
        obs = env.reset()
        return agent(obs)


class HardTask:
    """
    Hard Task: Full pipeline - disease detection and treatment suggestion.
    """

    def run(self, agent):
        env = CropEnv()
        obs = env.reset()
        return agent(obs)
