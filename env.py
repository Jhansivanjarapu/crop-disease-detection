import random
from typing import Optional, Any
from openenv.core.env_server import Environment
from models import CropAction, CropObservation, CropState


class CropEnv(Environment[CropAction, CropObservation, CropState]):
    """
    Simulated environment for Crop Disease Detection & Treatment.
    Extends the OpenEnv Environment base class with step()/reset()/state API.
    """

    DISEASES = ['healthy', 'leaf_blight', 'rust']
    DISEASE_TO_TREATMENT = {
        'healthy': 'none',
        'leaf_blight': 'pesticide_A',
        'rust': 'pesticide_B',
    }

    def __init__(self):
        super().__init__()
        self._crop_type = "tomato"
        self._leaf_color = "green"
        self._spots = False
        self._humidity = 50
        self._temperature = 25
        self.true_disease = "healthy"
        self.true_treatment = "none"
        self._step_count = 0
        self._episode_id = None

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        **kwargs: Any,
    ) -> CropObservation:
        """Reset the environment to a new random state."""
        if seed is not None:
            random.seed(seed)

        self._episode_id = episode_id
        self._step_count = 0

        self._crop_type = random.choice(['tomato', 'potato', 'rice'])
        self._leaf_color = random.choice(['green', 'yellow', 'brown'])
        self._spots = random.choice([True, False])
        self._humidity = random.randint(0, 100)
        self._temperature = random.randint(10, 40)

        self.true_disease = random.choice(self.DISEASES)
        self.true_treatment = self.DISEASE_TO_TREATMENT[self.true_disease]

        return CropObservation(
            crop_type=self._crop_type,
            leaf_color=self._leaf_color,
            spots=self._spots,
            humidity=self._humidity,
            temperature=self._temperature,
            done=False,
            reward=0.0,
        )

    def step(
        self,
        action: CropAction,
        timeout_s: Optional[float] = None,
        **kwargs: Any,
    ) -> CropObservation:
        """Take an action and return observation with reward."""
        self._step_count += 1
        reward = 0.0
        act = action.action

        if act.startswith('predict_'):
            predicted_disease = act.replace('predict_', '')
            if predicted_disease == self.true_disease:
                reward += 0.4
            else:
                reward -= 0.3

        elif act.startswith('suggest_'):
            suggested_treatment = act.replace('suggest_', '')
            if suggested_treatment == self.true_treatment:
                reward += 0.4
            else:
                reward -= 0.3

        return CropObservation(
            crop_type=self._crop_type,
            leaf_color=self._leaf_color,
            spots=self._spots,
            humidity=self._humidity,
            temperature=self._temperature,
            done=True,
            reward=reward,
        )

    @property
    def state(self) -> CropState:
        """Get the current environment state."""
        return CropState(
            crop_type=self._crop_type,
            leaf_color=self._leaf_color,
            spots=self._spots,
            humidity=self._humidity,
            temperature=self._temperature,
            true_disease=self.true_disease,
            true_treatment=self.true_treatment,
            step_count=self._step_count,
            episode_id=self._episode_id,
        )
