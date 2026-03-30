import random
from grader import grade_easy, grade_medium, grade_hard


class BaselineAgent:
    """
    Simple rule-based agent for crop disease detection.
    Logic: If leaf_color is yellow -> predict leaf_blight
           If spots -> predict rust
           Else -> healthy
    """

    def __call__(self, obs):
        """
        Predict disease based on observation.

        Args:
            obs: CropObservation model or dict with crop conditions.

        Returns:
            str: Predicted disease ('healthy', 'leaf_blight', 'rust').
        """
        if hasattr(obs, 'model_dump'):
            obs = obs.model_dump()

        if obs['leaf_color'] == 'yellow':
            return 'leaf_blight'
        elif obs['spots']:
            return 'rust'
        else:
            return 'healthy'


def run_all_tasks():
    """
    Run all tasks with the baseline agent and print scores.
    """
    random.seed(42)

    agent = BaselineAgent()

    # Easy task: binary classification
    easy_score = grade_easy(lambda s: 'healthy' if agent(s) == 'healthy' else 'diseased')
    print(f"Easy score: {easy_score:.4f}")

    # Medium task: multi-class disease classification
    medium_score = grade_medium(agent)
    print(f"Medium score: {medium_score:.4f}")

    # Hard task: full pipeline
    disease_to_treatment = {
        'healthy': 'none',
        'leaf_blight': 'pesticide_A',
        'rust': 'pesticide_B',
    }
    hard_score = grade_hard(lambda s: {'disease': agent(s), 'treatment': disease_to_treatment[agent(s)]})
    print(f"Hard score: {hard_score:.4f}")

    overall = (easy_score + medium_score + hard_score) / 3
    print(f"Overall score: {overall:.4f}")


if __name__ == "__main__":
    run_all_tasks()
