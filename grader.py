from env import CropEnv

NUM_EPISODES = 20


def grade_easy(agent):
    """
    Grade the agent on the easy task (binary classification).
    Runs multiple episodes and averages the score.

    Args:
        agent: Callable that takes CropObservation and returns 'healthy' or 'diseased'.

    Returns:
        float: Score between 0.0 and 1.0.
    """
    total_score = 0.0
    for _ in range(NUM_EPISODES):
        env = CropEnv()
        obs = env.reset()
        prediction = agent(obs)
        true_disease = env.true_disease

        if (prediction == 'healthy' and true_disease == 'healthy') or \
           (prediction == 'diseased' and true_disease != 'healthy'):
            reward = 0.4
        else:
            reward = -0.3

        total_score += (reward + 0.3) / 0.7

    return total_score / NUM_EPISODES


def grade_medium(agent):
    """
    Grade the agent on the medium task (multi-class disease classification).
    Runs multiple episodes and averages the score.

    Args:
        agent: Callable that takes CropObservation and returns 'healthy', 'leaf_blight', or 'rust'.

    Returns:
        float: Score between 0.0 and 1.0.
    """
    total_score = 0.0
    for _ in range(NUM_EPISODES):
        env = CropEnv()
        obs = env.reset()
        prediction = agent(obs)
        true_disease = env.true_disease

        reward = 0.4 if prediction == true_disease else -0.3
        total_score += (reward + 0.3) / 0.7

    return total_score / NUM_EPISODES


def grade_hard(agent):
    """
    Grade the agent on the hard task (full pipeline).
    Runs multiple episodes and averages the score.

    Args:
        agent: Callable that takes CropObservation and returns {'disease': str, 'treatment': str}.

    Returns:
        float: Score between 0.0 and 1.0.
    """
    total_score = 0.0
    for _ in range(NUM_EPISODES):
        env = CropEnv()
        obs = env.reset()
        prediction = agent(obs)
        true_disease = env.true_disease
        true_treatment = env.true_treatment

        reward = 0.0
        if prediction['disease'] == true_disease:
            reward += 0.4
        else:
            reward -= 0.3

        if prediction['treatment'] == true_treatment:
            reward += 0.4
        else:
            reward -= 0.3

        total_score += (reward + 0.6) / 1.4

    return total_score / NUM_EPISODES


def overall_score(agent):
    """
    Compute the overall score by averaging the three task scores.
    """
    score_easy = grade_easy(agent)
    score_medium = grade_medium(agent)
    score_hard = grade_hard(agent)
    return (score_easy + score_medium + score_hard) / 3
