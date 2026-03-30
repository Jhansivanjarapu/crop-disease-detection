import random


def _to_dict(state):
    """Convert Pydantic Observation model or dict to a plain dict."""
    if hasattr(state, 'model_dump'):
        return state.model_dump()
    return state


class DiagnosisAgent:
    """
    Agent responsible for diagnosing crop disease from observed state.
    Uses rule-based logic on leaf color, spots, humidity, and temperature.
    """

    def __init__(self):
        self.name = "DiagnosisAgent"

    def run(self, state):
        """
        Analyze crop state and return predicted disease.

        Args:
            state: Observation model or dict with crop conditions.

        Returns:
            str: Predicted disease ('healthy', 'leaf_blight', 'rust').
        """
        state = _to_dict(state)
        leaf_color = state['leaf_color']
        spots = state['spots']
        humidity = state['humidity']
        temperature = state['temperature']

        # Primary indicators
        if leaf_color == 'yellow' and humidity > 60:
            return 'leaf_blight'
        if leaf_color == 'yellow':
            return 'leaf_blight'
        if spots and temperature > 25:
            return 'rust'
        if spots:
            return 'rust'

        # Secondary: high humidity + brown leaves could indicate early blight
        if leaf_color == 'brown' and humidity > 80:
            return 'leaf_blight'

        return 'healthy'


class ConfidenceAgent:
    """
    Agent responsible for estimating the confidence of a diagnosis.
    Evaluates how strongly the observed symptoms support the predicted disease.
    """

    def __init__(self):
        self.name = "ConfidenceAgent"

    def run(self, state, disease):
        """
        Compute a confidence score for the given diagnosis.

        Args:
            state: Observation model or dict with crop conditions.
            disease: str, the predicted disease.

        Returns:
            float: Confidence score between 0.0 and 1.0.
        """
        state = _to_dict(state)
        supporting_factors = 0
        total_factors = 4  # leaf_color, spots, humidity, temperature

        if disease == 'leaf_blight':
            if state['leaf_color'] == 'yellow':
                supporting_factors += 1
            if state['humidity'] > 70:
                supporting_factors += 1
            if state['temperature'] > 25:
                supporting_factors += 1
            if not state['spots']:
                supporting_factors += 1  # blight typically no spots

        elif disease == 'rust':
            if state['spots']:
                supporting_factors += 1
            if state['leaf_color'] in ('yellow', 'brown'):
                supporting_factors += 1
            if state['humidity'] > 60:
                supporting_factors += 1
            if state['temperature'] > 20:
                supporting_factors += 1

        elif disease == 'healthy':
            if state['leaf_color'] == 'green':
                supporting_factors += 1
            if not state['spots']:
                supporting_factors += 1
            if 40 <= state['humidity'] <= 70:
                supporting_factors += 1
            if 20 <= state['temperature'] <= 30:
                supporting_factors += 1

        base_confidence = supporting_factors / total_factors
        # Add small random variation to simulate real-world uncertainty
        noise = random.uniform(-0.05, 0.05)
        confidence = max(0.1, min(1.0, base_confidence + noise))
        return round(confidence, 2)


class ExplanationAgent:
    """
    Agent responsible for generating a human-readable explanation
    of why a particular disease was diagnosed.
    """

    def __init__(self):
        self.name = "ExplanationAgent"

    def run(self, state, disease):
        """
        Generate an explanation for the diagnosis.

        Args:
            state: Observation model or dict with crop conditions.
            disease: str, the predicted disease.

        Returns:
            str: Human-readable explanation.
        """
        state = _to_dict(state)
        observations = []

        if state['leaf_color'] == 'yellow':
            observations.append("yellowing of leaves")
        elif state['leaf_color'] == 'brown':
            observations.append("browning of leaves")
        else:
            observations.append("healthy green leaves")

        if state['spots']:
            observations.append("visible spots on leaves")
        else:
            observations.append("no visible spots")

        if state['humidity'] > 70:
            observations.append(f"high humidity ({state['humidity']}%)")
        elif state['humidity'] < 30:
            observations.append(f"low humidity ({state['humidity']}%)")

        if state['temperature'] > 30:
            observations.append(f"high temperature ({state['temperature']}°C)")
        elif state['temperature'] < 15:
            observations.append(f"low temperature ({state['temperature']}°C)")

        obs_text = ", ".join(observations)
        disease_name = disease.replace('_', ' ').title()

        if disease == 'healthy':
            return (
                f"Based on observations ({obs_text}), the {state['crop_type']} crop "
                f"appears healthy. No disease indicators were found."
            )
        else:
            return (
                f"Based on observations ({obs_text}), the {state['crop_type']} crop "
                f"is likely affected by {disease_name}."
            )


class TreatmentAgent:
    """
    Agent responsible for recommending treatment based on the diagnosed disease.
    Considers crop type for tailored recommendations.
    """

    TREATMENTS = {
        'healthy': {
            'default': 'No treatment needed. Continue regular monitoring and maintenance.',
        },
        'leaf_blight': {
            'tomato': 'Apply copper-based fungicide spray twice a week for 2 weeks. Remove affected leaves. Ensure proper plant spacing for air circulation.',
            'potato': 'Apply Mancozeb fungicide every 7 days. Remove and destroy infected foliage. Avoid overhead irrigation.',
            'rice': 'Apply Tricyclazole fungicide at recommended doses. Drain excess water from fields. Ensure balanced nitrogen fertilization.',
            'default': 'Apply fungicide spray (Pesticide A) twice a week for 2 weeks. Remove affected plant parts.',
        },
        'rust': {
            'tomato': 'Apply systemic fungicide (Propiconazole) every 10 days. Remove severely infected leaves. Improve air circulation.',
            'potato': 'Apply Tebuconazole fungicide every 10-14 days. Avoid working with wet plants. Remove volunteer plants.',
            'rice': 'Apply Propiconazole at early infection stage. Maintain proper water management. Use resistant varieties if available.',
            'default': 'Apply systemic fungicide (Pesticide B) every 10 days. Monitor progress and reapply as needed.',
        },
    }

    def __init__(self):
        self.name = "TreatmentAgent"

    def run(self, disease, crop_type='default'):
        """
        Recommend treatment for the given disease and crop type.

        Args:
            disease: str, the diagnosed disease.
            crop_type: str, the type of crop (for tailored advice).

        Returns:
            str: Treatment recommendation.
        """
        disease_treatments = self.TREATMENTS.get(disease, self.TREATMENTS['healthy'])
        treatment = disease_treatments.get(crop_type, disease_treatments['default'])
        return treatment


class PipelineAgent:
    """
    Orchestrator agent that coordinates all specialist agents to produce
    a complete diagnosis report: disease prediction, confidence, explanation,
    and treatment recommendation.
    """

    def __init__(self):
        self.name = "PipelineAgent"
        self.diagnosis_agent = DiagnosisAgent()
        self.confidence_agent = ConfidenceAgent()
        self.explanation_agent = ExplanationAgent()
        self.treatment_agent = TreatmentAgent()

    def run(self, state):
        """
        Run the full diagnostic pipeline.

        Args:
            state: Observation model or dict with crop conditions.

        Returns:
            dict with keys: disease, confidence, explanation, treatment, agent_trace.
        """
        state = _to_dict(state)
        # Step 1: DiagnosisAgent predicts the disease
        disease = self.diagnosis_agent.run(state)

        # Step 2: ConfidenceAgent evaluates confidence
        confidence = self.confidence_agent.run(state, disease)

        # Step 3: ExplanationAgent generates reasoning
        explanation = self.explanation_agent.run(state, disease)

        # Step 4: TreatmentAgent recommends treatment
        treatment = self.treatment_agent.run(disease, state.get('crop_type', 'default'))

        # Build agent trace for transparency
        agent_trace = (
            f"[{self.diagnosis_agent.name}] Predicted: {disease}\n"
            f"[{self.confidence_agent.name}] Confidence: {confidence}\n"
            f"[{self.explanation_agent.name}] Generated explanation\n"
            f"[{self.treatment_agent.name}] Recommended treatment"
        )

        return {
            'disease': disease,
            'confidence': confidence,
            'explanation': explanation,
            'treatment': treatment,
            'agent_trace': agent_trace,
        }
