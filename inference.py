"""
Inference script for Crop Disease Detection & Treatment Agent.
Uses the OpenAI Client to call an LLM for disease prediction and treatment recommendation.

Required environment variables:
    API_BASE_URL  — The API endpoint for the LLM.
    MODEL_NAME    — The model identifier to use for inference.
    HF_TOKEN      — Your Hugging Face / API key.
"""

import os
import json
import random
from openai import OpenAI
from grader import grade_easy, grade_medium, grade_hard

# --- Configuration from environment variables ---
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.groq.com/openai/v1")
MODEL_NAME = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

if not API_BASE_URL or not MODEL_NAME or not HF_TOKEN:
    raise RuntimeError(
        "Missing required environment variables. "
        "Please set API_BASE_URL, MODEL_NAME, and HF_TOKEN."
    )

# --- OpenAI Client setup ---
client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

SYSTEM_PROMPT = """You are an expert agricultural AI agent specializing in crop disease detection and treatment.
Given crop observations, you must analyze the symptoms and provide a diagnosis.

You will receive a JSON object with these fields:
- crop_type: the type of crop (tomato, potato, or rice)
- leaf_color: the color of the leaves (green, yellow, or brown)
- spots: whether spots are visible on the leaves (true or false)
- humidity: humidity percentage (0-100)
- temperature: temperature in Celsius (10-40)

You must respond with ONLY a valid JSON object (no markdown, no explanation) in the requested format.
"""


def obs_to_dict(obs) -> dict:
    """Convert a CropObservation model to a plain dict for the LLM."""
    if hasattr(obs, 'model_dump'):
        data = obs.model_dump()
        return {k: data[k] for k in ['crop_type', 'leaf_color', 'spots', 'humidity', 'temperature'] if k in data}
    if isinstance(obs, dict):
        return obs
    return {'crop_type': 'tomato', 'leaf_color': 'green', 'spots': False, 'humidity': 50, 'temperature': 25}


def rule_based_fallback(state: dict) -> str:
    """Fallback rule-based prediction when LLM call fails."""
    if state.get('leaf_color') == 'yellow':
        return 'leaf_blight'
    elif state.get('spots'):
        return 'rust'
    return 'healthy'


def llm_predict_binary(obs) -> str:
    """Use the LLM to predict whether the crop is healthy or diseased."""
    state = obs_to_dict(obs)
    try:
        user_prompt = (
            f"Observations: {json.dumps(state)}\n\n"
            "Task: Is this crop healthy or diseased?\n"
            "Respond with ONLY a JSON object: {\"prediction\": \"healthy\"} or {\"prediction\": \"diseased\"}"
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=50,
        )
        text = response.choices[0].message.content.strip()
        try:
            result = json.loads(text)
            prediction = result.get("prediction", "healthy")
            return prediction if prediction in ("healthy", "diseased") else "healthy"
        except json.JSONDecodeError:
            if "diseased" in text.lower():
                return "diseased"
            return "healthy"
    except Exception as e:
        print(f"  [Warning] LLM call failed for easy task: {e}")
        fallback = rule_based_fallback(state)
        return "healthy" if fallback == "healthy" else "diseased"


def llm_predict_disease(obs) -> str:
    """Use the LLM to predict the specific disease."""
    state = obs_to_dict(obs)
    try:
        user_prompt = (
            f"Observations: {json.dumps(state)}\n\n"
            "Task: What disease does this crop have?\n"
            "Respond with ONLY a JSON object: {\"disease\": \"<disease>\"}\n"
            "where <disease> is one of: healthy, leaf_blight, rust"
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=50,
        )
        text = response.choices[0].message.content.strip()
        valid_diseases = ("healthy", "leaf_blight", "rust")
        try:
            result = json.loads(text)
            disease = result.get("disease", "healthy")
            return disease if disease in valid_diseases else "healthy"
        except json.JSONDecodeError:
            for d in valid_diseases:
                if d in text.lower():
                    return d
            return "healthy"
    except Exception as e:
        print(f"  [Warning] LLM call failed for medium task: {e}")
        return rule_based_fallback(state)


def llm_predict_full(obs) -> dict:
    """Use the LLM to predict disease and recommend treatment."""
    state = obs_to_dict(obs)
    disease_to_treatment = {
        'healthy': 'none',
        'leaf_blight': 'pesticide_A',
        'rust': 'pesticide_B',
    }
    try:
        user_prompt = (
            f"Observations: {json.dumps(state)}\n\n"
            "Task: Diagnose the disease and recommend a treatment.\n"
            "Respond with ONLY a JSON object:\n"
            "{\"disease\": \"<disease>\", \"treatment\": \"<treatment>\"}\n"
            "where <disease> is one of: healthy, leaf_blight, rust\n"
            "and <treatment> is one of: none, pesticide_A, pesticide_B"
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=100,
        )
        text = response.choices[0].message.content.strip()
        valid_diseases = ("healthy", "leaf_blight", "rust")
        valid_treatments = ("none", "pesticide_A", "pesticide_B")
        try:
            result = json.loads(text)
            disease = result.get("disease", "healthy")
            treatment = result.get("treatment", "none")
            if disease not in valid_diseases:
                disease = "healthy"
            if treatment not in valid_treatments:
                treatment = "none"
            return {"disease": disease, "treatment": treatment}
        except json.JSONDecodeError:
            return {"disease": "healthy", "treatment": "none"}
    except Exception as e:
        print(f"  [Warning] LLM call failed for hard task: {e}")
        disease = rule_based_fallback(state)
        return {"disease": disease, "treatment": disease_to_treatment[disease]}


def run_inference():
    """Run all three tasks with the LLM agent and print scores."""
    random.seed(42)

    print("=" * 50)
    print("Crop Disease Detection — LLM Inference")
    print(f"Model: {MODEL_NAME}")
    print(f"API:   {API_BASE_URL}")
    print("=" * 50)

    easy_score = grade_easy(llm_predict_binary)
    print(f"\n[Easy Task]   Binary classification score: {easy_score:.4f}")

    medium_score = grade_medium(llm_predict_disease)
    print(f"[Medium Task] Disease classification score: {medium_score:.4f}")

    hard_score = grade_hard(llm_predict_full)
    print(f"[Hard Task]   Full pipeline score:          {hard_score:.4f}")

    overall = (easy_score + medium_score + hard_score) / 3
    print(f"\n{'=' * 50}")
    print(f"Overall Score: {overall:.4f}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    run_inference()
