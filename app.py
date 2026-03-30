import gradio as gr
from openenv.core.env_server import create_fastapi_app
from env import CropEnv
from models import CropAction, CropObservation
from agents import PipelineAgent

# --- Create FastAPI app using OpenEnv's create_fastapi_app ---

app = create_fastapi_app(
    env=CropEnv,
    action_cls=CropAction,
    observation_cls=CropObservation,
)

pipeline_agent = PipelineAgent()


# --- Gradio UI ---

def predict(crop_type, leaf_color, spots, humidity, temperature):
    """Run the full agent pipeline on the given crop state."""
    state = {
        'crop_type': crop_type,
        'leaf_color': leaf_color,
        'spots': spots,
        'humidity': humidity,
        'temperature': temperature,
    }
    result = pipeline_agent.run(state)
    disease_text = (
        f"Predicted Disease: {result['disease'].replace('_', ' ').title()} "
        f"(Confidence: {result['confidence']:.2f})"
    )
    return disease_text, result['explanation'], result['treatment'], result['agent_trace']


def generate_random():
    """Generate a random crop case using CropEnv."""
    random_env = CropEnv()
    obs = random_env.reset()
    return obs.crop_type, obs.leaf_color, obs.spots, obs.humidity, obs.temperature


with gr.Blocks(title="Crop Disease Detection & Treatment Agent") as demo:
    gr.Markdown("# Crop Disease Detection & Treatment Agent")
    gr.Markdown(
        "A multi-agent system for agricultural disease detection and treatment recommendation.\n\n"
        "**Agents:**\n"
        "- **DiagnosisAgent** — Analyzes symptoms and predicts the disease\n"
        "- **ConfidenceAgent** — Estimates how confident the diagnosis is\n"
        "- **ExplanationAgent** — Generates human-readable reasoning\n"
        "- **TreatmentAgent** — Recommends crop-specific treatment\n"
        "- **PipelineAgent** — Orchestrates all agents in sequence"
    )

    with gr.Group():
        gr.Markdown("### Input Section")
        with gr.Row():
            crop_type = gr.Dropdown(['tomato', 'potato', 'rice'], label="Crop Type", value="tomato")
            leaf_color = gr.Dropdown(['green', 'yellow', 'brown'], label="Leaf Color", value="green")
            spots = gr.Checkbox(label="Spots", value=False)
        with gr.Row():
            humidity = gr.Slider(0, 100, label="Humidity (%)", value=50)
            temperature = gr.Slider(10, 40, label="Temperature (°C)", value=25)
        random_btn = gr.Button("Generate Random Case")

    predict_btn = gr.Button("Predict Disease & Treatment", variant="primary")

    with gr.Group():
        gr.Markdown("### Prediction Results")
        disease_out = gr.Textbox(label="Disease Prediction", interactive=False)
        explanation_out = gr.Textbox(label="Explanation", interactive=False)
        treatment_out = gr.Textbox(label="Recommended Treatment", interactive=False)

    with gr.Accordion("Agent Trace (pipeline details)", open=False):
        agent_trace_out = gr.Textbox(label="Agent Execution Trace", interactive=False, lines=5)

    random_btn.click(generate_random, outputs=[crop_type, leaf_color, spots, humidity, temperature])
    predict_btn.click(
        predict,
        inputs=[crop_type, leaf_color, spots, humidity, temperature],
        outputs=[disease_out, explanation_out, treatment_out, agent_trace_out],
    )

# Mount Gradio inside OpenEnv's FastAPI app
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
