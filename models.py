from typing import Literal, Optional
from pydantic import Field
from openenv.core.env_server import Action, Observation, State


class CropAction(Action):
    """Action model for the Crop Disease environment."""
    action: str = Field(
        description=(
            "Action to take. One of: "
            "predict_healthy, predict_leaf_blight, predict_rust, "
            "suggest_none, suggest_pesticide_A, suggest_pesticide_B"
        )
    )


class CropObservation(Observation):
    """Observation model for the Crop Disease environment."""
    crop_type: str = Field(default="tomato", description="Type of crop being observed")
    leaf_color: str = Field(default="green", description="Color of the crop leaves")
    spots: bool = Field(default=False, description="Whether spots are visible on the leaves")
    humidity: int = Field(default=50, ge=0, le=100, description="Humidity level (0-100)")
    temperature: int = Field(default=25, ge=10, le=40, description="Temperature in Celsius (10-40)")


class CropState(State):
    """State model for the Crop Disease environment."""
    crop_type: str = Field(default="tomato", description="Type of crop")
    leaf_color: str = Field(default="green", description="Leaf color")
    spots: bool = Field(default=False, description="Spots present")
    humidity: int = Field(default=50, description="Humidity level")
    temperature: int = Field(default=25, description="Temperature")
    true_disease: str = Field(default="healthy", description="Hidden ground truth disease")
    true_treatment: str = Field(default="none", description="Hidden ground truth treatment")
