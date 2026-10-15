"""Every pipeline C2 can run. Add new pipeline classes to PIPELINE_TYPES."""

from pipelines.apriltag.pipeline import AprilTagPipeline
from pipelines.pipeline import Pipeline

PIPELINE_TYPES: list[type[Pipeline]] = [
    AprilTagPipeline,
]
