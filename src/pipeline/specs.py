from dataclasses import dataclass

@dataclass
class ExperimentSpec:
    marker: str
    camera: str
    fps: float
    folder: str
    cosmed_path: str
    subject: str