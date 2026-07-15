import numpy as np


def get_formation_offsets(
    formation_name: str,
    spacing: float = 0.6,
) -> np.ndarray:
    """
    Return formation offsets relative to the swarm centroid.

    Supported formations:
    - triangle
    - line
    """

    if formation_name == "triangle":
        return np.array(
            [
                [spacing, 0.0, 0.0],
                [-spacing / 2.0, -spacing * 0.866, 0.0],
                [-spacing / 2.0, spacing * 0.866, 0.0],
            ],
            dtype=np.float32,
        )

    if formation_name == "line":
        return np.array(
            [
                [spacing, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [-spacing, 0.0, 0.0],
            ],
            dtype=np.float32,
        )

    raise ValueError(
        f"Unknown formation: {formation_name}. "
        "Supported formations are 'triangle' and 'line'."
    )


def interpolate_formations(
    start_offsets: np.ndarray,
    target_offsets: np.ndarray,
    progress: float,
) -> np.ndarray:
    """
    Smoothly interpolate between two formations.

    progress:
        0.0 = fully start formation
        1.0 = fully target formation
    """

    progress = float(np.clip(progress, 0.0, 1.0))

    return (
        (1.0 - progress) * start_offsets
        + progress * target_offsets
    ).astype(np.float32)