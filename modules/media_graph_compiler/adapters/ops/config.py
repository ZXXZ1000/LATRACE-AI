from __future__ import annotations


def get_processing_config() -> dict:
    """Local defaults migrated from memorization_agent processing config."""
    return {
        "cluster_size": 8,
        "face_cluster_min_size": 3,
        "face_cluster_distance_threshold": 0.5,
        "face_detection_score_threshold": 0.85,
        "face_quality_score_threshold": 20.0,
        "min_duration_for_audio": 2,
        "max_retries": 10,
    }


__all__ = ["get_processing_config"]
