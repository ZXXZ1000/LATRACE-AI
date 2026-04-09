from __future__ import annotations


def cluster_faces(faces, min_cluster_size=2, distance_threshold=0.5):
    import hdbscan
    import numpy as np

    face_embeddings = []
    face_types = []
    face_detection_scores = []
    face_quality_scores = []

    for face in faces:
        face_embeddings.append(face["face_emb"])
        face_types.append(face["extra_data"]["face_type"])
        face_detection_scores.append(float(face["extra_data"]["face_detection_score"]))
        face_quality_scores.append(float(face["extra_data"]["face_quality_score"]))

    face_embeddings = np.array(face_embeddings)
    detection_threshold = 0.8
    quality_threshold = 20
    good_mask = [
        face_detection_scores[i] >= detection_threshold and face_quality_scores[i] >= quality_threshold
        for i in range(len(face_types))
    ]

    all_labels = [-1] * len(face_types)
    if len(face_embeddings[good_mask]) >= min_cluster_size:
        good_embeddings = face_embeddings[good_mask]
        good_similarity = np.dot(good_embeddings, good_embeddings.T)
        good_distances = 1 - good_similarity
        good_distances = np.maximum(good_distances, 0).astype(np.float64)
        good_clusterer = hdbscan.HDBSCAN(
            min_cluster_size=min_cluster_size,
            metric="precomputed",
        )
        good_labels = good_clusterer.fit_predict(good_distances)
        good_idx = 0
        for index, is_good in enumerate(good_mask):
            if is_good:
                all_labels[index] = int(good_labels[good_idx])
                good_idx += 1

    result_faces = []
    for index, face in enumerate(faces):
        face_copy = face.copy()
        face_copy["cluster_id"] = all_labels[index]
        result_faces.append(face_copy)
    return result_faces


__all__ = ["cluster_faces"]
