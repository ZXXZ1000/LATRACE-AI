from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager

from modules.media_graph_compiler.adapters.ops.config import get_processing_config
from modules.media_graph_compiler.adapters.ops.face_clustering import cluster_faces
from modules.media_graph_compiler.adapters.ops.face_extraction import extract_faces

os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("ORT_LOG_SEVERITY_LEVEL", "4")
os.environ.setdefault("INSIGHTFACE_LOG_LEVEL", "ERROR")
os.environ.setdefault("INSIGHTFACE_SUPPRESS_VERBOSE", "1")

logger = logging.getLogger(__name__)
processing_config = get_processing_config()
cluster_size = int(processing_config.get("cluster_size", 8))
cluster_min_size = int(processing_config.get("face_cluster_min_size", 3))
cluster_distance_threshold = float(processing_config.get("face_cluster_distance_threshold", 0.5))

_FACE_APP = None
_FACE_APP_CONFIG_KEY = None


@contextmanager
def _suppress_native_output():
    import os as _os

    try:
        devnull = open(_os.devnull, "w")
        old_out, old_err = _os.dup(1), _os.dup(2)
        _os.dup2(devnull.fileno(), 1)
        _os.dup2(devnull.fileno(), 2)
        try:
            yield
        finally:
            _os.dup2(old_out, 1)
            _os.dup2(old_err, 2)
            _os.close(old_out)
            _os.close(old_err)
            devnull.close()
    except Exception:
        yield


def _ensure_face_app():
    global _FACE_APP, _FACE_APP_CONFIG_KEY
    try:
        from insightface.app import FaceAnalysis  # type: ignore
    except Exception as exc:
        logger.warning("face_processing: failed to import FaceAnalysis, fallback to no-op. err=%s", exc)
        _FACE_APP = None
        _FACE_APP_CONFIG_KEY = None
        return None

    providers_env = os.getenv("INSIGHTFACE_PROVIDERS")
    if providers_env:
        providers = tuple(p.strip() for p in providers_env.split(",") if p.strip())
    else:
        providers = ("CoreMLExecutionProvider", "CPUExecutionProvider")

    allowed_modules_env = os.getenv("MGC_FACE_ALLOWED_MODULES")
    if allowed_modules_env is not None:
        allowed_modules = tuple(
            module.strip() for module in allowed_modules_env.split(",") if module.strip()
        ) or None
    else:
        # We only need bbox detection + identity embedding on the default mainline.
        allowed_modules = ("detection", "recognition")

    config_key = (providers, allowed_modules)
    if _FACE_APP is not None and _FACE_APP_CONFIG_KEY == config_key:
        return _FACE_APP
    try:
        import onnxruntime as _ort  # type: ignore

        try:
            _ort.set_default_logger_severity(4)
        except Exception:
            pass
    except Exception:
        pass
    try:
        logging.getLogger("insightface").setLevel(logging.ERROR)
        logging.getLogger("onnxruntime").setLevel(logging.ERROR)
    except Exception:
        pass
    try:
        with _suppress_native_output():
            app = FaceAnalysis(
                name="buffalo_l",
                providers=list(providers),
                allowed_modules=list(allowed_modules) if allowed_modules is not None else None,
            )
            try:
                app.prepare(ctx_id=-1, providers=list(providers))
            except TypeError:
                app.prepare(ctx_id=-1)
        _FACE_APP = app
        _FACE_APP_CONFIG_KEY = config_key
        return _FACE_APP
    except Exception as exc:
        logger.warning("face_processing: failed to initialize FaceAnalysis, fallback to no-op. err=%s", exc)
        _FACE_APP = None
        _FACE_APP_CONFIG_KEY = None
        return None


class Face:
    def __init__(self, frame_id, bounding_box, face_emb, cluster_id, extra_data):
        self.frame_id = frame_id
        self.bounding_box = bounding_box
        self.face_emb = face_emb
        self.cluster_id = cluster_id
        self.extra_data = extra_data


def get_face(frames):
    app = _ensure_face_app()
    if app is None:
        return []
    extracted_faces = extract_faces(app, frames)
    return [
        Face(
            frame_id=f["frame_id"],
            bounding_box=f["bounding_box"],
            face_emb=f["face_emb"],
            cluster_id=f["cluster_id"],
            extra_data=f["extra_data"],
        )
        for f in extracted_faces
    ]


def cluster_face(faces):
    faces_json = [
        {
            "frame_id": f.frame_id,
            "bounding_box": f.bounding_box,
            "face_emb": f.face_emb,
            "cluster_id": f.cluster_id,
            "extra_data": f.extra_data,
        }
        for f in faces
    ]
    clustered_faces = cluster_faces(
        faces_json,
        int(cluster_min_size),
        float(cluster_distance_threshold),
    )
    return [
        Face(
            frame_id=f["frame_id"],
            bounding_box=f["bounding_box"],
            face_emb=f["face_emb"],
            cluster_id=f["cluster_id"],
            extra_data=f["extra_data"],
        )
        for f in clustered_faces
    ]


def process_faces(
    video_graph,
    base64_frames,
    save_path,
    preprocessing=None,
    *,
    segments=None,
    stride: int = 1,
    max_frames_per_segment: int = 0,
):
    preprocessing = preprocessing or []
    batch_size = max(len(base64_frames) // cluster_size, 4) if len(base64_frames) > 0 else 4

    def _process_batch(params):
        frames = params[0]
        offset = params[1]
        faces = get_face(frames)
        for face in faces:
            face.frame_id += offset
        return faces

    def get_embeddings(base64_frames, batch_size):
        num_batches = (len(base64_frames) + batch_size - 1) // batch_size
        batched_frames = [
            (base64_frames[i * batch_size : (i + 1) * batch_size], i * batch_size)
            for i in range(num_batches)
        ]
        faces = []
        with ThreadPoolExecutor(max_workers=num_batches) as executor:
            for batch_faces in executor.map(_process_batch, batched_frames):
                faces.extend(batch_faces)
        return cluster_face(faces)

    def _select_indices_by_segments(all_len: int, segments_list, stride_val: int, max_per_seg: int):
        selected = []
        for seg in (segments_list or []):
            fs = int(seg.get("frame_start", 0))
            fe = int(seg.get("frame_end", -1))
            if fe < fs:
                continue
            idxs = list(range(fs, fe + 1, max(1, int(stride_val or 1))))
            if max_per_seg and max_per_seg > 0 and len(idxs) > max_per_seg:
                step = max(1, int(len(idxs) / max_per_seg))
                idxs = idxs[::step][:max_per_seg]
            selected.extend(idxs)
        seen = set()
        out = []
        for index in selected:
            if 0 <= index < all_len and index not in seen:
                out.append(index)
                seen.add(index)
        return out

    def establish_mapping(faces, key="cluster_id", filter=None):
        mapping = {}
        for face in faces:
            if key not in face.keys():
                raise ValueError(f"key {key} not found in faces")
            if filter and not filter(face):
                continue
            face_id = face[key]
            mapping.setdefault(face_id, []).append(face)
        for face_id in mapping:
            mapping[face_id] = sorted(
                mapping[face_id],
                key=lambda item: (
                    float(item["extra_data"]["face_detection_score"]),
                    float(item["extra_data"]["face_quality_score"]),
                ),
                reverse=True,
            )
        return mapping

    def filter_score_based(face):
        dthresh = processing_config["face_detection_score_threshold"]
        qthresh = processing_config["face_quality_score_threshold"]
        return (
            float(face["extra_data"]["face_detection_score"]) > dthresh
            and float(face["extra_data"]["face_quality_score"]) > qthresh
        )

    def update_mapping(tempid2faces):
        if video_graph is None:
            id2faces = {}
            for tempid, faces in tempid2faces.items():
                if tempid == -1 or len(faces) == 0:
                    continue
                track_id = f"face_{int(tempid) + 1}"
                for face in faces:
                    face["matched_node"] = track_id
                id2faces[track_id] = list(faces)
            return id2faces

        id2faces = {}
        for tempid, faces in tempid2faces.items():
            if tempid == -1 or len(faces) == 0:
                continue
            face_info = {
                "embeddings": [face["face_emb"] for face in faces],
                "contents": [face["extra_data"]["face_base64"] for face in faces],
            }
            matched_nodes = video_graph.search_img_nodes(face_info)
            if len(matched_nodes) > 0:
                matched_node = matched_nodes[0][0]
                video_graph.update_node(matched_node, face_info)
            else:
                matched_node = video_graph.add_img_node(face_info)
            for face in faces:
                face["matched_node"] = matched_node
            id2faces.setdefault(matched_node, []).extend(faces)
        for face_id, faces in id2faces.items():
            id2faces[face_id] = sorted(
                faces,
                key=lambda item: (
                    float(item["extra_data"]["face_detection_score"]),
                    float(item["extra_data"]["face_quality_score"]),
                ),
                reverse=True,
            )
        return id2faces

    faces_json = None
    legacy_empty = False
    try:
        with open(save_path, "r") as f:
            payload = json.load(f)
        if isinstance(payload, dict):
            faces_json = list(payload.get("faces") or [])
        elif isinstance(payload, list):
            faces_json = list(payload)
            legacy_empty = len(faces_json) == 0
        else:
            faces_json = []
            legacy_empty = True
    except Exception:
        faces_json = None

    if faces_json is None or legacy_empty:
        app = _ensure_face_app()
        if app is None:
            return {}
        if segments or (stride and stride > 1) or (max_frames_per_segment and max_frames_per_segment > 0):
            idxs = _select_indices_by_segments(len(base64_frames), segments or [], stride, max_frames_per_segment)
            pairs = [(i, base64_frames[i]) for i in idxs]
            faces = get_face(pairs)
            faces = cluster_face(faces)
        else:
            faces = get_embeddings(base64_frames, batch_size)

        faces_json = [
            {
                "frame_id": face.frame_id,
                "bounding_box": face.bounding_box,
                "face_emb": face.face_emb,
                "cluster_id": int(face.cluster_id),
                "extra_data": face.extra_data,
            }
            for face in faces
        ]
        dir_path = os.path.dirname(save_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump({"schema": "faces_cache_v1", "faces": faces_json}, f)

    if "face" in preprocessing or len(faces_json) == 0:
        return {}

    tempid2faces = establish_mapping(faces_json, key="cluster_id", filter=filter_score_based)
    if len(tempid2faces) == 0:
        return {}
    return update_mapping(tempid2faces)


__all__ = ["process_faces"]
