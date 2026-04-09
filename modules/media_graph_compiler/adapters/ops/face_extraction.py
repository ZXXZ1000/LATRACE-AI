from __future__ import annotations

import base64
import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from threading import Lock

os.environ.setdefault("ALBUMENTATIONS_DISABLE_VERSION_CHECK", "1")


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


def extract_faces(face_app, image_list, num_workers=4):
    lock = Lock()
    faces = []

    def process_image(args):
        import cv2
        import numpy as np

        frame_idx, img_input = args
        try:
            img = None
            try:
                if isinstance(img_input, str) and os.path.exists(img_input):
                    img = cv2.imread(img_input, cv2.IMREAD_COLOR)
                else:
                    img_bytes = base64.b64decode(img_input)
                    img_array = np.frombuffer(img_bytes, dtype=np.uint8)
                    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            except Exception:
                img = None

            if img is None:
                return []

            with _suppress_native_output():
                detected_faces = face_app.get(img)
            frame_faces = []

            for face in detected_faces:
                bbox = [int(x) for x in face.bbox.astype(int).tolist()]
                dscore = face.det_score
                embedding = [float(x) for x in face.normed_embedding.tolist()]

                embedding_np = np.array(face.embedding)
                qscore = np.linalg.norm(embedding_np, ord=2)

                height = bbox[3] - bbox[1]
                width = bbox[2] - bbox[0]
                aspect_ratio = height / width
                face_type = "ortho" if 1 < aspect_ratio < 1.5 else "side"

                h, w = img.shape[:2]
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))
                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 8 or (y2 - y1) < 8:
                    continue
                face_img = img[y1:y2, x1:x2]
                if face_img is None or face_img.size == 0:
                    continue
                ok, buffer = cv2.imencode(".jpg", face_img)
                if not ok:
                    continue
                face_base64 = base64.b64encode(buffer).decode("utf-8")

                frame_faces.append(
                    {
                        "frame_id": frame_idx,
                        "bounding_box": [x1, y1, x2, y2],
                        "face_emb": embedding,
                        "cluster_id": -1,
                        "extra_data": {
                            "face_type": face_type,
                            "face_base64": face_base64,
                            "face_detection_score": str(dscore),
                            "face_quality_score": str(qscore),
                        },
                    }
                )
            return frame_faces
        except Exception:
            return []

    try:
        if (
            image_list
            and isinstance(image_list[0], (list, tuple))
            and len(image_list[0]) == 2
            and isinstance(image_list[0][0], int)
        ):
            indexed_inputs = list(image_list)
        else:
            indexed_inputs = list(enumerate(image_list))
    except Exception:
        indexed_inputs = list(enumerate(image_list))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for frame_faces in executor.map(process_image, indexed_inputs):
            with lock:
                faces.extend(frame_faces)
    return faces


__all__ = ["extract_faces"]
