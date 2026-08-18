import queue
import cv2 as cv
import numpy as np
import model
import time
import config_parser as config
from collections import deque
from dataclasses import dataclass
from typing import List

ctx = config.get_context('config.json')
FRAME_WIDTH = ctx['FRAME_WIDTH']
FRAME_HEIGHT = ctx['FRAME_HEIGHT']


@dataclass
class Tracker:
    track_id: int
    bbox: List[int]
    points_to_track: np.ndarray
    status: np.ndarray
    missed_frames: int = 0


onnx_model = model.get_model()

fps_deque = deque(maxlen=100)
inference_deque = deque(maxlen=100)


def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    intersect = max(0, xB - xA) * max(0, yB - yA)
    if intersect == 0:
        return 0.0

    total_a = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    total_b = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return intersect / float(min(total_a, total_b))


def optical_flow_shift(prev_pts, new_pts, status=None):
    if prev_pts is None or new_pts is None or len(prev_pts) == 0 or len(new_pts) == 0:
        return None

    prev_pts = np.array(prev_pts).reshape(-1, 2)
    new_pts = np.array(new_pts).reshape(-1, 2)

    if status is not None:
        valid_mask = status.ravel() == 1
        prev_pts = prev_pts[valid_mask]
        new_pts = new_pts[valid_mask]

    if len(prev_pts) == 0 or len(new_pts) == 0:
        return None

    deltas = new_pts - prev_pts
    delta_x = float(np.median(deltas[:, 0]))
    delta_y = float(np.median(deltas[:, 1]))

    return (delta_x, delta_y)


def extract_bboxes_from_results(results, frame_gray, current_trackers):
    if results is None or len(results) == 0:
        return current_trackers

    existing_by_id = {t.track_id: t for t in current_trackers}
    updated_trackers = []

    for r in results:
        if r.boxes is None or r.boxes.id is None:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        track_ids = r.boxes.id.int().cpu().numpy()

        for box, track_id in zip(boxes, track_ids):
            bbox = list(map(int, box))

            if track_id in existing_by_id:
                tr = existing_by_id[track_id]
                tr.bbox = bbox
                tr.missed_frames = 0
                updated_trackers.append(tr)
            else:
                mask = np.zeros(frame_gray.shape, dtype=np.uint8)
                mask[bbox[1]:bbox[3], bbox[0]:bbox[2]] = 255

                pts = cv.goodFeaturesToTrack(
                    frame_gray,
                    maxCorners=50,
                    qualityLevel=0.01,
                    minDistance=10,
                    mask=mask,
                )

                if pts is not None:
                    status = np.ones((len(pts), 1), dtype=np.uint8)
                    updated_trackers.append(
                        Tracker(
                            track_id=int(track_id),
                            bbox=bbox,
                            points_to_track=pts.astype(np.float32),
                            status=status,
                            missed_frames=0,
                        )
                    )
    return updated_trackers


def inference(in_queue: queue.Queue, out_queue: queue.Queue):
    while True:
        frame = in_queue.get()
        if frame is None:
            break

        result = onnx_model.track(
            frame,
            tracker="botsort.yaml",
            persist=True,
            retina_masks=False,
            conf=0.7,
            verbose=False,
        )

        if out_queue.full():
            try:
                out_queue.get_nowait()
            except queue.Empty:
                pass

        out_queue.put(result)

        if hasattr(onnx_model, 'predictor') and onnx_model.predictor is not None:
            if hasattr(onnx_model.predictor, 'trackers'):
                for t in onnx_model.predictor.trackers:
                    if hasattr(t, 'reset'):
                        t.reset()


def frame_process(in_queue: queue.Queue, out_queue: queue.Queue):
    video = cv.VideoCapture(0)
    video.set(cv.CAP_PROP_BUFFERSIZE, 1)
    video.set(cv.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    prev_frame_gray = None
    last_yolo_time = 0.0
    trackers_list = []

    bg_sub = cv.createBackgroundSubtractorMOG2(
        history=60, varThreshold=50, detectShadows=False
    )

    try:
        while True:
            start = time.time()
            ret, frame = video.read()
            if not ret:
                break

            frame = cv.resize(
                frame, (FRAME_WIDTH, FRAME_HEIGHT), interpolation=cv.INTER_CUBIC
            )
            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            try:
                inference_results = out_queue.get_nowait()
                if inference_results is not None:
                    trackers_list = extract_bboxes_from_results(
                        inference_results, frame_gray, trackers_list
                    )
            except queue.Empty:
                pass

            blurred = cv.bilateralFilter(frame_gray, 11, 17, 17)
            bg_mask = bg_sub.apply(blurred)
            _, bg_mask = cv.threshold(bg_mask, 200, 255, cv.THRESH_BINARY)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
            mc_mask = cv.morphologyEx(bg_mask, cv.MORPH_CLOSE, kernel)
            m_mask = cv.morphologyEx(mc_mask, cv.MORPH_OPEN, kernel)

            contours, _ = cv.findContours(
                m_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            motion_roi_mask = np.zeros_like(m_mask)

            mog_bbox = []
            has_motion = False
            for cnt in contours:
                if cv.contourArea(cnt) > 1500:
                    x, y, w, h = cv.boundingRect(cnt)
                    mog_bbox.append([x, y, x + w, y + h])
                    cv.drawContours(motion_roi_mask, [cnt], -1, 255, -1)
                    has_motion = True

            need_yolo_inference = False
            if len(trackers_list) == 0:
                need_yolo_inference = True
            elif has_motion:
                for m_box in mog_bbox:
                    max_iou = max(
                        [compute_iou(m_box, t.bbox) for t in trackers_list],
                        default=0.0,
                    )
                    if max_iou < 0.15:
                        need_yolo_inference = True
                        break

            current_time = time.time()
            if need_yolo_inference and (current_time - last_yolo_time >= 0.5):
                try:
                    in_queue.put_nowait(frame)
                    last_yolo_time = current_time
                except queue.Full:
                    pass

            for track in trackers_list:
                alive_mask = track.status.ravel() == 1
                alive_pts = track.points_to_track[alive_mask]

                if len(alive_pts) < 10 and has_motion:
                    mask = np.zeros(frame_gray.shape, dtype=np.uint8)
                    cv.rectangle(
                        mask,
                        (track.bbox[0], track.bbox[1]),
                        (track.bbox[2], track.bbox[3]),
                        255,
                        -1,
                    )
                    if len(alive_pts) > 0:
                        for c in alive_pts:
                            pt = c.ravel()
                            cv.circle(mask, (int(pt[0]), int(pt[1])), 4, 0, -1)

                    new_pts = cv.goodFeaturesToTrack(
                        frame_gray, 50, 0.01, 10, mask=mask
                    )
                    if new_pts is not None:
                        if len(alive_pts) > 0:
                            track.points_to_track = np.vstack(
                                (alive_pts.reshape(-1, 1, 2), new_pts)
                            ).astype(np.float32)
                        else:
                            track.points_to_track = new_pts.astype(np.float32)
                        track.status = np.ones(
                            (len(track.points_to_track), 1), dtype=np.uint8
                        )

                if prev_frame_gray is not None and len(track.points_to_track) > 0:
                    new_features, new_sts, _ = cv.calcOpticalFlowPyrLK(
                        prev_frame_gray,
                        frame_gray,
                        track.points_to_track.astype(np.float32),
                        None,
                        winSize=(15, 15),
                        maxLevel=3,
                    )

                    shift = optical_flow_shift(
                        track.points_to_track, new_features, new_sts
                    )
                    if shift is not None:
                        delta_x, delta_y = shift
                        track.bbox[0] += int(delta_x)
                        track.bbox[1] += int(delta_y)
                        track.bbox[2] += int(delta_x)
                        track.bbox[3] += int(delta_y)

                        valid_mask = new_sts.ravel() == 1
                        track.points_to_track = new_features[
                            valid_mask
                        ].reshape(-1, 1, 2)
                        track.status = new_sts[valid_mask]
                        track.missed_frames = 0
                    else:
                        track.missed_frames += 1
                else:
                    track.missed_frames += 1

            trackers_list = [
                t
                for t in trackers_list
                if t.points_to_track is not None
                and len(t.points_to_track[t.status.ravel() == 1]) > 0
                and t.missed_frames < 30
                and (t.bbox[2] > 0 and t.bbox[0] < FRAME_WIDTH and t.bbox[3] > 0 and t.bbox[1] < FRAME_HEIGHT)
            ]

            for track in trackers_list:
                cv.rectangle(
                    frame,
                    (track.bbox[0], track.bbox[1]),
                    (track.bbox[2], track.bbox[3]),
                    (0, 0, 255),
                    2,
                )
                alive_mask = track.status.ravel() == 1
                if len(track.points_to_track[alive_mask]) > 0:
                    for c in track.points_to_track[alive_mask]:
                        pt = c.ravel()
                        cv.circle(
                            frame, (int(pt[0]), int(pt[1])), 2, (0, 255, 0), -1
                        )
                cv.putText(
                    frame,
                    f'id: {track.track_id}',
                    (int(track.bbox[0]), max(20, int(track.bbox[1] - 10))),
                    cv.FONT_HERSHEY_COMPLEX,
                    0.5,
                    (0, 0, 255),
                )

            fps = 1.0 / max(1e-5, (time.time() - start))
            fps_deque.append(fps)

            if len(fps_deque) > 1:
                y_labels = np.array(fps_deque)
                x_points = np.arange(len(fps_deque))
                baseline_y = FRAME_HEIGHT - 20

                y_coords = baseline_y - y_labels
                points = (
                    np.column_stack((x_points, y_coords))
                    .astype(np.int32)
                    .reshape(-1, 1, 2)
                )
                cv.polylines(
                    frame,
                    [points],
                    isClosed=False,
                    color=(0, 255, 255),
                    thickness=1,
                )

                cv.putText(
                    frame,
                    f'FPS: {int(fps_deque[-1])}',
                    (10, 60),
                    cv.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    1,
                )

            prev_frame_gray = frame_gray.copy()
            yield frame

    finally:
        video.release()
        cv.destroyAllWindows()