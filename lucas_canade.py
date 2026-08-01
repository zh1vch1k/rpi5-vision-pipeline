import cv2 as cv
import numpy as np
import model

onnx_model = model.get_model()

def frame_process(): 

    video = cv.VideoCapture(0)

    video.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    video.set(cv.CAP_PROP_FRAME_HEIGHT, 360)

    prev_frame_features = {
        'past_frame': None,
        'features': None
    }

    nextPts, status, err = None, None, None

    bg_sub = cv.createBackgroundSubtractorMOG2(history=60,
                                            varThreshold=50,
                                            detectShadows=False)
    try: 
        while True:
            ret, frame = video.read()
            if not ret:
                break

            frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

            blurred = cv.bilateralFilter(frame_gray, 11, 17, 17)
            bg_mask = bg_sub.apply(blurred)

            _, bg_mask = cv.threshold(bg_mask, 200, 255, cv.THRESH_BINARY)

            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
            mc_mask = cv.morphologyEx(bg_mask, cv.MORPH_CLOSE, kernel)
            m_mask = cv.morphologyEx(mc_mask, cv.MORPH_OPEN, kernel)

            contours, _ = cv.findContours(m_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            motion_roi_mask = np.zeros_like(m_mask)
            has_motion = False

            #Denoising the mask, because MOG2 is sensible for noise
            for cnt in contours:
                if cv.contourArea(cnt) > 2000:
                    cv.drawContours(motion_roi_mask, [cnt], -1, 255, -1)
                    has_motion = True

            if not has_motion:
                if hasattr(onnx_model, 'predictor') and onnx_model.predictor is not None:
                    if hasattr(onnx_model.predictor, 'trackers'):
                        for t in onnx_model.predictor.trackers:
                            if hasattr(t, 'reset'):
                                t.reset()

                        delattr(onnx_model.predictor, 'trackers')

            if has_motion:
                results = onnx_model.track(frame,
                                        tracker="botsort.yaml",
                                        persist=True,
                                        retina_masks=False,
                                        conf=0.7)

                for r in results:
                    if r.boxes and r.boxes.id is not None:
                        boxes = r.boxes.xyxy.cpu().numpy()
                        track_ids = r.boxes.id.int().cpu().numpy()
                        cls = r.boxes.cls.int().cpu().numpy()

                        for box, track_id, cl in zip(boxes, track_ids, cls):
                            x1, y1, x2, y2 = map(int, box)
                            cv.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                            cv.putText(frame, f"ID: {track_id} Cls: {cl}", (x1, y1 - 10),
                                    cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if prev_frame_features['past_frame'] is None:
                prev_frame_features['past_frame'] = frame_gray
                if has_motion:

                    prev_frame_features['features'] = cv.goodFeaturesToTrack(
                        frame_gray, 100, 0.01, 10, mask=motion_roi_mask
                    )
            else:
                if prev_frame_features['features'] is not None and len(prev_frame_features['features']) > 0:
                    p0 = prev_frame_features['features']

                    nextPts, status, err = cv.calcOpticalFlowPyrLK(
                        prev_frame_features['past_frame'],
                        frame_gray,
                        p0,
                        None,
                        winSize=(13, 13),
                        maxLevel=3,
                        criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 30, 0.01)
                    )

                    if nextPts is not None and status is not None:
                        d = np.linalg.norm(nextPts - p0, axis=2).ravel()

                        valid_mask = (status.ravel() == 1) & (d > 0.5)
                        good_new = nextPts[valid_mask]
                    else:
                        good_new = np.empty((0, 2), dtype=np.float32)

                    if len(good_new) < 50 and has_motion:
                        feature_mask = motion_roi_mask.copy()

                        for pt in good_new:
                            x, y = pt.ravel()
                            cv.circle(feature_mask, (int(x), int(y)), 15, 0, -1)

                        new_pts = cv.goodFeaturesToTrack(
                            frame_gray,
                            maxCorners=100 - len(good_new),
                            qualityLevel=0.01,
                            minDistance=10,
                            mask=feature_mask
                        )

                        if new_pts is not None:
                            pts_old = good_new.reshape(-1, 2)
                            pts_new = new_pts.reshape(-1, 2)
                            all_pts = np.vstack((pts_old, pts_new))
                            prev_frame_features['features'] = all_pts.reshape(-1, 1, 2).astype(np.float32)
                        else:
                            prev_frame_features['features'] = good_new.reshape(-1, 1, 2).astype(np.float32)
                    else:
                        prev_frame_features['features'] = good_new.reshape(-1, 1, 2).astype(np.float32)

                elif has_motion:
                    prev_frame_features['features'] = cv.goodFeaturesToTrack(
                        frame_gray, 100, 0.01, 10, mask=motion_roi_mask)


                prev_frame_features['past_frame'] = frame_gray


            if prev_frame_features['features'] is not None and len(prev_frame_features['features']) > 0:
                pts = prev_frame_features['features'].reshape(-1, 2)
                for x, y in pts:
                    cv.circle(frame, (int(x), int(y)), 4, (0, 255, 0), -1)

                cv.imshow('Motion Flow Tracking', frame)
            else:
                cv.putText(frame, 'No motion found', (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            yield frame

    finally:
        video.release()
        cv.destroyAllWindows()

            

