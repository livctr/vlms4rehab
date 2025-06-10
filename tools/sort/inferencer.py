import numpy as np
from tools.sort import Sort

from tools import tools_logger


def track(box_data, max_age_s=2, min_hits_s=0.5):
    """
    Annotates the frame detections in place with the tracked objects using SORT.
    Adds another key "track_id" to each dictionary representing an object in a
    frame. Note that two objects with different target labels may end up with
    the same "track_id". However, it is guaranteed that (target_label, track_id)
    is unique per object.

    Args:
        box_data (List[List[Dict]]): List of detections per frame.
        target_label (str): The object label to track (e.g. "person").
    """
    assert len(box_data["Time_s"]) > 1, "At least two frames are required for tracking."
    sample_rate = (box_data["Time_s"][-1] - box_data["Time_s"][0]) / (len(box_data["Time_s"]) - 1)
    max_age = int(max_age_s / sample_rate)
    min_hits = int(min_hits_s / sample_rate)

    labels = set([det["text_label"] for frame_dets in box_data["Boxes"] for det in frame_dets])
    trackers = {label: Sort(max_age=max_age, min_hits=min_hits) for label in labels}

    box_data["metadata"]["box_format"] = "[x_min, y_min, x_max, y_max, score, id]"

    for i, frame_dets in enumerate(box_data["Boxes"]):

        # Get detections by label
        dets = {label: [] for label in labels}
        for det in frame_dets:
            dets[det["text_label"]].append(det["box"])  # [x1, y1, x2, y2, score]
        
        new_frame_dets = []

        for label in labels:
            if len(dets[label]) == 0:
                dets_np = np.empty((0, 5))
            else:
                dets_np = np.array(dets[label])
            
            tracker = trackers[label]

            for identified_det in tracker.update(dets_np, include_score=True):
                x1, y1, x2, y2, score, id_ = identified_det.tolist()
                box = [x1, y1, x2, y2]
                new_frame_dets.append({
                    "text_label": label,
                    "box": [round(coord, 2) for coord in box],
                    "score": round(score, 3),
                    "id": int(id_),
                })

        box_data["Boxes"][i] = new_frame_dets
    
    return box_data


if __name__ == "__main__":
    import json
    json_file = "detection_results.json"

    with open(json_file, 'r') as f:
        data = json.load(f)
    data = track(data)
    with open("detection_result_2.json", 'w') as f:
        json.dump(data, f, indent=4)
