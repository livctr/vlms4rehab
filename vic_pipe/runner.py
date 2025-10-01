if __name__ == "__main__":
    from tools.vqa.qwen2_5_vl import Qwen2_5_VL_VQA
    import os
    import json
    from lmms_eval.tasks.strokerehab.utils_primitives import load_strokerehab_primitives_dataset
    from lmms_eval.tasks.strokerehab.utils_primitives import _get_primitives_score
    from data.utils_strokerehab import DataPaths, PrimitiveLabelUtils
    from transformers.utils import logging
    from vic_pipe.visualize import visualize
    logging.set_verbosity_error()  # suppress warnings
    import pandas as pd

    def get_paths(regex):
        """Get paths for evaluation."""
        paths = pd.DataFrame(load_strokerehab_primitives_dataset(video_regex=regex)['test'])[['path_v', 'path_l']]
        path_ls = [os.path.join(DataPaths.RAW_LABEL_DIR, p) for p in paths['path_l'].tolist()]
        path_vs = [os.path.join(DataPaths.RAW_VIDEO_DIR, p) for p in paths['path_v'].tolist()]
        return path_ls, path_vs

    def get_primitives_score(pred, ref):
        """Get primitives score."""
        d = _get_primitives_score(pred, ref)
        return {'edit_score': d['edit_score'], 'action_error_rate': d['action_error_rate']}

    def _to_jsonable(x):
        """Make numpy / set / tuple structures JSON serializable."""
        try:
            import numpy as np
            if isinstance(x, np.ndarray):
                return x.tolist()
            if isinstance(x, (np.integer, np.floating)):
                return x.item()
        except Exception:
            pass
        if isinstance(x, (set, tuple)):
            return list(x)
        if isinstance(x, dict):
            return {k: _to_jsonable(v) for k, v in x.items()}
        if isinstance(x, list):
            return [_to_jsonable(v) for v in x]
        return x

    def dedupe_list(seq):
        dedup, cnt = [], []
        for item in seq:
            if dedup and item == dedup[-1]:
                cnt[-1] += 1
                continue
            dedup.append(item)
            cnt.append(1)
        return dedup, cnt

    def evaluate_prims(method_prims, path_ls, path_vs):
        n = 0
        sum_es = 0.0
        sum_aer = 0.0

        per_sample = {}
        skipped = []

        results = {}

        for path_l, path_v in zip(path_ls, path_vs):
            sample_id = os.path.basename(path_l).split('.')[0]

            print(f"Processing {sample_id}... ", end='')

            prim, ref = method_prims(path_l, path_v)
            s = get_primitives_score(prim, ref)  # expects keys: 'edit_score', 'action_error_rate'


            prim_dedup, prim_cnt = dedupe_list(prim)
            prim_dedup_str = ','.join(prim_dedup)
            prim_cnt_str = ','.join(map(str, prim_cnt))
            ref_str = ','.join(ref)

            # Aggregate
            sum_es += s['edit_score']
            sum_aer += s['action_error_rate']
            n += 1

            # Per-sample record
            per_sample[sample_id] = {
                "prediction": _to_jsonable(prim_dedup_str),
                "prediction_counts": _to_jsonable(prim_cnt_str),
                "ground_truth": _to_jsonable(ref_str),
                "metrics": {
                    "edit_score": s['edit_score'],
                    "action_error_rate": s['action_error_rate'],
                },
            }
            print(f"{sample_id}: \t\t {s['action_error_rate']:.4f} \t {s['edit_score']:.4f}")

            results[sample_id] = prim

        avg_es = (sum_es / n) if n else 0.0
        avg_aer = (sum_aer / n) if n else 0.0

        results.update({
            "aggregate": {
                "count": n,
                "edit_score": avg_es,
                "action_error_rate": avg_aer,
            },
            "samples": per_sample,
            "skipped": skipped,  # optional, helpful for debugging
        })
        return results

    from tools.ultralytics_pose import Pose2DStream
    from vic_pipe.stateful_contact_v2 import predict_with_state_machine

    vlm = Qwen2_5_VL_VQA(
        pretrained="Qwen/Qwen2.5-VL-32B-Instruct",
        device="cuda",
        device_map="auto",
    )
    streamer = Pose2DStream()

    def wrapper(path_l, path_v):
        handedness = PrimitiveLabelUtils.get_handedness(path_l)
        prims, prim_times, info = predict_with_state_machine(
            path_v, handedness, vlm, streamer,
            max_frames_num=4, sampling_fps=15
        )
        refs, refs_times = PrimitiveLabelUtils.convert_labels_to_prims_times(path_l)

        status = [str(s) for s in info['status']]
        objs = [obj if obj else "none" for obj in info['objs']]
        idles = [str(i) for i in info['idles']]
        contacts = [str(c) for c in info['contacts']]
        basic_contacts = [str(c) for c in info['basic_contacts']]

        visualize(
            path_v, prims, prim_times, refs, refs_times,
            ("pose_status", status, prim_times),
            ("held_objs", objs, prim_times),
            ("idles", idles, prim_times),
            ("state_contacts", contacts, prim_times),
            ("snapshot_contacts", basic_contacts, prim_times),
            bboxes=info['bboxes'], wrist_kps=info['kps_wrist'], elbow_kps=info['kps_elbow'], hand_kps=info['kps_hand'],
            overwrite=False
        )

        return prims, refs

    print("=" * 80)
    print(f"Evaluating stateful...\t\t AER \t\t ES")

    # Test on: C00020, S0005, S0001, S00021
    # Activities: glasses, drinking, combing, face wash, shelf right side, deodrant

    # C00020: 
    # C00020_videos = [
    #     "C00020/C00020_glasses1_1.mkv",
    #     "C00020/C00020_drinking1_1.mkv",
    #     "C00020/C00020_drinking1_2.mkv",
    #     "C00020/C00020_combing1_1.mkv",
    #     "C00020/C00020_combing1_2.mkv",
    #     "C00020/C00020_face wash1_1.mkv",
    #     "C00020/C00020_shelf right side1_1.mkv",
    #     "C00020/C00020_shelf right side1_2.mkv",
    #     "C00020/C00020_deodrant1_1.mkv",
    #     "C00020/C00020_deodrant1_2.mkv",
    #     "C00020/C00020_feeding1_1.mkv",
    #     "C00020/C00020_feeding1_2.mkv"
    # ]

    # S0005_videos = [
    #     "S0005/S0005_shelf left side1_2.avi",
    #     "S0005/S0005_combing2_2.avi"
    # ]

    # S0001_videos = [
    #     "S0001/S0001_shelf right side1_1.avi",
    #     "S0001/S0001_combing1_2.avi"
    # ]

    S00021_videos = [
        "S00021/S00021_RTT left side1_1.avi",
        # "S00021/S00021_combing1_1.avi"
    ]

    # videos = C00020_videos + S0005_videos + S0001_videos + S00021_videos
    videos = S00021_videos

    # regex = r'^(C00020/C00020_.*1_[12].mkv)$'
    # regex = r'^(C00020/C00020_glasses1_1.mkv|C00020/C00020_drinking1_1.mkv|C00020/C00020_combing1_1.mkv|C00020/C00020_face wash1_1.mkv|C00020/C00020_shelf right side1_1.mkv|C00020/C00020_deodrant1_1.mkv)$'
    regex = "|".join([f"({v})" for v in videos])
    regex = rf"^({regex})$"

    # regex = r'^(C00020/C00020_combing1_1.mkv)$'  # for quick debugging
    path_ls, path_vs = get_paths(regex)
    print(len(path_ls), "samples to evaluate.")
    # print(path_vs)
    results = evaluate_prims(wrapper, path_ls, path_vs)
    print("=" * 80)
    print(f"ES: {results['aggregate']['edit_score']:.4f}, "
            f"AER: {results['aggregate']['action_error_rate']:.4f}")
    print("=" * 80)
    os.makedirs("results", exist_ok=True)
    with open(f"results/eval_prims_stateful.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(results), f, indent=2, ensure_ascii=False, sort_keys=True)