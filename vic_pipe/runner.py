if __name__ == "__main__":
    print("HI!")

    from tools.vqa.qwen2_5_vl import Qwen2_5_VL_VQA
    print("Imported VLM.")
    import os
    import json
    from lmms_eval.tasks.strokerehab.utils_primitives import load_strokerehab_primitives_dataset
    from lmms_eval.tasks.strokerehab.utils_primitives import _get_primitives_score
    print("Imported LMMS utils.")
    from data.utils_strokerehab import DataPaths, PrimitiveLabelUtils
    print("Imported DataPaths.")
    from loguru import logger as eval_logger
    print("Imported logger.")

    from tools.ultralytics_pose import Pose2DStream
    print("Imported Pose2DStream.")
    from tools.final_pipe_v1 import predict_with_state_machine
    print("Imported predict_with_state_machine.")
    import pandas as pd
    import matplotlib.pyplot as plt

    def get_paths(patients='C00020,C00023'):
        """Get paths for evaluation."""
        C00020_videos = [
            "C00020/C00020_combing1_2.mkv",
            "C00020/C00020_shelf right side1_2.mkv",
        ]
        videos = C00020_videos
        regex = "|".join([f"({v})" for v in videos])
        regex = rf"^({regex})$"
        ds = load_strokerehab_primitives_dataset(
            video_regex=regex
        )
        # ds = load_strokerehab_primitives_dataset(
        #     patients=patients,
        #     reps='first',
        # )
        paths = pd.DataFrame(ds['test'])[['path_v', 'path_l']]
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

    def evaluate_prims(dfs, path_ls, path_vs):
        n = 0
        sum_es = 0.0
        sum_aer = 0.0

        per_sample = {}
        skipped = []

        results = {}

        for df, path_l, path_v in zip(dfs, path_ls, path_vs):
            sample_id = os.path.basename(path_l).split('.')[0]

            print(f"Processing {sample_id}... ", end='')

            prim, ref = df['predicted'].tolist(), df['reference'].tolist()
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

    print("Starting model load...")
    vlm = Qwen2_5_VL_VQA(
        pretrained="Qwen/Qwen2.5-VL-32B-Instruct",
        device="cuda",
        device_map=None,
    )
    print("Qwen2_5_VL_VQA loaded.")
    streamer = Pose2DStream()
    print("Pose model loaded.")
    eval_logger.debug("Model loaded.")

    def preds_to_df(path_l, path_v):
        handedness = PrimitiveLabelUtils.get_handedness(path_l)
        prims, times, infos = predict_with_state_machine(
            path_v, handedness, vlm, streamer
        )
        for key in infos:
            infos[key] = [str(i) for i in infos[key]]
        df = pd.DataFrame({**infos, "predicted": prims, "times": times})

        refs, refs_times = PrimitiveLabelUtils.convert_labels_to_prims_times(path_l, duplicate_last_prim=True)
        df_refs = pd.DataFrame({"times": refs_times, "reference": refs})
        out = pd.merge_asof(
            df,
            df_refs,
            on="times",
            direction="backward"
        )
        return out


    def evaluate_prims_individual(path_l, path_v):
        df = preds_to_df(path_l, path_v)
        return df


    def plot_factor_bars(df: pd.DataFrame, time_col: str, factor_cols: list[str]):
        # Collect all unique non-null strings
        all_vals = pd.unique(df[factor_cols].astype(str).values.ravel())
        colors = {v: plt.cm.tab20(i % 20) for i, v in enumerate(all_vals)}

        fig, ax = plt.subplots(figsize=(12, 1 + len(factor_cols) / 2.))
        for j, col in enumerate(factor_cols):
            for _, row in df.iterrows():
                val = str(row[col])
                ax.barh(j, 1, left=row[time_col], color=colors[val])
        ax.set_yticks(range(len(factor_cols)))
        ax.set_yticklabels(factor_cols)
        ax.set_xlabel("Time")
        ax.legend([plt.Rectangle((0,0),1,1,color=colors[v]) for v in all_vals],
                all_vals, title="Values", bbox_to_anchor=(1.05,1), loc="upper left")
        plt.tight_layout()
        video = df['video'].iloc[0] if 'video' in df else 'factors'
        plt.savefig(f"viz/factors_plot_{video}.png")


    path_ls, path_vs = get_paths(patients='C00020,C00023')
    # path_ls, path_vs = get_paths()
    print(f"Evaluating {len(path_ls)} videos...")
    print("Starting!")

    dfs = []
    for path_l, path_v in zip(path_ls, path_vs):
        print(f"Processing {path_l}...")
        eval_logger.debug(f"Processing {path_l}...")
        print(f"Processing {path_v}...")
        df = preds_to_df(path_l, path_v)
        df['video'] = os.path.basename(path_l).split('.')[0]
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    
    final_results = evaluate_prims(dfs, path_ls, path_vs)

    print("=" * 80)
    print(f"ES: {final_results['aggregate']['edit_score']:.4f}, "
            f"AER: {final_results['aggregate']['action_error_rate']:.4f}")
    print("=" * 80)
    os.makedirs("results", exist_ok=True)
    with open(f"results/eval_prims_individual.json", "w", encoding="utf-8") as f:
        json.dump(_to_jsonable(final_results), f, indent=2, ensure_ascii=False, sort_keys=True)

    scores = []
    for video in df['video'].unique():
        score = get_primitives_score(
            df[df['video'] == video]['predicted'].tolist(),
            df[df['video'] == video]['reference'].tolist()
        )
        scores.append((video, score['edit_score'], score['action_error_rate']))
    scores_df = pd.DataFrame(scores, columns=['video', 'edit_score', 'action_error_rate'])
    scores_df.to_csv("results/eval_prims_individual.csv", index=False)

    factor_cols = [
        'predicted', 'reference', 'held_object', 'status'
    ]
    for video in df['video'].unique():
        plot_factor_bars(df[df['video'] == video], time_col='times', factor_cols=factor_cols)
