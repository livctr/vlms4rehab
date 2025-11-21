import os
import numpy as np
import pandas as pd

from data.utils_strokerehab import DataPaths, PrimitiveLabelUtils
from lmms_eval.tasks.strokerehab.utils_primitives import load_strokerehab_primitives_dataset
from postprocess.left_v_right.left_v_right_prompter import get_left_v_right_answers
from tools.ultralytics_pose import Pose2DStream
from lmms_eval.models.qwen2_5_vl_signal_generator import Qwen2_5_VL_SignalGenerator as Qwen2_5_VL_VQA
# from tools.vqa.qwen2_5_vl import Qwen2_5_VL_VQA


def convert_answers_times_to_pandas(answers, times):
    records = []
    for ans, t in zip(answers, times):
        record = {
            'left_hand_active_crop' : "yes" in ans[0].lower(),
            'right_hand_active_crop': "yes" in ans[1].lower(),
            'left_hand_active': "yes" in ans[2].lower(),
            'right_hand_active': "yes" in ans[3].lower(),
            'iou': float(ans[4]) if ans[4] != "N/A" else np.nan,
            'time': t,
        }
        records.append(record)
    return pd.DataFrame.from_records(records)


if __name__ == "__main__":
    vlm = Qwen2_5_VL_VQA(
        pretrained="Qwen/Qwen2.5-VL-32B-Instruct",
        device="cuda",
        device_map=None,
        use_cache=True,
    )
    streamer = Pose2DStream()

    # DATASET
    ds = load_strokerehab_primitives_dataset(activity='RTT left side,RTT right side')
    df = ds['test'].to_pandas()
    control_mask = ~df['stroke']
    first_rep_mask = df['path_v'].str.contains('RTT right side1') | df['path_v'].str.contains('RTT left side1')
    df = df[control_mask & first_rep_mask]
    patient_has_all_four = (df.groupby('patient')['id'].count() == 4)
    df = df[df['patient'].isin(patient_has_all_four[patient_has_all_four].index.tolist())]
    control_patient_list = "C00022,C00023,C00024,C00028,C00029".split(',')
    df = df[df['patient'].isin(control_patient_list)].copy()

    adfs = []

    for path_v, path_l, video_id in zip(df['path_v'], df['path_l'], df['id']):
        print(f"Processing video ID: {video_id}")

        # VLM answers for left v. right
        answers, times = get_left_v_right_answers(
            video_path=os.path.join(DataPaths.RAW_VIDEO_DIR, path_v),
            vlm=vlm,
            pose_stream=streamer,
            max_frames_num=8,
            sampling_strategy="dense",
            overlap_frames_num=0,
            sampling_fps=15,
        )
        adf = convert_answers_times_to_pandas(answers, times)

        adf['id'] = video_id
        adf['path_l'] = path_l
        path_l = os.path.join(DataPaths.RAW_LABEL_DIR, path_l)
        handedness = PrimitiveLabelUtils.get_handedness(path_l)
        adf['hand_in_focus'] = handedness

        adfs.append(adf)

    adfs = pd.concat(adfs, ignore_index=True)
    adfs.to_csv("postprocess/left_v_right/left_v_right_activation_results.csv", index=False)
