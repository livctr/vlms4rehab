import pandas as pd
from typing import Optional, Tuple, List
import os


HEALTHY_PATIENTS = (
    "C00011,C00012,C00015,C00019,C00020,C00022,C00023,C00024,C00025,C00026,"
    "C00027,C00028,C00029,C00030,C00031,C00032,C0004,C0005,C0007,C0009"
)
MILD_PATIENTS = (
    "S0005,S0007,S0009,S00010,S00012,S00013,S00016,S00023,S00026,S00028,"
    "S00030,S00032,S00033,S00035,S00037,S00040,S00041,S00042,S00043,S00047"
)
MODERATE_PATIENTS = (
    "S0001,S0002,S0003,S0004,S0006,S0008,S00011,S00017,S00018,S00019,S00020,"
    "S00022,S00024,S00025,S00027,S00036,S00039,S00044,S00045,S00046,S00048,"
    "S00053,S00054"
)
SEVERE_PATIENTS = (
    "S00021,S00029,S00031,S00034,S00049,S00050,S00051,S00055"
)
AFFECTED_PATIENTS = (
    MILD_PATIENTS + "," + \
    MODERATE_PATIENTS + "," + \
    SEVERE_PATIENTS
)
PATIENTS = (
    AFFECTED_PATIENTS + "," + \
    HEALTHY_PATIENTS
)


###################### Impairment assessment ######################
FM_ITEM_TO_FM_RANGE = {
    3: (3, 8), 4: (3, 8), 5: (3, 8), 6: (3, 8), 7: (3, 8), 8: (3, 8),
    9: (9, 11), 10: (9, 11), 11: (9, 11),
    12: (12, 12), 13: (13, 13), 14: (14, 14), 15: (15, 15), 16: (16, 16), 17: (17, 17),
    18: (18, 18), 19: (19, 19), 20: (20, 20), 21: (21, 21), 22: (22, 22), 23: (23, 23),
    24: (24, 25), 25: (24, 25),
    26: (26, 26), 27: (27, 27), 28: (28, 28), 29: (29, 29), 30: (30, 30),
    31: (31, 33), 32: (31, 33), 33: (31, 33),
}


##################### Path and File Utilities #####################
class DataPaths:

    DATA_DIR = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/"
    RAW_VIDEO_DIR = os.path.join(DATA_DIR, "VideoData/rawVideosADLsandFM")
    RAW_LABEL_DIR = os.path.join(DATA_DIR, "rawVideoLabels")

    METADATA_DIR = os.path.join(DATA_DIR, "primitives_metadata")
    METADATA_PATH = os.path.join(METADATA_DIR, "metadata.csv")
    VIDEO_METADATA_PATH = os.path.join(METADATA_DIR, "video_metadata.csv")
    LABEL_METADATA_PATH = os.path.join(METADATA_DIR, "label_metadata.csv")

    FP_METADATA_PATH = os.path.join("./data/fp/fp_metadata.csv")
    IA_VIDEO_METADATA_PATH1 = os.path.join("./data/ia/ia_video_metadata1.csv")
    IA_VIDEO_METADATA_PATH2 = os.path.join("./data/ia/ia_video_metadata2.csv")
    IA_VIDEO_METADATA_PATH3 = os.path.join("./data/ia/ia_video_metadata3.csv")
    IA_VIDEO_METADATA_PATH4 = os.path.join("./data/ia/ia_video_metadata4.csv")
    IA_CLIPS_PATH = os.path.join("./data/ia/fm_item_clip_times.csv")
    IA_SCORES_PATH = os.path.join("./data/ia/fm_item_scores.csv")
    IA_QUESTIONS_PATH1 = os.path.join("./data/ia/fm_item_questions1.csv")
    IA_QUESTIONS_PATH2 = os.path.join("./data/ia/fm_item_questions2.csv")
    IA_QUESTIONS_PATH3 = os.path.join("./data/ia/fm_item_questions3.csv")
    IA_QUESTIONS_PATH4 = os.path.join("./data/ia/fm_item_questions4.csv")
    VIEWS_PATH = os.path.join("./data/ia/fm_item_views.csv")

    # the folder structure of the IA raw and clipped dirs mirrors that of the raw video dir
    IA_VIDEO_DIR = os.path.join(DATA_DIR, "VideoDataDerived/FMVideoClips/")
    IA_RAW_VIDEO_DIR = os.path.join(IA_VIDEO_DIR, "rawVideos/")
    IA_CLIPPED_VIDEO_DIR = os.path.join(IA_VIDEO_DIR, "clippedVideos/")

    FM_POSE_DIR = os.path.join(DATA_DIR, "VideoDataDerived/FMPoseData")
    ADL_POSE_DIR = os.path.join(DATA_DIR, "VideoDataDerived/ADLPoseData")


##################### Metadata Utilities #####################
class PrimitiveLabelUtils:

    PRIMITIVES = ["reach", "reposition", "transport", "stabilize", "idle"]

    @staticmethod
    def get_handedness(path):
        """Returns the handedness of the markers in the given label file."""
        labels = pd.read_csv(path)
        marker_names = labels['MarkerNames']
        num_lefts = marker_names.str.startswith('l_').sum() + marker_names.str.contains('_l_').sum()
        num_rights = marker_names.str.startswith('r_').sum() + marker_names.str.contains('_r_').sum()
        return "left" if num_lefts > num_rights else "right"

    @staticmethod
    def convert_labels_to_action_sequence(path, handedness=None):
        """
        Converts marker names to an action sequence with start times and durations.
        
        E.g.
        Input:
            MarkerNames: ['l_reach', 'l_reach', 'l_transport_prox', 'l_stabilize', 'r_reach']
            Time_s:       [0.0,       0.5,       1.0,            1.5,         2.0]
            handedness='left'
        Output:
            [
                {'action': 'reach', 'start_time': 0.0, 'duration': 0.5},
                {'action': 'transport', 'start_time': 1.0, 'duration': 0.5},
                {'action': 'stabilize', 'start_time': 1.5, 'duration': None},
            ]
        """
        df = pd.read_csv(path)
        times = df['Time_s'].tolist()
        markers = df['MarkerNames']
        if handedness is None:
            num_lefts = markers.str.startswith('l_').sum() + markers.str.contains('_l_').sum()
            num_rights = markers.str.startswith('r_').sum() + markers.str.contains('_r_').sum()
            handedness = "left" if num_lefts > num_rights else "right"
        markers = markers.tolist()

        sequence = []

        def append_unique(time, action):
            if not sequence or sequence[-1]['action'] != action:
                sequence.append({'action': action, 'start_time': time})

        for time, marker in zip(times, markers):
            if (handedness == 'left' and (marker.startswith('l_') or '_l_' in marker)) or \
            (handedness == 'right' and (marker.startswith('r_') or '_r_' in marker)):
                if 'reach' in marker:
                    append_unique(time, 'reach')
                elif 'reposition' in marker or 'retract' in marker:
                    append_unique(time, 'reposition')
                elif 'transport' in marker:
                    append_unique(time, 'transport')
                elif 'stabilize' in marker:
                    append_unique(time, 'stabilize')
                elif 'idle' in marker or 'rest' in marker:
                    append_unique(time, 'idle')
        last_time = times[-1] if times else 0.0

        # Compute durations for each action; last action duration remains None
        for i in range(len(sequence)):
            start_time = sequence[i]['start_time']
            if i < len(sequence) - 1:
                sequence[i]['duration'] = sequence[i+1]['start_time'] - start_time
            else:
                sequence[i]['duration'] = last_time - start_time

        return sequence
    

    @staticmethod
    def convert_labels_to_prims_times(
        path: str, 
        handedness: Optional[str] = None,
        duplicate_last_prim: bool = False
    ) -> Tuple[List[str], List[float]]:
        """
        Reads the CSV at `path`, extracts the action sequence for the dominant hand,
        and returns (prims, times) where
        - prims: list of primitives, e.g. ['reach','transport','stabilize']
        - times: list of timestamps, one more than prims,
                e.g. [t0, t1, t2, t3] so len(times) == len(prims)+1
        """
        df = pd.read_csv(path)
        times = df['Time_s'].tolist()
        markers = df['MarkerNames']

        # infer handedness if needed
        if handedness not in ('left','right'):
            num_lefts = markers.str.startswith('l_').sum() + markers.str.contains('_l_').sum()
            num_rights = markers.str.startswith('r_').sum() + markers.str.contains('_r_').sum()
            handedness = 'left' if num_lefts > num_rights else 'right'

        prims: List[str] = []
        prim_times: List[float] = []

        def append_unique(action: str, t: float):
            if not prims or prims[-1] != action:
                prims.append(action)
                prim_times.append(t)

        for t, m in zip(times, markers.tolist()):
            m_low = m.lower()
            if handedness == 'left' and not (m_low.startswith('l_') or '_l_' in m_low):
                continue
            if handedness == 'right' and not (m_low.startswith('r_') or '_r_' in m_low):
                continue

            if 'reach' in m_low:
                append_unique('reach', t)
            elif 'reposition' in m_low or 'retract' in m_low:
                append_unique('reposition', t)
            elif 'transport' in m_low:
                append_unique('transport', t)
            elif 'stabilize' in m_low:
                append_unique('stabilize', t)
            elif 'idle' in m_low or 'rest' in m_low:
                append_unique('idle', t)

        # if no primitives found, return empty
        if not prims:
            return [], []

        if duplicate_last_prim:  # If we want number of prims / times to match
            prims.append(prims[-1])        
        prim_times.append(times[-1])

        return prims, prim_times



##################### Response String Utilities #####################
def resps_to_string(prims: tuple[str, ...], times: tuple[float, ...]) -> str:
    """
    Turns two parallel tuples into a single semicolon-delimited string:
      "PRIM1@time1;PRIM2@time2;..."
    """
    if len(times) != len(prims) + 1:
        raise ValueError("`times` must have exactly one more element than `prims`")

    # For each primitive we pair it with its start time...
    segments = [f"{p}@{t:.3f}" for p, t in zip(prims, times)]
    # ...and then append the final end-time for the last primitive
    segments.append(f"{prims[-1]}@{times[-1]:.3f}")
    return ";".join(segments)


def string_to_resps(s: str, drop_duplicated: bool = True) -> tuple[tuple[str, ...], tuple[float, ...]]:
    """
    Parses the semicolon-delimited string back into
    (prims_tuple, times_tuple), where len(times_tuple) == len(prims_tuple)+1.
    """
    if not s:
        return (), ()

    tokens = s.split(";")
    prims: list[str] = []
    times: list[float] = []

    for tok in tokens:
        p, t_str = tok.split("@", 1)
        prims.append(p)
        times.append(float(t_str))

    # Drop the duplicated final primitive, but keep all timestamps
    if len(prims) >= 2 and drop_duplicated:
        prims = prims[:-1]

    return tuple(prims), tuple(times)

