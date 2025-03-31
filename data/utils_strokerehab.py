import os

import pandas as pd
import datasets


VIDEO_DIR = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/"
CHUNKED_VIDEO_DIR = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/ProcessedVideoData_1fps"
LABEL_DIR = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/"
ACTIVITY_GROUND_TRUTH_PATH = "/gpfs/data/schambralab/quantitativeRehabilitation/__lab_member_homes/victor/cvfm4rehab/data/public/activities_ground_truth.yaml"
METADATA_PATH = "/gpfs/data/schambralab/quantitativeRehabilitation/__lab_member_homes/victor/cvfm4rehab/data/public/cleaned_metadata.csv"
HUMAN_INPUT_JSON_PATH = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoDataSam2Input/human_input.json"
HUMAN_INPUT_JSON_PATH_BACKUP = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoDataSam2Input/human_input_backup.json"

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
PATIENTS = (
    HEALTHY_PATIENTS + "," + \
    MILD_PATIENTS + "," + \
    MODERATE_PATIENTS + "," + \
    SEVERE_PATIENTS
)

def strokerehab_load_dataset(patients='all', activity='all', reps='all', filter_for_testset=False):
    """
    Loads the StrokeRehab dataset from a cleaned metadata file and applies AND-ed filters.

    Args:
        patients (str): The patient IDs to include in the dataset. Default is 'all'.
            If specifying individual patients, separate them with commas
            Example: 'S0001,S0002'
        activity (str): The activity names to include in the dataset. Default is 'all' (11 total).
            If specifying individual activities, separate them with commas
            Example: 'RTT left side,RTT right side,brushing,combing,deodrant,drinking,face wash,feeding,glasses,shelf left side,shelf right side'
        reps (str): Either 'all' or 'first'.
        filter_for_testset (bool): If True, include only the VIDEOS in the test set of
            the original StrokeRehab paper. https://pubmed.ncbi.nlm.nih.gov/37766938/
            This data is located in './data/public/strokerehab_test_set.txt' and already
            incorporated into the CSV metadata file.
    
    Returns:
        dataset (datasets.Dataset): The StrokeRehab dataset with the specified filters applied.
    """
    df = pd.read_csv(METADATA_PATH)
    if patients != 'all':
        patients = patients.split(',')
        df = df[df['patient'].isin(patients)]
    if activity != 'all':
        activity = activity.split(',')
        df = df[df['activity'].isin(activity)]
    if reps != 'all':
        if reps != 'first':
            raise ValueError("Invalid value for reps. Must be 'all' or 'first'.")
        df = df.sort_values('id').groupby(['patient', 'activity']).agg('first').reset_index()
    if filter_for_testset:
        df = df[df['is_in_strokerehab_test_set']]
    dataset = datasets.Dataset.from_pandas(df)
    dataset_dict = datasets.DatasetDict({'test': dataset})
    return dataset_dict


##################### Metadata Utilities #####################
class LabelUtils:

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
    def convert_labels_to_action_sequence(path, handedness):
        """Converts the marker names to a sequence.
        
        E.g.
        Input: ['l_reach', 'l_reach', 'l_transport_prox', 'l_stabilize', 'r_reach']
            handedness='left
        Output: ['reach', 'transport', 'stabilize']  # ignore the right reach
        """
        df = pd.read_csv(path)
        times = df['Time_s'].tolist()
        actions = df['MarkerNames'].tolist()
        action_seq = []

        def deduped_action_append(action):
            if len(action_seq) == 0:
                action_seq.append(action)
            else:
                if action[1] != action_seq[-1][1]:
                    action_seq.append(action)

        for i, action in enumerate(actions):
            if (handedness == "left" and (action.startswith('l_') or '_l_' in action)) \
                or (handedness == "right" and (action.startswith('r_') or '_r_' in action)):
                if "reach" in action:
                    deduped_action_append((times[i], "reach"))
                elif "reposition" in action or "retract" in action:
                    deduped_action_append((times[i], "reposition"))
                elif "transport" in action:
                    deduped_action_append((times[i], "transport"))
                elif "stabilize" in action:
                    deduped_action_append((times[i], "stabilize"))
                elif "idle" in action or "rest" in action:
                    deduped_action_append((times[i], "idle"))
        return action_seq


##################### Path and File Utilities #####################
class DataPaths:

    DATA_DIR = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/"
    RAW_VIDEO_DIR = os.path.join(DATA_DIR, "VideoData/rawVideosADLsandFM")
    RAW_LABEL_DIR = os.path.join(DATA_DIR, "rawVideoLabels")

    METADATA_DIR = os.path.join(DATA_DIR, "metadata")
    METADATA_PATH = os.path.join(METADATA_DIR, "metadata.csv")
    VIDEO_METADATA_PATH = os.path.join(METADATA_DIR, "video_metadata.csv")
    LABEL_METADATA_PATH = os.path.join(METADATA_DIR, "label_metadata.csv")

    VERIFICATION_PATH = os.path.join(DATA_DIR, "video_n_labels")
