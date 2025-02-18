import os
import pandas as pd


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

        if num_lefts > 0 and num_rights > 0:
            raise ValueError(f"Both left and right markers found in {path}.")
        
        if num_lefts == 0 and num_rights == 0:
            raise ValueError(f"No markers with handedness labels found in {path}.")
        
        if num_lefts > 0:
            return "left"
        return "right"

    @staticmethod
    def get_df_with_cleaned_labels(path):
        """Returns the DataFrame at the path with cleaned labels."""
        df = pd.read_csv(path)
        actions = df['MarkerNames'].tolist()
        cleaned_actions = ["" for _ in range(len(actions))]

        for i, action in enumerate(actions):
            if "reach" in action:
                cleaned_actions[i] = "reach"
            elif "reposition" in action or "retract" in action:
                cleaned_actions[i] = "reposition"
            elif "transport" in action:
                cleaned_actions[i] = "transport"
            elif "stabilize" in action:
                cleaned_actions[i] = "stabilize"
            elif "idle" in action or "rest" in action:
                cleaned_actions[i] = "idle"

        df['MarkerNames'] = cleaned_actions
        return df


##################### Path and File Utilities #####################
class DataPaths:

    DATA_DIR = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/"
    RAW_VIDEO_DIR = os.path.join(DATA_DIR, "VideoData/rawVideosADLsandFM")
    RAW_LABEL_DIR = os.path.join(DATA_DIR, "rawVideoLabels")

    METADATA_DIR = os.path.join(DATA_DIR, "metadata")
    METADATA_PATH = os.path.join(METADATA_DIR, "metadata.csv")

    VERIFICATION_PATH = os.path.join(DATA_DIR, "video_n_labels")
