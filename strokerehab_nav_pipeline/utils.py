import os

import pandas as pd

##################### Label Utilities #####################

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