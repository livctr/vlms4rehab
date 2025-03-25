from functools import partial

import datasets
from loguru import logger as eval_logger
import pandas as pd


METADATA_PATH = "./data/csvs_txts_yamls/cleaned_metadata.csv"
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
            This data is located in './data/csvs_txts_yamls/strokerehab_test_set.txt' and already
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

strokerehab_load_dataset_debug = partial(strokerehab_load_dataset, patients='S0001', activity='face wash,glasses', reps='first')
strokerehab_load_dataset_S0001_small = partial(strokerehab_load_dataset, patients='S0001', reps='first')
strokerehab_load_dataset_S0001 = partial(strokerehab_load_dataset, patients='S0001')
strokerehab_load_dataset_3patients = partial(strokerehab_load_dataset, patients='C00011,S0001,S0002')
strokerehab_load_dataset_onerep = partial(strokerehab_load_dataset, reps='first')
strokerehab_load_dataset_test = partial(strokerehab_load_dataset, filter_for_testset=True)
strokerehab_load_dataset_healthy = partial(strokerehab_load_dataset, patients=HEALTHY_PATIENTS)
strokerehab_load_dataset_mild = partial(strokerehab_load_dataset, patients=MILD_PATIENTS)
strokerehab_load_dataset_moderate = partial(strokerehab_load_dataset, patients=MODERATE_PATIENTS)
strokerehab_load_dataset_severe = partial(strokerehab_load_dataset, patients=SEVERE_PATIENTS)
