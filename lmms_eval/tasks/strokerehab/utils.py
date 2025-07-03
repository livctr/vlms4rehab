from functools import partial

from data.utils_strokerehab import (
    strokerehab_load_dataset,
    HEALTHY_PATIENTS,
    MILD_PATIENTS,
    MODERATE_PATIENTS,
    SEVERE_PATIENTS,
)

summarization_activities = 'brushing,combing,deodrant,drinking,face wash,feeding,glasses'

strokerehab_load_C00015_dataset = partial(strokerehab_load_dataset,
                                     patients='C00015',
                                     activity='brushing,combing,deodrant,drinking,feeding,glasses',
                                     reps='first')

strokerehab_load_dataset_debug = partial(strokerehab_load_dataset, patients='S0001', activity='face wash', reps='first')
strokerehab_load_dataset_S0001_small = partial(strokerehab_load_dataset, patients='S0001', reps='first')
strokerehab_load_dataset_S0001 = partial(strokerehab_load_dataset, patients='S0001')
strokerehab_load_dataset_3patients = partial(strokerehab_load_dataset, patients='C00011,S0001,S0002')
strokerehab_load_dataset_onerep = partial(strokerehab_load_dataset, reps='first')
strokerehab_load_dataset_test = partial(strokerehab_load_dataset, filter_for_testset=True)
strokerehab_load_dataset_test_subset = partial(strokerehab_load_dataset, filter_for_subsampled_testset=True)
strokerehab_load_dataset_healthy = partial(strokerehab_load_dataset, patients=HEALTHY_PATIENTS)
strokerehab_load_dataset_mild = partial(strokerehab_load_dataset, patients=MILD_PATIENTS)
strokerehab_load_dataset_moderate = partial(strokerehab_load_dataset, patients=MODERATE_PATIENTS)
strokerehab_load_dataset_severe = partial(strokerehab_load_dataset, patients=SEVERE_PATIENTS)

strokerehab_load_summarization_data = partial(strokerehab_load_dataset, patients='S0001', activity=summarization_activities, reps='first')
strokerehab_load_primitives_data = partial(strokerehab_load_dataset, patients='S0001', activity='brushing,combing', reps='first')