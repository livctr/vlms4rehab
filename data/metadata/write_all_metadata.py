"""
Merges action label and video metadata. Only keeps pairs with exact alignment, i.e.,
the number of labels equals the number of frames in video's metadata and from
looping. See `nbs/merge_metadata.ipynb`.
"""
from .write_action_metadata import write_label_metadata
from .write_video_metadata import write_video_metadata

import pandas as pd

from data.utils import extract_folder_file_from_path
from data.utils_strokerehab import DataPaths


def filter_metadata(df):
    """Filter out videos with markers on both left and right arms, and videos
    focusing on a stroke patient's unaffected side.

    Don't use this.
    """

    def retrieve_handedness(path):
        labels = pd.read_csv(path)
        marker_names = labels['MarkerNames']
        has_l_prefix = marker_names.str.startswith('l_').sum() + marker_names.str.contains('_l_').sum()
        has_r_prefix = marker_names.str.startswith('r_').sum() + marker_names.str.contains('_r_').sum()
        return has_l_prefix, has_r_prefix

    df[['n_left', 'n_right']] = df['path_l'].apply(retrieve_handedness).apply(pd.Series)

    # Ignore videos with both left and right markers (114/5775 videos)
    df = df[(df['n_left'] == 0) | (df['n_right'] == 0)].copy()
    df['right_focused'] = df['n_right'] > 0
    df['left_focused'] = df['n_left'] > 0

    # Filter out videos focusing on a stroke patient's unaffected side (6/5661)
    patient_focus = df.groupby('patient').agg({'right_focused': 'sum', 'left_focused': 'sum'}).sort_values('patient').reset_index()
    focused_arm = {}
    for _, x in patient_focus.iterrows():
        if x['right_focused'] > x['left_focused']:
            focused_arm[x['patient']] = 'right'
        else:
            focused_arm[x['patient']] = 'left'

    def is_focused_arm(row):
        if row['right_focused'] and focused_arm[row['patient']] == 'right':
            return True
        elif row['left_focused'] and focused_arm[row['patient']] == 'left':
            return True
        return False

    df['is_focused_arm'] = df.apply(is_focused_arm, axis=1)

    affected_side_filter = (
        (df['stroke'] & df['is_focused_arm']) |
        (~df['stroke'])
    )
    df = df[affected_side_filter].copy()

    # clean up the columns
    df.drop(columns=['n_left', 'n_right', 'right_focused', 'left_focused', 'is_focused_arm'], inplace=True)

    return df  # 5655 videos


def add_is_in_strokerehab_test_set_col(df):
    """Reads the lines of `data/metadata/strokerehab_test_set.txt`
    and adds a column to the dataframe `df` that indicates whether
    the video is in the stroke rehab test set. Each line may look
    like this: `S00044_brushing3_1`
    """
    df['is_in_strokerehab_test_set'] = False
    
    with open('data/metadata/strokerehab_test_set.txt', 'r') as f:
        test_set_lines = [line.strip() for line in f.readlines()]
    
    # Track which test lines matched
    matched_lines = set()
    
    def normalize_str(s):
        """Replace spaces and underscores with a common character"""
        # Get basename before extension and normalize
        base = s.split('/')[-1].split('.')[0]
        return base.replace(' ', '_').replace('__', '_')
    
    def check_path(path):
        path_normalized = normalize_str(path)
        for test_str in test_set_lines:
            test_normalized = normalize_str(test_str)
            if test_normalized in path_normalized:
                matched_lines.add(test_str)
                return True
        return False
    
    df['is_in_strokerehab_test_set'] = df['path_v'].apply(check_path)
    
    # Print unmatched lines
    unmatched = set(test_set_lines) - matched_lines
    if unmatched:
        print("Unmatched test set lines:")
        for line in sorted(unmatched):
            print(f"  {line}")
    else:
        print("All test set lines matched!")
    
    return df


def merge_metadata(video_path, label_path, out_path, filter_two_handed=False):

    # Video metadata
    vdf = pd.read_csv(video_path)
    vdf['id'] = vdf['path'].apply(lambda x: extract_folder_file_from_path(x)[1].split('.')[0])
    # filter for videos contained in expected locations (~ -7 videos of S00010 found in S00020)
    mode_path_length = vdf['path'].apply(lambda x: len(x.split('/'))).value_counts().index[0]
    vdf = vdf[vdf['path'].apply(lambda x: len(x.split('/')) == mode_path_length)]

    # Found that videos w/ the same id are duplicates (defined as same metadata and same first frame)
    ids = vdf['id'].value_counts().values
    if len(ids) > 0:
        assert ids[0] <= 2  # at most 2 videos per id

    # Filter out duplicates, filter out first from MissingVideos
    vdf['inMissingVideo'] = vdf['path'].str.contains('MissingBigPurplevideos')
    vdf = vdf.sort_values('inMissingVideo', ascending=True)
    vdf = vdf[~vdf['id'].duplicated()].copy()  # drop duplicates
    vdf.drop(columns='inMissingVideo', inplace=True)

    # Label metadata
    ldf = pd.read_csv(label_path)
    ldf['id'] = ldf['path'].apply(lambda x: extract_folder_file_from_path(x)[1].split('.')[0])
    assert len(ldf['id'].unique()) == len(ldf)  # all unique ids!

    # Merge metadata
    df = pd.merge(ldf, vdf, how='left', on='id', suffixes=('_l', '_v'))

    # Filter metadata
    if filter_two_handed:
        print("Filtering metadata...")
        df = filter_metadata(df)
    

    add_is_in_strokerehab_test_set_col(df)

    # Save
    df.to_csv(out_path, index=False)


if __name__ == "__main__":

    # The header for the csvs is either 'Time_s,Time_s (Rounded),MarkerNames' or 'Time_s,MarkerNames'. Always.
    # Are there any gaps in 'Time_s'? Yes.
    write_label_metadata(verbose=True)

    write_video_metadata()

    merge_metadata(DataPaths.VIDEO_METADATA_PATH, DataPaths.LABEL_METADATA_PATH, DataPaths.METADATA_PATH)

    # Inspection
    import pandas as pd
    df = pd.read_csv(DataPaths.METADATA_PATH)
    import pdb; pdb.set_trace()
    df.head()
