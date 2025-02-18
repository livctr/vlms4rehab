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


def merge_metadata(video_path, label_path, out_path):

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
    print("Filtering metadata...")
    df = filter_metadata(df)

    # Save
    df.to_csv(out_path, index=False)


if __name__ == "__main__":

    # The header for the csvs is either 'Time_s,Time_s (Rounded),MarkerNames' or 'Time_s,MarkerNames'. Always.
    # Are there any gaps in 'Time_s'? Yes.
    write_label_metadata(verbose=True)

    write_video_metadata()

    merge_metadata(DataPaths.VIDEO_METADATA_PATH, DataPaths.LABEL_METADATA_PATH, DataPaths.METADATA_PATH)

    # Inspection
    # import pandas as pd
    # import pdb ; pdb.set_trace()
    # df = pd.read_csv(DataPaths.METADATA_PATH)
    # df.head()
