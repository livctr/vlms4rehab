"""
Creates the metadata for the functional primitives task. See
"./data/fp/fp_metadata.csv" for the final output.
"""
import cv2
import logging
import pandas as pd
import regex as re
from collections import Counter
from functools import partial

from data.utils import extract_folder_file_from_path, write_metadata
from data.utils_strokerehab import DataPaths


def cnt_frames_av(path):
    import av
    container = av.open(path)
    return sum(1 for _ in container.decode(video=0))


def get_codec_av(file_path):
    import av
    container = av.open(file_path)
    video_stream = next((s for s in container.streams if s.type == 'video'), None)
    codec = video_stream.codec_context.name if video_stream else None
    return codec


def get_video_info(path):
    try:
        cap = cv2.VideoCapture(path)
        cv2_nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Count frames using cv2 while loop
        while_loop_cnt = 0
        while True:
            ret, _ = cap.read()
            if not ret:
                break
            while_loop_cnt += 1
        
        # get count from av
        av_cnt = cnt_frames_av(path)

        cap.release()
        return {
            "path": path,
            "codec": get_codec_av(path),
            "fps": fps,
            "height": height,
            "width": width,
            "duration": cv2_nframes / fps if fps > 0 else None,
            "av_nframes": av_cnt,
            "cv2_nframes": cv2_nframes,
            "cv2_nframes_while_loop": while_loop_cnt,
            "aligned_nframes": cv2_nframes == while_loop_cnt == av_cnt,
        }
    except Exception as e:
        logging.warning(f"Error processing {path}: {e}")
        return None


def write_video_metadata():
    """
    Writes video metadata in folder `DataPaths.RAW_VIDEO_DIR` to
    `DataPaths.VIDEO_METADATA_PATH`.
    """
    write_metadata(DataPaths.RAW_VIDEO_DIR, DataPaths.VIDEO_METADATA_PATH, get_video_info)


def fill_gap(path: str, fps: int = -1, verbose: bool = False):
    """Returns a list of times and actions from a csv file.

    Some files have gaps in the time. If the surrounding actions are identical,
    the gap is filled with the same action. Otherwise, it is filled with an empty string.

    If `fps` is provided, the function will check if the gap size is consistent with the fps.
    """

    df = pd.read_csv(path)
    times = df['Time_s'].tolist()
    actions = df['MarkerNames'].tolist()

    # Get time differences
    time_diff_ms = [round(1000 * (times[i+1] - times[i])) for i in range(len(times)-1)]

    # Verify gap size with provided fps
    gap_freqs = Counter(time_diff_ms).most_common(2)
    inferred_ms_gap = tuple(gf[0] for gf in gap_freqs)
    if len(inferred_ms_gap) == 1:
        inferred_ms_gap = (inferred_ms_gap[0], inferred_ms_gap[0])
    if fps != -1:
        expected_ms_gaps = (int(1000 / fps), int(1000 / fps) + 1)
        # no overlap
        if inferred_ms_gap[0] not in expected_ms_gaps and \
            inferred_ms_gap[1] not in expected_ms_gaps:
            logging.warning(f'Gap in data {inferred_ms_gap}-ms does not match expected '
                            f'gap {expected_ms_gaps} from fps={fps} for {path}.')

    gap_idxs = []
    for i, t in enumerate(time_diff_ms):
        if t != inferred_ms_gap[0] and t != inferred_ms_gap[1]:
            gap_idxs.append(i)
            if verbose:
                gap_print_out = list(zip(times[i-1:i+3], actions[i-1:i+3])) \
                    if i > 1 and i+3 <= len(times) else ""
                logging.warning(f'Filled unexpected gap {t}-ms (fps={fps}) between frames '
                                f'at index {i} and {i+1} in {path}: {gap_print_out}.')

    if verbose and len(gap_idxs) > 3:
        logging.warning(f'Found {len(gap_idxs)} unexpected gaps in the file. Please manually check {path}.')

    for i in gap_idxs[::-1]:
        times.insert(i+1, round(times[i] + 1 / fps, 3))
        if actions[i] == actions[i+1]:
            actions.insert(i+1, actions[i])
        else:
            actions.insert(i+1, '')

    return times, actions


def get_label_info(path: str, fps: int = -1, verbose: bool = False):
    """Get csv label info. Use nonnegative fps to verify time gaps."""
    times, _ = fill_gap(path, fps, verbose)

    folder, file = extract_folder_file_from_path(path)
    patient = folder
    stroke = folder[0] == 'S'
    activity = re.sub(r'\d', '', file.split('_')[1])

    return {
        "path": path, "tstart": times[0], "tend": times[-1], "nlabels": len(times),
        "patient": patient, "stroke": stroke, "activity": activity,
    }


def write_label_metadata(**kwargs):
    """
    Writes video metadata in folder `DataPaths.RAW_LABEL_DIR` to
    `DataPaths.LABEL_METADATA_PATH`. See `get_label_info` for kwargs.
    """
    fn = partial(get_label_info, **kwargs)
    write_metadata(DataPaths.RAW_LABEL_DIR, DataPaths.LABEL_METADATA_PATH, fn)
    print(f"Wrote metadata to {DataPaths.LABEL_METADATA_PATH}.")


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
    
    with open('data/csvs_and_txts/strokerehab_test_set.txt', 'r') as f:
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

    def get_folder_file(path):
        folder, file = extract_folder_file_from_path(path)
        return folder + '/' + file

    df['path_v'] = df['path_v'].apply(get_folder_file)
    df['path_l'] = df['path_l'].apply(get_folder_file)

    # Filter metadata
    if filter_two_handed:
        print("Filtering metadata...")
        df = filter_metadata(df)
    
    add_is_in_strokerehab_test_set_col(df)

    # Save
    df.to_csv(out_path, index=False)


def clean_metadata():
    df = pd.read_csv(DataPaths.METADATA_PATH)
    df['duration_s'] = df['tend'] - df['tstart']
    df.drop(columns=['duration', 'tstart', 'tend', 'codec', 'av_nframes', 'cv2_nframes', 'cv2_nframes_while_loop', 'aligned_nframes'], inplace=True)

    cols = [
        'id',
        'is_in_strokerehab_test_set',
        'path_v',
        'patient',
        'stroke',
        'activity',
        'fps',
        'height',
        'width',
        'duration_s',
        'path_l',
        'nlabels',
    ]
    df = df[cols]

    # Cut off bases for path_v and path_l
    import os
    df['path_v'] = df['path_v'].apply(lambda x: os.path.relpath(x, DataPaths.RAW_VIDEO_DIR))
    df['path_l'] = df['path_l'].apply(lambda x: os.path.relpath(x, DataPaths.RAW_LABEL_DIR))

    # Add a column for subsampling within the original strokerehab test set
    test_set_df = df[df["is_in_strokerehab_test_set"] == True]
    sample_indexes = test_set_df.sample(n=50, random_state=42).index
    df['subsampled_test_set'] = False
    df.loc[sample_indexes, 'subsampled_test_set'] = True

    # # take the first repetition of each activity
    # subset_df = df.sort_values('id').groupby(['patient', 'activity']).agg('first').reset_index()
    # subset_df = subset_df[cols]

    df.to_csv("./data/fp/fp_metadata.csv", index=False)
    # subset_df.to_csv("./data/csvs_and_txts/fp_metadata_subset.csv", index=False)


if __name__ == "__main__":

    # The header for the csvs is either 'Time_s,Time_s (Rounded),MarkerNames' or 'Time_s,MarkerNames'. Always.
    # Are there any gaps in 'Time_s'? Yes.
    write_label_metadata(verbose=True)

    write_video_metadata()

    merge_metadata(DataPaths.VIDEO_METADATA_PATH, DataPaths.LABEL_METADATA_PATH, DataPaths.METADATA_PATH)

    clean_metadata()

    # # Inspection
    # import pandas as pd
    # df = pd.read_csv(DataPaths.METADATA_PATH)
    # import pdb; pdb.set_trace()
    # df.head()
