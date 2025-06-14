from data.visualization import (
    BoxRenderer,
    FrameRenderer,
    HorizontalLabelBarRenderer,
    ProgressRenderer,
    TextRenderer,
    DecordVideoStreamer,
    ProgressStreamer,
    StaticStreamer,
    StaticTabularStreamer,
    TabularStreamer,
    LeafOrchestrator,
    CompositeOrchestrator,
    write_to_file
)

from data.utils_strokerehab import LabelUtils


def write_video_with_label(title_str, video_path, label_path, output_path, fps):

    handedness = LabelUtils.get_handedness(label_path)
    action_seq = LabelUtils.convert_labels_to_action_sequence(label_path)
    action_seq = {"Time_s": [action[0] for action in action_seq],
                  "MarkerNames": [f"Label ({handedness} hand): {action[1]}"  for action in action_seq]}
    print(action_seq)
    # Create text nodes for title and labels with larger font
    title = LeafOrchestrator(
        "title",
        StaticStreamer(f"Title: {title_str}"),
        TextRenderer(font_scale=0.5)
    )
    label = LeafOrchestrator(
        "label",
        TabularStreamer(action_seq, "MarkerNames", "Time_s", stream_method="nearest_left"),
        TextRenderer(font_scale=0.5)
    )

    frame_streamer = DecordVideoStreamer(video_path, read_from_cpu_id=0)
    frame = LeafOrchestrator(
        "frame",
        frame_streamer,
        FrameRenderer(sized_streamer=frame_streamer)  # Get size info from streamer
    )

    label_bar = LeafOrchestrator(
        "label_bar",
        StaticTabularStreamer(action_seq, "MarkerNames", "Time_s", num_samples=500, subsample_method="nearest_left"),
        HorizontalLabelBarRenderer(
            height=20
        )
    )

    label_time_marker = LeafOrchestrator(
        "label_time_marker",
        ProgressStreamer(frame_streamer),
        ProgressRenderer()
    )

    root_orchestrator = CompositeOrchestrator()
    root_orchestrator.add_children(
        [title, label, frame, label_bar, label_time_marker],
        [(0, 0, 0), (0, 1, 0), (0, 3, 0), (0, 2, 0), (1, 2, 0)]
    )
    root_orchestrator.compile(sample_rate=1.0 / fps)
    write_to_file(root_orchestrator, output_path, num_render_workers=4, use_multiprocessing=False, fourcc_str='mp4v')



def write_video_with_bbox(title_str, video_path, label_path, detections_path, output_path, fps):

    # /gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00012/C00012_deodrant1_2.mkv

    handedness = LabelUtils.get_handedness(label_path)
    action_seq = LabelUtils.convert_labels_to_action_sequence(label_path)
    action_seq = {"Time_s": [action[0] for action in action_seq],
                  "MarkerNames": [f"Label ({handedness} hand): {action[1]}"  for action in action_seq]}
    print(action_seq)
    # Create text nodes for title and labels with larger font
    title = LeafOrchestrator(
        "title",
        StaticStreamer(f"Title: {title_str}"),
        TextRenderer(font_scale=0.5)
    )
    label = LeafOrchestrator(
        "label",
        TabularStreamer(action_seq, "MarkerNames", "Time_s", stream_method="nearest_left"),
        TextRenderer(font_scale=0.5)
    )

    frame_streamer = DecordVideoStreamer(video_path, read_from_cpu_id=0)
    frame = LeafOrchestrator(
        "frame",
        frame_streamer,
        FrameRenderer(sized_streamer=frame_streamer)  # Get size info from streamer
    )

    label_bar = LeafOrchestrator(
        "label_bar",
        StaticTabularStreamer(action_seq, "MarkerNames", "Time_s", num_samples=500, subsample_method="nearest_left"),
        HorizontalLabelBarRenderer(
            height=20
        )
    )

    label_time_marker = LeafOrchestrator(
        "label_time_marker",
        ProgressStreamer(frame_streamer),
        ProgressRenderer()
    )

    bbox = LeafOrchestrator(
        "bbox",
        # timestamps
        TabularStreamer(detections_path, "detections", "times", stream_method="nearest"),
        BoxRenderer(
            color=(0, 255, 0),  # Green
            thickness=2
        )
    )

    root_orchestrator = CompositeOrchestrator()
    root_orchestrator.add_children(
        [title, label, frame, bbox, label_bar, label_time_marker],
        [(0, 0, 0), (0, 1, 0), (0, 3, 0), (1, 3, 0), (0, 2, 0), (1, 2, 0)]
    )
    root_orchestrator.compile(sample_rate=1.0 / fps)
    write_to_file(root_orchestrator, output_path, num_render_workers=4, use_multiprocessing=False, fourcc_str='mp4v')


if __name__ == "__main__":

    fps = 15.0
    sample_rate = 1.0 / fps
    # video_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00012/C00012_deodrant1_2.mkv"
    # label_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00012/C00012_deodrant1_1.csv"
    # output_path = "./detection_example_C00012_deodrant1_1.mp4"

    test_video_paths = [
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_brushing1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_brushing1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_combing1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_combing1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_deodrant1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_deodrant1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_drinking1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_drinking1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_face wash1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_face wash1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_feeding1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_feeding1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_glasses1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_glasses1_2.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_RTT left side1_1.avi",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_RTT left side1_2.avi",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_shelf right side1_1.mkv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00015/C00015_shelf right side1_1.mkv",
        "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00012/C00012_deodrant1_2.mkv",
    ]
    test_label_paths = [
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_brushing1_1.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_brushing1_2.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_combing1_1.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_combing1_2.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_deodrant1_1.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_deodrant1_2.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_drinking1_1.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_drinking1_2.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_face wash1_1.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_face wash1_2.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_feeding1_1.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_feeding1_2.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_glasses1_1.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_glasses1_2.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_RTT left side1_1.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_RTT left side1_2.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_shelf right side1_1.csv",
        # "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00015/C00015_shelf right side1_1.csv",
        "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00012/C00012_deodrant1_2.csv",
    ]

    for test_video_path, test_label_path in zip(test_video_paths, test_label_paths):
        try:
            video_name = test_video_path.split('/')[-1].split('.')[0]
            output_path = f"./test_videos_out/detection_example_{video_name}.mp4"
            detections_path = "./detection_results_crop.json"
            # detections_path = f"./test_videos_out/gd_with_cropping_{video_name}.json"
            write_video_with_bbox(video_name, test_video_path, test_label_path, detections_path, output_path, fps)
            print(f"Write success for {test_video_path}!")
        except Exception as e:
            print(f"Error writing video with bbox for {test_video_path}: {e}")


    # fps = 15.0
    # sample_rate = 1.0 / fps
    # video_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00011/C00011_brushing1_1.mkv"
    # label_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00011/C00011_brushing1_1.csv"
    # output_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/AnnotatedVideoData/VideosWithLabels/test.mp4"

    # write_video_with_label("C00011 Brushing 1", video_path, label_path, output_path, fps)
