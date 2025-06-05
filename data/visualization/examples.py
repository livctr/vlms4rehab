from data.visualization import (
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


if __name__ == "__main__":

    fps = 15.0
    sample_rate = 1.0 / fps
    video_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00011/C00011_brushing1_1.mkv"
    label_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00011/C00011_brushing1_1.csv"
    output_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/AnnotatedVideoData/VideosWithLabels/test.mp4"

    write_video_with_label("C00011 Brushing 1", video_path, label_path, output_path, fps)
