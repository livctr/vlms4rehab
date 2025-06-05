from data.visualization import (
    BoxRenderer,
    COCOKeypoints3DRenderer,
    COCOKeypointsRenderer,
    FrameRenderer,
    ProgressRenderer,
    TextRenderer,
    DecordVideoStreamer,
    ProgressStreamer,
    StaticStreamer,
    StaticHorizontalLabelBarStreamer,
    TabularStreamer,
    LeafNode,
    CompositeNode,
    VideoAnnotationWriter
)

from data.utils_strokerehab import LabelUtils


def write_video_with_label(title, video_path, label_path, output_path, fps):

    action_seq = LabelUtils.convert_labels_to_action_sequence(label_path)
    action_seq = {"Time_s": [action[0] for action in action_seq],
                  "MarkerNames": [action[1] for action in action_seq]}

    # Create text nodes for title and labels with larger font
    title_node = LeafNode("title", TextRenderer(StaticStreamer(title), font_scale=2.0))
    label_node = LeafNode("label", TextRenderer(TabularStreamer(action_seq, "MarkerNames", "Time_s"), font_scale=2.0))

    # Create progress bar to show position in video timeline
    progress_node = LeafNode("progress_bar", FrameRenderer(StaticHorizontalLabelBarStreamer(label_path, "MarkerNames", "Time_s")))

    # Set up video frame display
    frame_streamer = DecordVideoStreamer(video_path, read_from_cpu_id=1)
    frame_node = LeafNode("video", FrameRenderer(frame_streamer))

    # Add marker to show current position on progress bar
    progress_marker_node = LeafNode("progress_marker", ProgressRenderer(ProgressStreamer(frame_streamer)))

    # Arrange nodes in a grid layout:
    # The positions are specified as (layer, row, column)
    # Layer 0: Title, Label, Progress bar, Video frame
    # Layer 1: Progress marker
    root_node = CompositeNode()
    root_node.add_children(
        [title_node, label_node, progress_node, frame_node, progress_marker_node],
        [(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 0), (1, 2, 0)]
    )
    root_node.compile(1.0 / fps)
    writer = VideoAnnotationWriter(root_node)
    writer.write_to_file(output_path, num_render_workers=1, use_multiprocessing=False)


if __name__ == "__main__":

    fps = 15.0
    sample_rate = 1.0 / fps
    video_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00011/C00011_brushing1_1.mkv"
    label_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/rawVideoLabels/C00011/C00011_brushing1_1.csv"
    output_path = "/gpfs/data/schambralab/quantitativeRehabilitation/__data/video_n_labels/test.mp4"

    write_video_with_label("C00011 Brushing 1", video_path, label_path, output_path, fps)
