import sys
import argparse
import cv2
import os
import numpy as np
import torch
import torch.nn as nn
import glob
from tqdm import tqdm
import copy

from tools.motionagformer.lib.utils import normalize_screen_coordinates, camera_to_world
from tools.motionagformer.lib.preprocess import h36m_coco_format, revise_kpts
from tools.motionagformer.model.MotionAGFormer import MotionAGFormer
from tools.hrnet.gen_kpts import gen_video_kpts as hrnet_pose

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

def show2Dpose(kps, img):
    connections = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5],
                   [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                   [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]

    LR = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0], dtype=bool)

    lcolor = (255, 0, 0)
    rcolor = (0, 0, 255)
    thickness = 3

    for j,c in enumerate(connections):
        start = map(int, kps[c[0]])
        end = map(int, kps[c[1]])
        start = list(start)
        end = list(end)
        cv2.line(img, (start[0], start[1]), (end[0], end[1]), lcolor if LR[j] else rcolor, thickness)
        cv2.circle(img, (start[0], start[1]), thickness=-1, color=(0, 255, 0), radius=3)
        cv2.circle(img, (end[0], end[1]), thickness=-1, color=(0, 255, 0), radius=3)

    return img


def show3Dpose(vals, ax):
    ax.view_init(elev=15., azim=70)

    lcolor=(0,0,1)
    rcolor=(1,0,0)

    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    LR = np.array([0, 1, 0, 1, 0, 1, 0, 0, 0,   1,  0,  0,  1,  1, 0, 0], dtype=bool)

    for i in np.arange( len(I) ):
        x, y, z = [np.array( [vals[I[i], j], vals[J[i], j]] ) for j in range(3)]
        ax.plot(x, y, z, lw=2, color = lcolor if LR[i] else rcolor)

    RADIUS = 0.72
    RADIUS_Z = 0.7

    xroot, yroot, zroot = vals[0,0], vals[0,1], vals[0,2]
    ax.set_xlim3d([-RADIUS+xroot, RADIUS+xroot])
    ax.set_ylim3d([-RADIUS+yroot, RADIUS+yroot])
    ax.set_zlim3d([-RADIUS_Z+zroot, RADIUS_Z+zroot])
    ax.set_aspect('auto') # works fine in matplotlib==2.2.2

    white = (1.0, 1.0, 1.0, 0.0)
    ax.xaxis.set_pane_color(white)
    ax.yaxis.set_pane_color(white)
    ax.zaxis.set_pane_color(white)

    ax.tick_params('x', labelbottom = False)
    ax.tick_params('y', labelleft = False)
    ax.tick_params('z', labelleft = False)


def get_pose2D(video_path, output_dir):
    import pdb ; pdb.set_trace()
    cap = cv2.VideoCapture(video_path)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    print('\nGenerating 2D pose...')
    keypoints, scores = hrnet_pose(video_path, det_dim=416, num_peroson=1, gen_output=True)
    keypoints, scores, valid_frames = h36m_coco_format(keypoints, scores)

    # Add conf score to the last dim
    keypoints = np.concatenate((keypoints, scores[..., None]), axis=-1)

    output_dir += 'input_2D/'
    os.makedirs(output_dir, exist_ok=True)

    output_npz = output_dir + 'keypoints.npz'
    np.savez_compressed(output_npz, reconstruction=keypoints)


def img2video(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS)) + 5

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    names = sorted(glob.glob(os.path.join(output_dir + 'pose/', '*.png')))
    img = cv2.imread(names[0])
    size = (img.shape[1], img.shape[0])

    videoWrite = cv2.VideoWriter(output_dir + "sample_video" + '.mp4', fourcc, fps, size)

    for name in names:
        img = cv2.imread(name)
        videoWrite.write(img)

    videoWrite.release()


def showimage(ax, img):
    ax.set_xticks([])
    ax.set_yticks([])
    plt.axis('off')
    ax.imshow(img)


def resample(n_frames):
    even = np.linspace(0, n_frames, num=243, endpoint=False)
    result = np.floor(even)
    result = np.clip(result, a_min=0, a_max=n_frames - 1).astype(np.uint32)
    return result


def turn_into_clips(keypoints):
    clips = []
    n_frames = keypoints.shape[1]
    if n_frames <= 243:
        new_indices = resample(n_frames)
        clips.append(keypoints[:, new_indices, ...])
        downsample = np.unique(new_indices, return_index=True)[1]
    else:
        for start_idx in range(0, n_frames, 243):
            keypoints_clip = keypoints[:, start_idx:start_idx + 243, ...]
            clip_length = keypoints_clip.shape[1]
            if clip_length != 243:
                new_indices = resample(clip_length)
                clips.append(keypoints_clip[:, new_indices, ...])
                downsample = np.unique(new_indices, return_index=True)[1]
            else:
                clips.append(keypoints_clip)
    return clips, downsample

def turn_into_h36m(keypoints):
    new_keypoints = np.zeros_like(keypoints)
    new_keypoints[..., 0, :] = (keypoints[..., 11, :] + keypoints[..., 12, :]) * 0.5
    new_keypoints[..., 1, :] = keypoints[..., 11, :]
    new_keypoints[..., 2, :] = keypoints[..., 13, :]
    new_keypoints[..., 3, :] = keypoints[..., 15, :]
    new_keypoints[..., 4, :] = keypoints[..., 12, :]
    new_keypoints[..., 5, :] = keypoints[..., 14, :]
    new_keypoints[..., 6, :] = keypoints[..., 16, :]
    new_keypoints[..., 8, :] = (keypoints[..., 5, :] + keypoints[..., 6, :]) * 0.5
    new_keypoints[..., 7, :] = (new_keypoints[..., 0, :] + new_keypoints[..., 8, :]) * 0.5
    new_keypoints[..., 9, :] = keypoints[..., 0, :]
    new_keypoints[..., 10, :] = (keypoints[..., 1, :] + keypoints[..., 2, :]) * 0.5
    new_keypoints[..., 11, :] = keypoints[..., 6, :]
    new_keypoints[..., 12, :] = keypoints[..., 8, :]
    new_keypoints[..., 13, :] = keypoints[..., 10, :]
    new_keypoints[..., 14, :] = keypoints[..., 5, :]
    new_keypoints[..., 15, :] = keypoints[..., 7, :]
    new_keypoints[..., 16, :] = keypoints[..., 9, :]

    return new_keypoints


def flip_data(data, left_joints=[1, 2, 3, 14, 15, 16], right_joints=[4, 5, 6, 11, 12, 13]):
    """
    data: [N, F, 17, D] or [F, 17, D]
    """
    flipped_data = copy.deepcopy(data)
    flipped_data[..., 0] *= -1  # flip x of all joints
    flipped_data[..., left_joints + right_joints, :] = flipped_data[..., right_joints + left_joints, :]  # Change orders
    return flipped_data

@torch.no_grad()
def get_pose3D(video_path, output_dir):
    args, _ = argparse.ArgumentParser().parse_known_args()
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 26, 3, 128, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 243
    args = vars(args)

    ## Reload
    model = nn.DataParallel(MotionAGFormer(**args)).cuda()

    # Put the pretrained model of MotionAGFormer in 'checkpoint/'
    model_path = sorted(glob.glob(os.path.join('tools', 'motionagformer', 'checkpoint', 'motionagformer-l-h36m.pth.tr')))[0]

    pre_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(pre_dict['model'], strict=True)

    model.eval()

    ## input
    keypoints = np.load(output_dir + 'input_2D/keypoints.npz', allow_pickle=True)['reconstruction']
    # keypoints = np.load('demo/lakeside3.npy')
    # keypoints = keypoints[:240]
    # keypoints = keypoints[None, ...]
    # keypoints = turn_into_h36m(keypoints)


    clips, downsample = turn_into_clips(keypoints)


    cap = cv2.VideoCapture(video_path)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ## 3D
    print('\nGenerating 2D pose image...')
    for i in tqdm(range(video_length)):
        ret, img = cap.read()
        if img is None:
            continue
        img_size = img.shape

        input_2D = keypoints[0][i]

        image = show2Dpose(input_2D, copy.deepcopy(img))

        output_dir_2D = output_dir +'pose2D/'
        os.makedirs(output_dir_2D, exist_ok=True)
        cv2.imwrite(output_dir_2D + str(('%04d'% i)) + '_2D.png', image)


    print('\nGenerating 3D pose...')
    all_poses_3d = []
    for idx, clip in enumerate(clips):
        input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0])
        input_2D_aug = flip_data(input_2D)

        input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
        input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

        output_3D_non_flip = model(input_2D)
        output_3D_flip = flip_data(model(input_2D_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        output_3D[:, :, 0, :] = 0
        post_out_all = output_3D[0].cpu().detach().numpy()
        all_poses_3d.append(post_out_all)

        for j, post_out in enumerate(post_out_all):
            rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
            rot = np.array(rot, dtype='float32')
            post_out = camera_to_world(post_out, R=rot, t=0)
            post_out[:, 2] -= np.min(post_out[:, 2])
            max_value = np.max(post_out)
            post_out /= max_value

            fig = plt.figure(figsize=(9.6, 5.4))
            gs = gridspec.GridSpec(1, 1)
            gs.update(wspace=-0.00, hspace=0.05)
            ax = plt.subplot(gs[0], projection='3d')
            show3Dpose(post_out, ax)

            output_dir_3D = output_dir +'pose3D/'
            os.makedirs(output_dir_3D, exist_ok=True)
            str(('%04d'% (idx * 243 + j)))
            plt.savefig(output_dir_3D + str(('%04d'% (idx * 243 + j))) + '_3D.png', dpi=200, format='png', bbox_inches='tight')
            plt.close(fig)


    print('Generating 3D pose successful!')

    ## all
    image_2d_dir = sorted(glob.glob(os.path.join(output_dir_2D, '*.png')))
    image_3d_dir = sorted(glob.glob(os.path.join(output_dir_3D, '*.png')))

    print('\nGenerating demo...')
    for i in tqdm(range(len(image_2d_dir))):
        image_2d = plt.imread(image_2d_dir[i])
        image_3d = plt.imread(image_3d_dir[i])

        ## crop
        edge = (image_2d.shape[1] - image_2d.shape[0]) // 2
        image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

        edge = 130
        image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

        ## show
        font_size = 12
        fig = plt.figure(figsize=(15.0, 5.4))
        ax = plt.subplot(121)
        showimage(ax, image_2d)
        ax.set_title("Input", fontsize = font_size)

        ax = plt.subplot(122)
        showimage(ax, image_3d)
        ax.set_title("Reconstruction", fontsize = font_size)

        ## save
        output_dir_pose = output_dir +'pose/'
        os.makedirs(output_dir_pose, exist_ok=True)
        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(output_dir_pose + str(('%04d'% i)) + '_pose.png', dpi=200, bbox_inches = 'tight')
        plt.close(fig)

    return np.concatenate(all_poses_3d, axis=0) # Return the concatenated 3D poses


def process_2d_to_3d_pose(keypoints_2d: np.ndarray, img_width: int, img_height: int, is_normalized: bool = False, gpu_id: str = '0') -> np.ndarray:
    """
    Processes 2D poses to generate 3D poses using MotionAGFormer.

    Args:
        keypoints_2d (np.ndarray): Input 2D poses as a numpy array.
                                   Shape: (1, T, J, 3) or (T, J, 3).
                                   The last dimension contains (x, y, confidence).
        img_width (int): Width of the original image/video frame.
        img_height (int): Height of the original image/video frame.
        is_normalized (bool, optional): True if x and y coordinates in keypoints_2d are already normalized (0-1),
                                        False otherwise (pixel coordinates). Defaults to False.
        gpu_id (str, optional): GPU ID to use. Defaults to '0'.

    Returns:
        np.ndarray: Output 3D poses as a numpy array.
                    Shape: (T, J, 3) where the last dimension contains (x, y, z).
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id

    # Ensure keypoints_2d is (1, T, J, 3)
    if keypoints_2d.ndim == 3:
        keypoints_2d = np.expand_dims(keypoints_2d, axis=0)
    elif keypoints_2d.ndim != 4 or keypoints_2d.shape[0] != 1:
        raise ValueError("keypoints_2d must have shape (1, T, J, 3) or (T, J, 3)")

    args = argparse.ArgumentParser().parse_known_args()[0]
    args.n_layers, args.dim_in, args.dim_feat, args.dim_rep, args.dim_out = 26, 3, 128, 512, 3
    args.mlp_ratio, args.act_layer = 4, nn.GELU
    args.attn_drop, args.drop, args.drop_path = 0.0, 0.0, 0.0
    args.use_layer_scale, args.layer_scale_init_value, args.use_adaptive_fusion = True, 0.00001, True
    args.num_heads, args.qkv_bias, args.qkv_scale = 8, False, None
    args.hierarchical = False
    args.use_temporal_similarity, args.neighbour_num, args.temporal_connection_len = True, 2, 1
    args.use_tcn, args.graph_only = False, False
    args.n_frames = 243
    args = vars(args)

    model = nn.DataParallel(MotionAGFormer(**args)).cuda()

    model_path = sorted(glob.glob(os.path.join('motionagformer', 'checkpoint', 'motionagformer-l-h36m.pth.tr')))[0]
    pre_dict = torch.load(model_path, weights_only=False)
    model.load_state_dict(pre_dict['model'], strict=True)
    model.eval()

    if not is_normalized:
        input_2D = normalize_screen_coordinates(keypoints_2d, w=img_width, h=img_height)
    else:
        input_2D = keypoints_2d.copy()

    clips, downsample = turn_into_clips(input_2D)
    all_predicted_3d_poses = []

    print('\nGenerating 3D pose from input 2D keypoints...')
    for idx, clip in enumerate(tqdm(clips)):
        import pdb ; pdb.set_trace()
        input_2D_clip = clip[:, :, :2] # only x, y for model input
        input_2D_clip_aug = flip_data(input_2D_clip)

        input_2D_clip = torch.from_numpy(input_2D_clip.astype('float32')).cuda()
        input_2D_clip_aug = torch.from_numpy(input_2D_clip_aug.astype('float32')).cuda()


        output_3D_non_flip = model(input_2D_clip)
        output_3D_flip = flip_data(model(input_2D_clip_aug))
        output_3D = (output_3D_non_flip + output_3D_flip) / 2

        if idx == len(clips) - 1:
            output_3D = output_3D[:, downsample]

        # Set root joint (pelvis) to origin for each frame
        output_3D[:, :, 0, :] = 0

        post_out_all = output_3D[0].cpu().detach().numpy()

        # Apply camera to world transformation (using the hardcoded values from the original demo)
        rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
        rot = np.array(rot, dtype='float32')
        transformed_poses = []
        for pose_frame in post_out_all:
            transformed_pose = camera_to_world(pose_frame, R=rot, t=0)
            transformed_pose[:, 2] -= np.min(transformed_pose[:, 2]) # Shift Z to have minimum at 0
            # Note: The original code scales by max_value after shifting Z.
            # Depending on your application, you might want to adjust this scaling or remove it.
            # max_value = np.max(transformed_pose)
            # transformed_pose /= max_value
            transformed_poses.append(transformed_pose)
        all_predicted_3d_poses.append(np.array(transformed_poses))

    return np.concatenate(all_predicted_3d_poses, axis=0)


if __name__ == "__main__":
    # get_pose2D(
    #     "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00012/C00012_deodrant1_2.mkv",
    #     "./example/test"
    # )
    # get_pose3D(
    #     "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00012/C00012_deodrant1_2.mkv",
    #     "./example/test"
    # )
    img2video(
        "/gpfs/data/schambralab/quantitativeRehabilitation/__data/VideoData/rawVideosADLsandFM/C00012/C00012_deodrant1_2.mkv",
        "./example/test"
    )
    exit()

    parser = argparse.ArgumentParser()
    parser.add_argument('--video', type=str, default='sample_video.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--img_width', type=int, default=1000, help='Image width for normalization')
    parser.add_argument('--img_height', type=int, default=1000, help='Image height for normalization')
    parser.add_argument('--is_normalized', action='store_true', help='Set if 2D coordinates are already normalized (0-1)')
    args = parser.parse_args()

    # Example usage of the new function:
    # Let's create some dummy 2D keypoints for demonstration purposes.
    # In a real scenario, you would load your 2D keypoints here.
    # Assuming T=240 frames, J=17 joints, 3 for (x, y, confidence)
    dummy_2d_poses = np.random.rand(240, 17, 3) * 1000 # Example pixel coordinates for 1000x1000 image
    
    # If your 2D poses are already normalized (0-1), set is_normalized=True
    # dummy_2d_poses_normalized = np.random.rand(240, 17, 3) # Example normalized coordinates

    print("--- Starting 2D to 3D Pose Conversion ---")
    predicted_3d_poses = process_2d_to_3d_pose(
        keypoints_2d=dummy_2d_poses,
        img_width=args.img_width,
        img_height=args.img_height,
        is_normalized=args.is_normalized,
        gpu_id=args.gpu
    )
    print(f"\nShape of the output 3D poses: {predicted_3d_poses.shape}")
    print("--- 2D to 3D Pose Conversion Complete ---")

    # The original demo functionality remains for visual output if you run it with a video
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    video_path = './demo/video/' + args.video
    video_name = video_path.split('/')[-1].split('.')[0]
    output_dir = './demo/output/' + video_name + '/'

    # To run the original demo video processing and visualization:
    # get_pose2D(video_path, output_dir)
    # get_pose3D(video_path, output_dir)
    # img2video(video_path, output_dir)
    # print('Generating demo successful!')