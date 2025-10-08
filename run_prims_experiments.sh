#!/bin/bash
#SBATCH --job-name=video_lmms_eval
#SBATCH --partition=a100_short
#SBATCH --gres=gpu:a100:1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --time=3-00:00:00
#SBATCH --output=results/output_%j.txt  # Save output in a file named with job ID
#SBATCH --error=results/output_%j.txt    # Save error logs

# # Window of 1 sec
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=1,sampling_strategy=dense,sampling_fps=1,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=2,sampling_strategy=dense,sampling_fps=2,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=4,sampling_strategy=dense,sampling_fps=4,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=8,sampling_strategy=dense,sampling_fps=8,overlap_frames_num=0"

# # Window of 2 seconds
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=2,sampling_strategy=dense,sampling_fps=1,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=4,sampling_strategy=dense,sampling_fps=2,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=8,sampling_strategy=dense,sampling_fps=4,overlap_frames_num=0"

# # Window of 0.5 seconds
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=1,sampling_strategy=dense,sampling_fps=2,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=2,sampling_strategy=dense,sampling_fps=4,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=4,sampling_strategy=dense,sampling_fps=8,overlap_frames_num=0"

# # Heavy, window of 1 second
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=15,sampling_strategy=dense,sampling_fps=15,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=30,sampling_strategy=dense,sampling_fps=30,overlap_frames_num=0"

# Window of 0.25 seconds and 0.125 seconds
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=1,sampling_strategy=dense,sampling_fps=4,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=2,sampling_strategy=dense,sampling_fps=8,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=1,sampling_strategy=dense,sampling_fps=8,overlap_frames_num=0"



# # SMC
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_3 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=1,sampling_strategy=dense,sampling_fps=4,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_3 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=2,sampling_strategy=dense,sampling_fps=8,overlap_frames_num=0"

# # Ideal
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_1 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=1,sampling_strategy=dense,sampling_fps=4,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_1 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=2,sampling_strategy=dense,sampling_fps=8,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_1 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=1,sampling_strategy=dense,sampling_fps=2,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_1 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=2,sampling_strategy=dense,sampling_fps=4,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_1 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=1,sampling_strategy=dense,sampling_fps=1,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_1 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=2,sampling_strategy=dense,sampling_fps=2,overlap_frames_num=0"

sb evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_1 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=8,sampling_strategy=dense,sampling_fps=8,overlap_frames_num=0"
sb evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_1 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=8,sampling_strategy=dense,sampling_fps=4,overlap_frames_num=0"
sb evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_1 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=8,sampling_strategy=dense,sampling_fps=2,overlap_frames_num=0"
sb evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_1 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=15,sampling_strategy=dense,sampling_fps=15,overlap_frames_num=0"
sb evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_1 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=15,sampling_strategy=dense,sampling_fps=8,overlap_frames_num=0"
sb evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_1 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=15,sampling_strategy=dense,sampling_fps=4,overlap_frames_num=0"

# # SP
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=4,sampling_strategy=dense,sampling_fps=15,overlap_frames_num=0"
# # Send this back
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=8,sampling_strategy=dense,sampling_fps=30,overlap_frames_num=0"



# SMC FPS 2 Frames 1
# # SMC FPS 4 Frames 2
# # SMC FPS 8 Frames 4
# # SMC FPS 15 Frames 4
# # SMC FPS 8 Frames 1
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_3 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=1,sampling_strategy=dense,sampling_fps=2,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_3 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=2,sampling_strategy=dense,sampling_fps=4,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_3 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=4,sampling_strategy=dense,sampling_fps=8,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_3 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=4,sampling_strategy=dense,sampling_fps=15,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_3 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=1,sampling_strategy=dense,sampling_fps=8,overlap_frames_num=0"

# # SP 15, 2 / SMC 15, 2
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_2 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=2,sampling_strategy=dense,sampling_fps=15,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_3 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=2,sampling_strategy=dense,sampling_fps=15,overlap_frames_num=0"


# SMC FPS 15 Frames 8, FPS 30 Frames 8?
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_3 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=8,sampling_strategy=dense,sampling_fps=15,overlap_frames_num=0"
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_3 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=8,sampling_strategy=dense,sampling_fps=30,overlap_frames_num=0"

# SP 15, 8 | SP 30, 15


# Send SMC
# Small
# # sb evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_3
# sb evaluate.sh --model llava_ov_0p5b --task strokerehab_primitives_3
# sb evaluate.sh --model llava_ov_7b --task strokerehab_primitives_3
# sb evaluate.sh --model llava_next_video_7b --task strokerehab_primitives_3
# sb evaluate.sh --model internvl3p5_1b --task strokerehab_primitives_3
# sb evaluate.sh --model internvl3p5_8b --task strokerehab_primitives_3
# sb evaluate.sh --model longva_7b --task strokerehab_primitives_3
# sb evaluate.sh --model nvila_8b --task strokerehab_primitives_3
# sb evaluate.sh --model nvila_15b --task strokerehab_primitives_3
sb evaluate.sh --model internvl3p5_2b --task strokerehab_primitives_3
sb evaluate.sh --model internvl3p5_4b --task strokerehab_primitives_3
sb evaluate.sh --model internvl3p5_14b --task strokerehab_primitives_3
# Medium
# sb evaluate.sh --model qwen2_5_vl_32b --task strokerehab_primitives_3
# sb evaluate.sh --model internvl3p5_38b --task strokerehab_primitives_3
# sb evaluate.sh --model internvl3p5_30b_a3b --task strokerehab_primitives_3

# Big (use 3 A100 GPUs)
# sb evaluate.sh --model qwen2_5_vl_72b --task strokerehab_primitives_3
# sb evaluate.sh --model llava_ov_72b --task strokerehab_primitives_3
# sb evaluate.sh --model llava_next_video_72b --task strokerehab_primitives_3
# sb evaluate.sh --model internvl3_78b --task strokerehab_primitives_3
sb evaluate.sh --model internvl3p5_241b_a28b --task strokerehab_primitives_3  # Have not sent yet


# SMC FPS 30, 15. 
# bash evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_3 --model_args "pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_frames_num=15,sampling_strategy=dense,sampling_fps=30,overlap_frames_num=0"


# Healthy patients
sb evaluate.sh --model llava_next_video_7b --task strokerehab_primitives_7
sb evaluate.sh --model qwen2_5_vl_7b --task strokerehab_primitives_7
sb evaluate.sh --model internvl3p5_1b --task strokerehab_primitives_7
sb evaluate.sh --model internvl3p5_8b --task strokerehab_primitives_7

sb evaluate2.sh --model qwen2_5_vl_32b --task strokerehab_primitives_7
sb evaluate3.sh --model qwen2_5_vl_72b --task strokerehab_primitives_7

sb evaluate2.sh --model internvl3p5_38b --task strokerehab_primitives_7
sb evaluate3.sh --model internvl3_78b --task strokerehab_primitives_7

### IA RUNS ###
# # One GPU
# sb evaluate.sh --model qwen2_5_vl_7b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# sb evaluate.sh --model llava_ov_0p5b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# sb evaluate.sh --model llava_ov_7b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# sb evaluate.sh --model llava_next_video_7b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# sb evaluate.sh --model internvl_2p5_2b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# sb evaluate.sh --model internvl_2p5_8b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# sb evaluate.sh --model longva_7b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# sb evaluate.sh --model nvila_8b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# # Two GPUs
# sb evaluate.sh --model qwen2_5_vl_32b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# sb evaluate.sh --model internvl_2p5_38b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# sb evaluate.sh --model nvila_15b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# # Three GPUs
# sb evaluate.sh --model qwen2_5_vl_72b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# sb evaluate.sh --model llava_ov_72b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# sb evaluate.sh --model llava_next_video_72b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
# sb evaluate.sh --model internvl_2p5_78b --task strokerehab_ia3_3_30,strokerehab_ia3_31_33,strokerehab_ia4_3_30,strokerehab_ia4_31_33
