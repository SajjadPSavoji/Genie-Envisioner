import json
from pathlib import Path
import subprocess
import glob
import cv2
import shutil
import numpy as np


def load_action_config(task_info_path: str, episode_id: int):
    """
    Load action configuration for a specific episode from task info JSON.
    
    Args:
        task_info_path: Path to task_XXX.json file
        episode_id: Episode ID to retrieve
        
    Returns:
        Dictionary containing episode info with action_config
    """
    with open(task_info_path, 'r') as f:
        task_info = json.load(f)
    
    # Find the episode
    for episode in task_info:
        if episode['episode_id'] == episode_id:
            return episode
    
    raise ValueError(f"Episode {episode_id} not found in {task_info_path}")

def extract_all_frames(video_path: str):
    """
    Extract all frames from a video using ffmpeg (more reliable for AV1 codec).
    Automatically determines output directory and camera name from video path.
    
    Args:
        video_path: Path to the video file 
                   (e.g., agibot_world_sample/351/observations/351/794073/videos/head_color.mp4)
        
    Returns:
        tuple: (frames_dir, total_frames, fps)
            - frames_dir: Path where frames are saved
            - total_frames: Number of frames extracted
            - fps: Video FPS
    """
    video_path = Path(video_path)
    camera_name = video_path.stem
    task_root = video_path.parents[4]
    parts = video_path.parts
    episode_id = parts[-3]
    frames_dir = task_root / episode_id / camera_name
    frames_dir.mkdir(parents=True, exist_ok=True)
    
    # Use system ffmpeg/ffprobe
    ffprobe_bin = 'ffprobe'
    ffmpeg_bin = 'ffmpeg'
    
    # Get video info using ffprobe
    fps_cmd = [
        ffprobe_bin, '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=r_frame_rate,nb_frames',
        '-of', 'default=noprint_wrappers=1', str(video_path)
    ]
    result = subprocess.run(fps_cmd, capture_output=True, text=True)

    # Check if ffprobe succeeded
    if result.returncode != 0:
        print(f"Warning: ffprobe failed, using default fps=30.0")
        print(f"Error: {result.stderr}")
        fps = 30.0
    else:
        # Parse FPS
        fps = 30.0  # Default
        for line in result.stdout.split('\n'):
            if 'r_frame_rate=' in line:
                fps_str = line.split('=')[1]
                num, den = map(int, fps_str.split('/'))
                fps = num / den
                print(f"FFPROBE EXTRACTED FPS: {fps}")
                break
    
    print(f"Extracting frames from {camera_name} at {fps} fps...")
    
    # Extract frames using ffmpeg (suppresses AV1 warnings)
    ffmpeg_cmd = [
        ffmpeg_bin, '-i', str(video_path),
        '-vsync', '0',  # Don't duplicate frames
        '-frame_pts', '1',  # Use presentation timestamp
        str(frames_dir / '%06d.png'),
        '-loglevel', 'error',  # Suppress warnings
        '-y'  # Overwrite existing files
    ]
    
    subprocess.run(ffmpeg_cmd, check=True)
    
    # Count extracted frames
    frame_files = sorted(glob.glob(str(frames_dir / '*.png')))
    total_frames = len(frame_files)
    
    print(f"✓ Extracted {total_frames} frames to {frames_dir}")
    
    return str(frames_dir), total_frames, fps

def create_sample_folder(
    episode_dir: str,
    output_dir: str,
    target_frame: int,
    n_frames: int = 4,
    frame_spacing: int = 10,
    prompt: str = ""
):
    """
    Create a sample folder with memory frames leading up to a target frame.
    
    Args:
        episode_dir: Path to episode folder (e.g., /path/to/794073)
        output_dir: Path to output sample folder (e.g., sample_1)
        target_frame: The frame to predict from (e.g., 184 for action transition)
        n_frames: Number of memory frames to extract (default: 4)
        frame_spacing: Frame interval between memory frames (default: 10)
        prompt: Task description text
    """
    episode_path = Path(episode_dir)
    output_path = Path(output_dir)
    
    # Create output directory
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get camera folders
    camera_folders = ['head_color', 'hand_left_color', 'hand_right_color']
    
    # Select frames working backwards from target_frame
    # E.g., target=184, spacing=10, n_frames=4 -> [154, 164, 174, 184]
    selected_frames = []
    for i in range(n_frames-1, -1, -1):
        frame_idx = target_frame - i * frame_spacing
        frame_idx = max(0, frame_idx)  # Don't go below 0
        selected_frames.append(frame_idx)
    
    print(f"Target frame: {target_frame}")
    print(f"Selected memory frames: {selected_frames}")
    
    # Copy frames for each camera
    for cam in camera_folders:
        cam_input = episode_path / cam
        cam_output = output_path / cam
        cam_output.mkdir(parents=True, exist_ok=True)
        
        for idx, frame_num in enumerate(selected_frames):
            src = cam_input / f"{frame_num:06d}.png"
            dst = cam_output / f"{idx}.png"
            
            if src.exists():
                shutil.copy(src, dst)
                print(f"Copied {src.name} -> {dst}")
            else:
                print(f"Warning: {src} does not exist!")
    
    # Write prompt.txt
    prompt_file = output_path / "prompt.txt"
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    print(f"Wrote prompt to {prompt_file}")
    
    print(f"\nSample folder created at: {output_path}")

def create_multiview_video(
    episode_dir: str,
    output_path: str,
    start_frame: int,
    n_frames: int = 4,
    frame_spacing: int = 10,
    future_frames: int = 60,
    fps: int = 30
):
    """
    Create a multi-view video showing head, left hand, right hand cameras side by side.
    
    Args:
        episode_dir: Path to episode folder (e.g., /path/to/794073)
        output_path: Output video file path (e.g., output.mp4)
        start_frame: Target frame index (e.g., 184)
        n_frames: Number of memory frames (default: 4)
        frame_spacing: Frame interval between memory frames (default: 10)
        future_frames: How many frames forward to include (default: 60)
        fps: Output video framerate (default: 30)
    """
    episode_path = Path(episode_dir)
    
    # Calculate frame range
    # start_frame - (n_frames * frame_spacing) to start_frame + future_frames
    first_frame = start_frame - (n_frames * frame_spacing)
    last_frame = start_frame + future_frames
    first_frame = max(0, first_frame)  # Don't go below 0
    
    print(f"Creating video from frames {first_frame} to {last_frame}")
    print(f"Total frames: {last_frame - first_frame + 1}")
    
    # Camera folders in order: head, left, right
    camera_folders = ['head_color', 'hand_left_color', 'hand_right_color']
    
    # Read first frame to get dimensions
    first_img_path = episode_path / camera_folders[0] / f"{first_frame:06d}.png"
    if not first_img_path.exists():
        print(f"Error: {first_img_path} does not exist!")
        return
    
    sample_img = cv2.imread(str(first_img_path))
    h, w = sample_img.shape[:2]
    
    # Output video will be 3 views side by side
    out_width = w * 3
    out_height = h
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_width, out_height))
    
    print(f"Output resolution: {out_width}x{out_height}")
    
    # Process each frame
    for frame_idx in range(first_frame, last_frame + 1):
        frames = []
        
        for cam in camera_folders:
            img_path = episode_path / cam / f"{frame_idx:06d}.png"
            
            if img_path.exists():
                img = cv2.imread(str(img_path))
                frames.append(img)
            else:
                # If frame doesn't exist, use black frame
                print(f"Warning: {img_path} not found, using black frame")
                frames.append(np.zeros((h, w, 3), dtype=np.uint8))
        
        # Concatenate horizontally: [head | left | right]
        combined = np.hstack(frames)
        out.write(combined)
    
    out.release()
    print(f"\nVideo saved to: {output_path}")