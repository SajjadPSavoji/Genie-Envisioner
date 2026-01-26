import json
from pathlib import Path

def load_annotation(annotation_path: str):
    """
    Load annotation from 2025-challenge-demos.
    
    Args:
        annotation_path: Path to annotation JSON file
                        (e.g., 2025-challenge-demos/annotations/task-0000/episode_00000010.json)
    
    Returns:
        dict with keys: task_name, skill_annotation, primitive_annotation, meta_data
    """
    with open(annotation_path, 'r') as f:
        annotation = json.load(f)
    return annotation


def get_prompt_from_annotation(annotation: dict) -> str:
    """
    Generate a prompt from the annotation for GE-BASE.
    
    Args:
        annotation: Annotation dict from load_annotation()
    
    Returns:
        Prompt string combining task name and skill descriptions
    """
    task_name = annotation.get('task_name', '')
    skills = annotation.get('skill_annotation', [])
    
    skill_descriptions = []
    for skill in skills:
        desc_list = skill.get('skill_description', [])
        if desc_list:
            skill_descriptions.append(desc_list[0])
    
    if skill_descriptions:
        prompt = f"{task_name}: {', '.join(skill_descriptions)}"
    else:
        prompt = task_name
    
    return prompt


import subprocess
import glob

def extract_all_frames_behavior(video_path: str):
    """
    Extract all frames from a BEHAVIOR-1K video using ffmpeg.
    
    Args:
        video_path: Path to video file 
                   (e.g., 2025-challenge-demos/videos/task-0000/observation.images.rgb.head/episode_00000010.mp4)
        
    Returns:
        tuple: (frames_dir, total_frames, fps)
            - frames_dir: Path where frames are saved
            - total_frames: Number of frames extracted
            - fps: Video FPS
    """
    video_path = Path(video_path)
    
    # Extract components from path
    # Path structure: .../videos/task-XXXX/camera_name/episode_XXXXXXXX.mp4
    episode_name = video_path.stem  # e.g., episode_00000010
    camera_name = video_path.parent.name  # e.g., observation.images.rgb.head
    task_name = video_path.parents[1].name  # e.g., task-0000
    
    # Create output directory: 2025-challenge-demos/temp_frames/task-0000/episode_00000010/camera_name/
    data_root = video_path.parents[3]  # Go up to videos folder's parent
    frames_dir = data_root / "temp_frames" / task_name / episode_name / camera_name
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
                break
    
    print(f"Extracting frames from {camera_name} at {fps} fps...")
    
    # Extract frames using ffmpeg
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

def get_annotation_path(data_root: str, task_id: int, episode_id: int) -> str:
    """
    Build annotation path from task_id and episode_id.
    
    Args:
        data_root: Root directory (e.g., "/shared_work/physical_intelligence/2025-challenge-demos")
        task_id: Task ID (e.g., 0)
        episode_id: Episode ID (e.g., 10)
    
    Returns:
        Path to annotation JSON file
    """
    data_root = Path(data_root)
    task_name = f"task-{task_id:04d}"
    episode_name = f"episode_{task_id:04d}{episode_id:04d}"
    annotation_path = data_root / "annotations" / task_name / f"{episode_name}.json"
    return str(annotation_path)


def get_video_path(data_root: str, task_id: int, episode_id: int, camera: str) -> str:
    """
    Build video path from task_id, episode_id, and camera name.
    
    Args:
        data_root: Root directory (e.g., "/shared_work/physical_intelligence/2025-challenge-demos")
        task_id: Task ID (e.g., 0)
        episode_id: Episode ID (e.g., 10)
        camera: Camera name, one of:
                - "observation.images.rgb.head"
                - "observation.images.rgb.left_wrist"
                - "observation.images.rgb.right_wrist"
    
    Returns:
        Path to video file
    """
    data_root = Path(data_root)
    task_name = f"task-{task_id:04d}"
    episode_name = f"episode_{task_id:04d}{episode_id:04d}"
    video_path = data_root / "videos" / task_name / camera / f"{episode_name}.mp4"
    return str(video_path)

import shutil

def create_sample_folder_behavior(
    data_root: str,
    task_id: int,
    episode_id: int,
    output_dir: str,
    target_frame: int,
    n_frames: int = 4,
    frame_spacing: int = 10,
):
    """
    Create a sample folder with memory frames for GE-BASE testing.
    
    Args:
        data_root: Root directory (e.g., "/shared_work/physical_intelligence/2025-challenge-demos")
        task_id: Task ID (e.g., 0)
        episode_id: Episode ID (e.g., 10)
        output_dir: Path to output sample folder (e.g., "video_gen_examples/sample_b1k_task0_ep10")
        target_frame: The frame to predict from (e.g., 265 for end of first skill)
        n_frames: Number of memory frames to extract (default: 4)
        frame_spacing: Frame interval between memory frames (default: 10)
    """
    data_root = Path(data_root)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Get annotation and generate prompt
    annotation_path = get_annotation_path(str(data_root), task_id, episode_id)
    annotation = load_annotation(annotation_path)
    prompt = get_prompt_from_annotation(annotation)
    
    # Camera mapping: BEHAVIOR-1K -> GE-BASE naming
    camera_mapping = {
        "observation.images.rgb.head": "head_color",
        "observation.images.rgb.left_wrist": "hand_left_color",
        "observation.images.rgb.right_wrist": "hand_right_color"
    }
    
    # Build paths to extracted frames
    task_name = f"task-{task_id:04d}"
    episode_name = f"episode_{task_id:04d}{episode_id:04d}"
    
    # Select frames working backwards from target_frame
    # E.g., target=265, spacing=10, n_frames=4 -> [235, 245, 255, 265]
    selected_frames = []
    for i in range(n_frames-1, -1, -1):
        frame_idx = target_frame - i * frame_spacing
        frame_idx = max(1, frame_idx)  # ffmpeg frames start at 1
        selected_frames.append(frame_idx)
    
    print(f"Target frame: {target_frame}")
    print(f"Selected memory frames: {selected_frames}")
    print(f"Prompt: {prompt}")
    
    # Copy frames for each camera
    for b1k_camera, ge_camera in camera_mapping.items():
        # Source: temp_frames/task-0000/episode_00000010/observation.images.rgb.head/
        frames_source = data_root / "temp_frames" / task_name / episode_name / b1k_camera
        
        # Destination: output_dir/head_color/
        cam_output = output_path / ge_camera
        cam_output.mkdir(parents=True, exist_ok=True)
        
        for idx, frame_num in enumerate(selected_frames):
            src = frames_source / f"{frame_num:06d}.png"
            dst = cam_output / f"{idx}.png"
            
            if src.exists():
                shutil.copy(src, dst)
                print(f"Copied {b1k_camera}/{src.name} -> {ge_camera}/{dst.name}")
            else:
                print(f"Warning: {src} does not exist!")
    
    # Write prompt.txt
    prompt_file = output_path / "prompt.txt"
    with open(prompt_file, 'w') as f:
        f.write(prompt)
    print(f"Wrote prompt to {prompt_file}")
    
    print(f"\n✓ Sample folder created at: {output_path}")

def create_ground_truth_video(
    data_root: str,
    task_id: int,
    episode_id: int,
    output_dir: str,
    start_frame: int,
    n_frames: int = 61,  # Default for n_chunk=1 (slow model)
    fps: int = 30
):
    """
    Create ground truth video with 3 camera views side-by-side for comparison with GE-BASE output.
    
    Args:
        data_root: Root directory (e.g., "/shared_work/physical_intelligence/2025-challenge-demos")
        task_id: Task ID (e.g., 0)
        episode_id: Episode ID (e.g., 10)
        output_dir: Path to output directory (same as sample folder)
        start_frame: First frame to include in ground truth (typically target_frame + 1)
        n_frames: Number of frames to include in ground truth video 
                  - 61 for n_chunk=1 (slow model, ~2 seconds)
                  - 122 for n_chunk=2 (~4 seconds)
                  - 244 for n_chunk=4 (~8 seconds)
        fps: Frame rate for output video (default: 30)
    """
    data_root = Path(data_root)
    output_path = Path(output_dir)
    
    # Build paths
    task_name = f"task-{task_id:04d}"
    episode_name = f"episode_{task_id:04d}{episode_id:04d}"
    
    # Camera order: head, left_wrist, right_wrist
    cameras = [
        "observation.images.rgb.head",
        "observation.images.rgb.left_wrist",
        "observation.images.rgb.right_wrist"
    ]
    
    print(f"Creating ground truth video from frames {start_frame} to {start_frame + n_frames - 1}")
    
    # Create temporary directories for each camera
    temp_dirs = []
    for cam in cameras:
        frames_source = data_root / "temp_frames" / task_name / episode_name / cam
        temp_dir = output_path / f"temp_gt_{cam.split('.')[-1]}"
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_dirs.append(temp_dir)
        
        # Copy and rename frames sequentially
        for i in range(n_frames):
            src_frame_num = start_frame + i
            src = frames_source / f"{src_frame_num:06d}.png"
            dst = temp_dir / f"{i+1:06d}.png"
            
            if src.exists():
                shutil.copy(src, dst)
            else:
                print(f"Warning: {src} does not exist!")
    
    # Create side-by-side video using ffmpeg
    # Resize all to same height (480) before stacking
    output_video = output_path / "ground_truth.mp4"
    
    # FFmpeg filter: scale head to 480 height, then stack all horizontally
    ffmpeg_cmd = [
        'ffmpeg',
        '-framerate', str(fps),
        '-i', str(temp_dirs[0] / '%06d.png'),  # head
        '-framerate', str(fps),
        '-i', str(temp_dirs[1] / '%06d.png'),  # left_wrist
        '-framerate', str(fps),
        '-i', str(temp_dirs[2] / '%06d.png'),  # right_wrist
        '-filter_complex', '[0:v]scale=-1:480[v0];[v0][1:v][2:v]hstack=inputs=3',  # Scale head, then stack
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        '-y',  # Overwrite
        str(output_video)
    ]
    
    result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"FFmpeg error: {result.stderr}")
        raise subprocess.CalledProcessError(result.returncode, ffmpeg_cmd, result.stdout, result.stderr)
    
    print(f"✓ Created {output_video}")
    
    # Clean up temp directories
    for temp_dir in temp_dirs:
        shutil.rmtree(temp_dir)
    
    print(f"\n✓ Ground truth video created: {output_video}")