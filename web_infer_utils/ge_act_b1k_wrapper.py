#!/usr/bin/env python3
"""
GE-ACT BEHAVIOR-1K Wrapper
==========================

This file adapts BEHAVIOR-1K observations to GE-ACT format for inference.

WHAT THIS FILE DOES:
1. Receives observations from BEHAVIOR-1K (camera images + proprioception)
2. Converts them to the format GE-ACT expects
3. Calls the GE-ACT model for action prediction
4. Converts the action back to BEHAVIOR-1K's format

ARCHITECTURE:
    BEHAVIOR-1K → WebsocketPolicyServer → GEActB1KWrapper.act() → MVActor.play() → action
                                              ↓
                                   _process_observation()  → images (3, H, W, 3)
                                              ↓
                                   _pad_action_to_b1k()    → action (1, 23)

REFERENCE:
    Based on OpenPI's BEHAVIOR-1K integration pattern:
    b1k-baselines/baselines/openpi/src/openpi/shared/eval_b1k_wrapper.py

    Key differences from OpenPI:
    - OpenPI uses B1KPolicyWrapper.process_obs() → we use _process_observation()
    - OpenPI resizes to 224x224 → we use 256x256 (Calvin model training size)
      NOTE: Once we have a GE-ACT model trained on BEHAVIOR-1K, switch to 224x224
            to match OpenPI and the B1K dataset's native resolution.
    - OpenPI has temporal ensemble logic → we use simple per-step inference
      * Temporal ensemble: Predict multiple future actions, blend predictions over time (smoother)
      * Per-step: Predict action, execute immediately, repeat (simpler, good for testing)
      TODO: Implement temporal ensemble for better performance (see eval_b1k_wrapper.py:67-128)

LIMITATIONS:
    - Currently using Calvin-trained model (action_dim=22) with B1K data (action_dim=23)
    - Proprioception: B1K provides 256-dim raw state, we extract 23-dim joint state,
      then drop the last dimension (right gripper) to fit Calvin's 22-dim expectation.
      TODO: Train GE-ACT model on B1K data with action_dim=23 for proper compatibility.
"""

import logging
import numpy as np
import torch

from typing import Optional, Tuple, Dict, Any

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# These are the exact key names BEHAVIOR-1K uses for camera observations
# Found in: BEHAVIOR-1K/OmniGibson/omnigibson/learning/utils/eval_utils.py
B1K_CAMERA_NAMES = {
    "head": "robot_r1::robot_r1:zed_link:Camera:0::rgb",
    "left_wrist": "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb",
    "right_wrist": "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb",
}

# BEHAVIOR-1K R1Pro robot action dimension (23 joints total)
# Breakdown: 3 base + 4 torso + 7 left arm + 1 left gripper + 7 right arm + 1 right gripper
B1K_ACTION_DIM = 23

# Calvin model dimension breakdown (from stats_calvin_rel.json):
#   - calvin_eef: 7 dims (action: 3 pos + 3 rot + 1 gripper)
#   - calvin_state_eef: 15 dims (proprioceptive state from robot)
#   - Total: 7 action + 15 state = 22
#
# When add_state=True (see configs/ltx_model/calvin/action_model_calvin.yaml:50),
# the model input/output includes both action and state, hence action_in_channels=22.
#
# Source: configs/ltx_model/calvin/stats_calvin_rel.json
CALVIN_ACTION_DIM = 22


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def resize_with_pad(images: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """
    Resize image(s) with padding to maintain aspect ratio.
    
    Replicates OpenPI's resize_with_pad behavior using PIL.
    Reference: b1k-baselines/baselines/openpi/packages/openpi-client/src/openpi_client/image_tools.py
    
    Args:
        images: Image(s) in [..., height, width, channel] format (supports arbitrary batch dimensions)
        target_h: Target height
        target_w: Target width
        
    Returns:
        Resized and padded image(s) in [..., target_h, target_w, channel] format
    """
    from PIL import Image
    
    # If already correct size, return as is
    if images.shape[-3:-1] == (target_h, target_w):
        return images
    
    original_shape = images.shape
    
    # Reshape to (N, H, W, C) for processing
    images_flat = images.reshape(-1, *original_shape[-3:])
    
    # Process each image
    resized_images = []
    for img in images_flat:
        # Convert numpy to PIL Image
        pil_img = Image.fromarray(img)
        cur_width, cur_height = pil_img.size
        
        # If already correct size, skip resize
        if cur_width == target_w and cur_height == target_h:
            resized_images.append(img)
            continue
        
        # Calculate resize ratio (OpenPI's method)
        ratio = max(cur_width / target_w, cur_height / target_h)
        resized_height = int(cur_height / ratio)
        resized_width = int(cur_width / ratio)
        
        # Resize image
        resized_image = pil_img.resize((resized_width, resized_height), resample=Image.BILINEAR)
        
        # Create zero-padded canvas
        zero_image = Image.new(resized_image.mode, (target_w, target_h), 0)
        
        # Calculate padding to center the image
        pad_height = max(0, int((target_h - resized_height) / 2))
        pad_width = max(0, int((target_w - resized_width) / 2))
        
        # Paste resized image onto canvas
        zero_image.paste(resized_image, (pad_width, pad_height))
        
        # Sanity check (from OpenPI)
        assert zero_image.size == (target_w, target_h), \
            f"Expected size ({target_w}, {target_h}), got {zero_image.size}"
        
        # Convert back to numpy
        resized_images.append(np.array(zero_image))
    
    # Stack and reshape back to original batch dimensions
    resized = np.stack(resized_images)
    return resized.reshape(*original_shape[:-3], *resized.shape[-3:])


def validate_image(img: np.ndarray, name: str) -> None:
    """
    Sanity check for image data.
    
    Args:
        img: Image array to validate
        name: Name for logging
        
    Raises:
        ValueError: If image is invalid
    """
    if img is None:
        raise ValueError(f"{name}: Image is None")
    if not isinstance(img, np.ndarray):
        raise ValueError(f"{name}: Expected numpy array, got {type(img)}")
    if len(img.shape) != 3:
        raise ValueError(f"{name}: Expected 3D array (H, W, C), got shape {img.shape}")
    if img.shape[2] not in [3, 4]:
        raise ValueError(f"{name}: Expected 3 or 4 channels, got {img.shape[2]}")
    if img.dtype != np.uint8:
        logger.warning(f"{name}: Expected uint8, got {img.dtype}")


def validate_action(action: np.ndarray, expected_dim: int, name: str) -> None:
    """
    Sanity check for action data.
    
    Args:
        action: Action array to validate
        expected_dim: Expected dimension
        name: Name for logging
    """
    if action is None:
        raise ValueError(f"{name}: Action is None")
    if not isinstance(action, np.ndarray):
        logger.warning(f"{name}: Converting {type(action)} to numpy array")
    if len(action.shape) == 0:
        raise ValueError(f"{name}: Action is scalar, expected array")
    
    actual_dim = action.shape[-1] if len(action.shape) > 0 else action.shape[0]
    if actual_dim != expected_dim:
        logger.warning(f"{name}: Expected dim {expected_dim}, got {actual_dim}")


# =============================================================================
# MAIN WRAPPER CLASS
# =============================================================================

class GEActB1KWrapper:
    """
    Wrapper that adapts BEHAVIOR-1K observations to GE-ACT format.
    
    This class is the bridge between two systems:
    - BEHAVIOR-1K: Sends observations as a dict with camera images + proprioception
    - GE-ACT: Expects observations as stacked numpy arrays
    
    USAGE:
        wrapper = GEActB1KWrapper(config, weights, prompt)
        action = wrapper.act(obs_dict)  # obs_dict from BEHAVIOR-1K
        # action is torch.Tensor (1, 23) ready for BEHAVIOR-1K
    """
    
    def __init__(
        self,
        config_file: str,
        weight_file: str,
        text_prompt: str,
        domain_name: str = "behavior1k",
        image_height: int = 256,  # Calvin model trained on 256x256
        image_width: int = 256,   # For B1K-trained model, use 224x224
        action_dim: int = 22,  # Calvin model uses 22 (16 action + 6 state)
        num_inference_steps: int = 5,
        device: str = "cuda:0",
        debug: bool = False,
        save_debug_images: bool = False,
        debug_image_dir: str = "/tmp/ge_act_b1k_debug",
        test_mode: bool = False,
    ):
        """
        Initialize the wrapper.
        
        This is where the heavy lifting happens - loading the GE-ACT model.
        
        Args:
            config_file: Path to GE-ACT config YAML (e.g., action_model_b1k.yaml)
            weight_file: Path to .safetensors weights (e.g., ge_act_calvin.safetensors)
            text_prompt: Task description (e.g., "Turn on the radio...")
            domain_name: For statistics lookup in config
            image_height: Target height for resizing (default 192)
            image_width: Target width for resizing (default 256)
            action_dim: Model's action dimension (22 for Calvin model)
            num_inference_steps: Diffusion denoising steps (more = slower but better)
            device: GPU device (e.g., "cuda:0", "cuda:1")
            debug: Enable debug logging
            save_debug_images: Save received images to disk for inspection
            debug_image_dir: Directory to save debug images
            test_mode: If True, return zero actions without running the model (for testing)
        """
        # =====================================================================
        # STEP 1: Store configuration
        # =====================================================================
        self.text_prompt = text_prompt
        self.image_height = image_height
        self.image_width = image_width
        self.action_dim = action_dim
        self.step_counter = 0  # Steps in current episode (reset to 0 on reset)
        self.episode_counter = 1  # Current episode number (starts at 1)
        self.last_logged_step = -1  # Track last step we logged debug images for
        self.debug = debug
        self.save_debug_images = save_debug_images
        self.test_mode = test_mode

        # Create output directory for debug images
        # Auto-generates a timestamped subdirectory, then prints the path
        # so user can copy it to BEHAVIOR-1K's log_path parameter.
        if self.save_debug_images:
            import os
            import json
            from datetime import datetime

            # Generate timestamped run directory
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            run_dir_name = f"ge_act_{timestamp}"
            self.output_dir = os.path.join(debug_image_dir, run_dir_name)
            self.images_dir = os.path.join(self.output_dir, "images")

            # Create camera-specific subdirectories
            self.images_head_dir = os.path.join(self.images_dir, "head")
            self.images_left_wrist_dir = os.path.join(self.images_dir, "left_wrist")
            self.images_right_wrist_dir = os.path.join(self.images_dir, "right_wrist")

            os.makedirs(self.images_head_dir, exist_ok=True)
            os.makedirs(self.images_left_wrist_dir, exist_ok=True)
            os.makedirs(self.images_right_wrist_dir, exist_ok=True)

            # Save run metadata
            self.run_metadata = {
                "start_time": timestamp,
                "text_prompt": text_prompt,
                "image_height": image_height,
                "image_width": image_width,
                "action_dim": action_dim,
                "num_inference_steps": num_inference_steps,
                "device": device,
                "test_mode": test_mode,
            }

            metadata_path = os.path.join(self.output_dir, "ge_act_metadata.json")
            with open(metadata_path, "w") as f:
                json.dump(self.run_metadata, f, indent=2)

            # Create/update symlink "ge_act_latest" pointing to this run
            # This allows user to always use the same path for BEHAVIOR-1K log_path
            latest_link = os.path.join(debug_image_dir, "ge_act_latest")
            if os.path.islink(latest_link):
                os.remove(latest_link)
            os.symlink(self.output_dir, latest_link)

            logger.info("=" * 80)
            logger.info("OUTPUT DIRECTORY:")
            logger.info(f"  Actual: {self.output_dir}")
            logger.info(f"  Symlink: {latest_link}")
            logger.info("")
            logger.info("For BEHAVIOR-1K, use:")
            logger.info(f"  log_path={latest_link}")
            logger.info("=" * 80)
        
        logger.info("=" * 60)
        logger.info("Initializing GEActB1KWrapper")
        logger.info("=" * 60)
        logger.info(f"  Config file: {config_file}")
        logger.info(f"  Weight file: {weight_file}")
        logger.info(f"  Text prompt: {text_prompt}")
        logger.info(f"  Image size: {image_height} x {image_width}")
        logger.info(f"  Action dim: {action_dim}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Debug mode: {debug}")
        logger.info(f"  Save debug images: {save_debug_images}")
        logger.info(f"  Test mode (zero actions): {test_mode}")
        
        # =====================================================================
        # STEP 2: Load the GE-ACT model (MVActor)
        # =====================================================================
        # MVActor = Multi-View Actor: handles policy inference from multiple camera views
        # Loads text encoder, VAE, diffusion transformer, and scheduler for action prediction
        # Import here to avoid circular imports at module load time
        from web_infer_utils.MVActor import MVActor
        
        logger.info("Loading MVActor model...")
        try:
            self.actor = MVActor(
                config_file=config_file,
                transformer_file=weight_file,
                domain_name=domain_name,
                action_dim=action_dim,
                gripper_dim=1,
                load_weights=True,
                num_inference_steps=num_inference_steps,
                device=torch.device(device),
            )
            logger.info("✓ GE-ACT model loaded successfully")
            logger.info("=" * 60)
        except Exception as e:
            logger.error("=" * 60)
            logger.error("Failed to load MVActor model!")
            logger.error(f"Error: {e}")
            logger.error("=" * 60)
            raise
    
    def _save_debug_image(self, img: np.ndarray, name: str) -> None:
        """
        Save image to disk for debugging.

        Images are saved every 50 steps to camera-specific subdirectories:
        - images/head/
        - images/left_wrist/
        - images/right_wrist/

        Files are named: ep{episode}_step{step}_total{total}_{type}.png
        where type is "raw" or "resized"

        Args:
            img: Image array (H, W, 3) uint8
            name: Filename suffix (e.g., "head_raw", "left_wrist_resized")
        """
        # Only save every 50 steps to reduce disk usage
        if self.step_counter % 50 != 0:
            return

        from PIL import Image
        import os

        # Determine which camera subdirectory to use based on name
        if "head" in name:
            camera_dir = self.images_head_dir
            image_type = name.replace("head_", "")  # "raw" or "resized"
        elif "left_wrist" in name:
            camera_dir = self.images_left_wrist_dir
            image_type = name.replace("left_wrist_", "")
        elif "right_wrist" in name:
            camera_dir = self.images_right_wrist_dir
            image_type = name.replace("right_wrist_", "")
        else:
            # Fallback to main images directory if camera not identified
            camera_dir = self.images_dir
            image_type = name

        # Filename format: ep001_step0050_total00050_raw.png
        filename = f"ep{self.episode_counter:03d}_step{self.step_counter:04d}_{image_type}.png"
        filepath = os.path.join(camera_dir, filename)
        Image.fromarray(img).save(filepath)

        # Log only once per step (not per camera) and only every 100 steps
        if self.step_counter % 100 == 0 and self.last_logged_step != self.step_counter:
            logger.info(f"Saved debug images for step {self.step_counter}")
            self.last_logged_step = self.step_counter
    
    def _process_observation(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert BEHAVIOR-1K observation dict to GE-ACT format.

        INPUT FORMAT (from BEHAVIOR-1K):
            obs = {
                "robot_r1::robot_r1:zed_link:Camera:0::rgb": Tensor (720, 720, 3),
                "robot_r1::robot_r1:left_realsense_link:Camera:0::rgb": Tensor (480, 480, 3),
                "robot_r1::robot_r1:right_realsense_link:Camera:0::rgb": Tensor (480, 480, 3),
                "robot_r1::proprio": Tensor (256,),
                ... other keys ...
            }

        OUTPUT FORMAT (for MVActor.play()):
            images: np.ndarray (3, H, W, 3) - uint8 [0-255], 3 camera views
                    where H=image_height (default 256), W=image_width (default 256)
            state: np.ndarray (action_dim,) - float32, proprioception (default 22 dims)

        Args:
            obs: Dictionary from BEHAVIOR-1K

        Returns:
            Tuple of (images, state)
        """
        if self.debug:
            logger.info(f"[Step {self.step_counter}] Processing observation...")
            logger.info(f"  Observation keys: {list(obs.keys())[:5]}...")  # First 5 keys
        
        # =====================================================================
        # STEP 1: Extract and process camera images
        # =====================================================================
        images = []
        for camera_id, camera_key in enumerate(["head", "left_wrist", "right_wrist"]):
            obs_key = B1K_CAMERA_NAMES[camera_key]
            
            if obs_key in obs:
                img = obs[obs_key]
                
                # Convert torch tensor to numpy if needed
                if isinstance(img, torch.Tensor):
                    img = img.cpu().numpy()
                
                # Validate the image
                validate_image(img, f"Camera_{camera_key}")
                
                # Ensure uint8 (0-255 range)
                if img.dtype != np.uint8:
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)
                
                # Remove alpha channel if present (RGBA → RGB)
                if img.shape[-1] == 4:
                    img = img[..., :3]
                
                if self.debug:
                    logger.info(f"  {camera_key}: shape={img.shape}, dtype={img.dtype}, "
                              f"min={img.min()}, max={img.max()}")
            else:
                # Camera not found - use placeholder
                logger.warning(f"Camera {obs_key} not found, using black placeholder")
                img = np.zeros((self.image_height, self.image_width, 3), dtype=np.uint8)
            
            # Save raw image before resize (for debugging)
            if self.save_debug_images:
                self._save_debug_image(img, f"{camera_key}_raw")

            # Resize to target resolution
            img = resize_with_pad(img, self.image_height, self.image_width)

            # Save resized image (for debugging)
            if self.save_debug_images:
                self._save_debug_image(img, f"{camera_key}_resized")
            
            images.append(img)
        
        # Stack images: (3, H, W, 3) - 3 cameras, H height, W width, 3 RGB channels
        images = np.stack(images, axis=0)
        
        if self.debug:
            logger.info(f"  Stacked images shape: {images.shape}")
        
        # =====================================================================
        # STEP 2: Extract proprioception state
        # =====================================================================
        proprio_key = "robot_r1::proprio"
        if proprio_key not in obs:
            raise KeyError(f"Required key '{proprio_key}' not found in observation. "
                          f"Available keys: {list(obs.keys())}")

        state = obs[proprio_key]
        if isinstance(state, torch.Tensor):
            state = state.cpu().numpy()
        state = state.flatten()

        if self.debug:
            logger.info(f"  Raw proprio: shape={state.shape}, min={state.min():.3f}, max={state.max():.3f}")

        # Extract relevant state from B1K's 256-dim proprioception
        # Following omnigibson.learning.utils.eval_utils.PROPRIOCEPTION_INDICES["R1Pro"]
        # and openpi.policies.b1k_policy.extract_state_from_proprio
        if len(state) == 256:
            # Extract 23-dim state from 256-dim B1K proprioception:
            # base_qvel (3) + trunk_qpos (4) + arm_left_qpos (7) + arm_right_qpos (7)
            # + left_gripper_width (1) + right_gripper_width (1) = 23
            base_qvel = state[253:256]  # 3
            trunk_qpos = state[236:240]  # 4
            arm_left_qpos = state[158:165]  # 7
            arm_right_qpos = state[197:204]  # 7
            # Sum the two gripper finger positions to get gripper width
            left_gripper_width = np.array([state[193:195].sum()])  # 2 -> 1
            right_gripper_width = np.array([state[232:234].sum()])  # 2 -> 1

            state = np.concatenate([
                base_qvel,
                trunk_qpos,
                arm_left_qpos,
                arm_right_qpos,
                left_gripper_width,
                right_gripper_width,
            ])  # 23 dims

            if self.debug:
                logger.info(f"  Extracted 23-dim state from 256-dim proprio")

            # TODO: Match with B1K action_dim=23 in the future
            # For now, drop the last dimension (right gripper) to fit Calvin model's 22 dims
            if self.action_dim == 22:
                state = state[:22]  # Drop last dim (right gripper width)
                if self.debug:
                    logger.info(f"  Truncated to 22 dims to match Calvin model (dropped right gripper)")

        # Validate state dimension
        if len(state) != self.action_dim:
            raise ValueError(
                f"Proprioception dimension mismatch! "
                f"Expected {self.action_dim} dims, got {len(state)} dims. "
                f"Raw proprio shape was {obs[proprio_key].flatten().shape}. "
                f"Check that the model's action_dim is compatible with the input data."
            )

        if self.debug:
            logger.info(f"  Final state: shape={state.shape}, min={state.min():.3f}, max={state.max():.3f}")

        return images, state.astype(np.float32)
    
    def _pad_action_to_b1k(self, action: np.ndarray) -> torch.Tensor:
        """
        Convert GE-ACT action to BEHAVIOR-1K's 23-dim action space.
        
        INPUT (from GE-ACT):
            action: np.ndarray (22,) - 16 action dims + 6 state dims (Calvin model)
        
        OUTPUT (for BEHAVIOR-1K):
            action: torch.Tensor (1, 23) - padded to match R1Pro robot
        
        NOTE: This is a simple zero-padding for infrastructure testing.
        For proper task performance, you'd need to map action spaces properly.
        """
        if self.debug:
            logger.info(f"  Raw action from GE-ACT: shape={action.shape}")
        
        # Validate input
        if action is None:
            raise ValueError("Action from GE-ACT is None!")
        
        # =====================================================================
        # Calvin model outputs 22-dimensional action
        # For EEF space: 7 action (3 pos + 3 rot + 1 gripper) + 15 state
        # For dual-arm: (10 arm + 1 gripper) × 2 arms = 22
        # Either way, the full 22-dim is the action prediction to use
        # =====================================================================
        action_to_use = action[:CALVIN_ACTION_DIM] if len(action) >= CALVIN_ACTION_DIM else action
        
        if self.debug:
            logger.info(f"  Using action dims (first {CALVIN_ACTION_DIM}): {action_to_use[:4]}... (showing first 4)")
        
        # =====================================================================
        # Pad to BEHAVIOR-1K's 23 dimensions
        # B1K R1Pro: 3 base + 4 torso + 7 left arm + 1 gripper + 7 right arm + 1 gripper = 23
        # =====================================================================
        padded = np.zeros(B1K_ACTION_DIM, dtype=np.float32)
        padded[:len(action_to_use)] = action_to_use
        
        if self.debug:
            logger.info(f"  Padded action: shape=({B1K_ACTION_DIM},), non-zero={np.count_nonzero(padded)}")
        
        # Convert to torch tensor with batch dimension: (1, 23)
        return torch.from_numpy(padded).unsqueeze(0).float()
    
    def _get_zero_action(self) -> torch.Tensor:
        """
        Return a zero action for testing infrastructure without running the model.
        
        This is useful for:
        - Testing client-server communication
        - Verifying observation processing
        - Debugging without waiting for model inference
        
        Returns:
            torch.Tensor: Zero action of shape (1, 23) matching B1K's action space
        """
        if self.debug:
            logger.info("  Returning zero action (test mode)")
        actions = torch.zeros(1, B1K_ACTION_DIM, dtype=torch.float32)
        actions[0, :3] = 1.0
        return actions
    
    def act(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Main entry point: Compute action from BEHAVIOR-1K observation.
        
        This method is called by WebsocketPolicyServer for each simulation step.
        
        Args:
            obs: BEHAVIOR-1K observation dictionary
            
        Returns:
            torch.Tensor: Action tensor of shape (1, 23)
        """
        self.step_counter += 1

        if self.debug:
            logger.info("=" * 40)
            logger.info(f"GEActB1KWrapper.act() - Episode {self.episode_counter}, Step {self.step_counter}")
            logger.info("=" * 40)
        else:
            # Always log progress every 100 steps
            if self.step_counter % 100 == 0:
                logger.info(f"Progress: Episode {self.episode_counter}, Step {self.step_counter}")
        
        # =====================================================================
        # STEP 1: Process observation (B1K format → GE-ACT format)
        # =====================================================================
        images, state = self._process_observation(obs)
        
        # Sanity check
        assert images.shape == (3, self.image_height, self.image_width, 3), \
            f"Expected images shape (3, {self.image_height}, {self.image_width}, 3), got {images.shape}"
        assert state.shape == (self.action_dim,), \
            f"Expected state shape ({self.action_dim},), got {state.shape}"
        
        # =====================================================================
        # TEST MODE: Return zero actions without running the model
        # =====================================================================
        if self.test_mode:
            if self.debug:
                logger.info("  TEST MODE: Skipping model inference, returning zero action")
            return self._get_zero_action()
        
        # =====================================================================
        # STEP 2: Call GE-ACT model for action prediction
        # =====================================================================
        if self.debug:
            logger.info("  Calling MVActor.play()...")
        
        action = self.actor.play(
            obs=images,              # (3, H, W, 3) uint8
            prompt=self.text_prompt, # Task description string
            state=state,             # (22,) float32
            execution_step=1,        # For temporal consistency
        )
        
        if self.debug:
            logger.info(f"  MVActor returned action: shape={action.shape}")
        
        # =====================================================================
        # STEP 3: Handle action chunks (model may return multiple timesteps)
        # =====================================================================
        # GE-ACT returns action chunks (multiple future actions)
        # We only use the first action for immediate execution
        if len(action.shape) > 1:
            action = action[0]  # Take first action from chunk
        
        # =====================================================================
        # STEP 4: Convert to BEHAVIOR-1K format
        # =====================================================================
        action_tensor = self._pad_action_to_b1k(action)
        
        # Final sanity check
        assert action_tensor.shape == (1, B1K_ACTION_DIM), \
            f"Expected action shape (1, {B1K_ACTION_DIM}), got {action_tensor.shape}"
        
        if self.debug:
            logger.info(f"  Final action: shape={action_tensor.shape}")
            logger.info("=" * 40)
        
        return action_tensor
    
    def reset(self) -> None:
        """
        Reset the policy state.

        Called at the start of each episode to clear any internal history.
        Note: This may be called multiple times during evaluation setup (e.g., on initial
        connection, episode start, trial start). This is normal BEHAVIOR-1K behavior.
        """
        import traceback

        # Only count as a new episode if we actually ran some steps in the previous episode
        # This prevents the episode counter from incrementing during startup resets
        had_steps = self.step_counter > 0

        if had_steps:
            logger.info(f"Episode {self.episode_counter} completed: {self.step_counter} steps")
            self._save_metrics()  # Save episode metrics before incrementing counter
            self.episode_counter += 1
            logger.info(f"Starting Episode {self.episode_counter}")

        self.step_counter = 0
        self.actor.reset()

        if self.debug:
            # Print call stack to understand who's calling reset
            if not had_steps:
                logger.info(f"GE-ACT policy reset (startup/reconnect) - Episode {self.episode_counter}")
                logger.info("Call stack:")
                for line in traceback.format_stack()[-4:-1]:
                    logger.info("  " + line.strip())
    
    def forward(self, obs: Dict[str, Any]) -> torch.Tensor:
        """
        Alias for act() to match BEHAVIOR-1K policy interface.

        Some parts of BEHAVIOR-1K call policy.forward() instead of policy.act().
        """
        return self.act(obs)

    def _save_metrics(self) -> None:
        """
        Save metrics to disk for analysis.

        Saves episode-level metrics to metrics/episode_{N}.json
        """
        if not self.save_debug_images:
            return

        import json
        import os

        metrics = {
            "episode": self.episode_counter,
            "steps": self.step_counter,
        }

        # Save to metrics subdirectory (BEHAVIOR-1K also writes here)
        metrics_dir = os.path.join(self.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        
        filename = f"ge_act_episode_{self.episode_counter:03d}.json"
        filepath = os.path.join(metrics_dir, filename)

        with open(filepath, "w") as f:
            json.dump(metrics, f, indent=2)

    def print_statistics(self) -> None:
        """
        Print inference statistics.

        Useful for debugging and monitoring performance.
        """
        # Note: step_counter shows steps in last episode only
        logger.info("=" * 60)
        logger.info("GE-ACT Inference Statistics")
        logger.info("=" * 60)
        logger.info(f"  Total episodes:        {self.episode_counter}")
        logger.info(f"  Steps in last episode: {self.step_counter}")
        logger.info(f"  Current episode:       {self.episode_counter}")
        logger.info(f"  Current episode steps: {self.step_counter}")
        logger.info("=" * 60)

        # Also save to file if debug output is enabled
        if self.save_debug_images:
            import json
            import os

            stats = {
                "total_episodes": self.episode_counter,
                "current_episode": self.episode_counter,
                "current_episode_steps": self.step_counter,
            }

            filepath = os.path.join(self.metrics_dir, "overall_statistics.json")
            with open(filepath, "w") as f:
                json.dump(stats, f, indent=2)
            logger.info(f"Statistics saved to: {filepath}")
