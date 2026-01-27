#!/usr/bin/env python3
"""
GE-ACT Server for BEHAVIOR-1K
=============================

This script starts a WebSocket server that:
1. Receives observations from BEHAVIOR-1K simulation
2. Passes them to GE-ACT model for action prediction
3. Returns actions back to BEHAVIOR-1K

HOW TO USE:
    Terminal 1 (GE-ACT server):
        CUDA_VISIBLE_DEVICES=1 python serve_ge_act_b1k.py \\
            --config configs/ltx_model/b1k/action_model_b1k.yaml \\
            --weight /path/to/ge_act_calvin.safetensors \\
            --task_name turning_on_radio \\
            --port 8000
    
    Terminal 2 (BEHAVIOR-1K):
        CUDA_VISIBLE_DEVICES=0 python eval.py policy=websocket ...

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────────┐
    │                     serve_ge_act_b1k.py                          │
    │                                                                  │
    │   main()                                                         │
    │     │                                                            │
    │     ├─→ get_task_prompt()     # Fetch prompt from metadata       │
    │     │                                                            │
    │     ├─→ GEActB1KWrapper()     # Create policy wrapper            │
    │     │       │                                                    │
    │     │       └─→ MVActor()     # Load GE-ACT model                │
    │     │                                                            │
    │     └─→ WebsocketPolicyServer() # Start server on port 8000      │
    │               │                                                  │
    │               └─→ serve_forever()  # Wait for connections        │
    │                       │                                          │
    │                       └─→ policy.act(obs)  # On each request     │
    └─────────────────────────────────────────────────────────────────┘

Based on: b1k-baselines/baselines/openpi/scripts/serve_b1k.py
"""

import argparse
import logging
import os
import socket
import sys

# =============================================================================
# PATH SETUP
# =============================================================================
# Add Genie-Envisioner root to Python path so we can import our modules
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

# Add BEHAVIOR-1K to Python path for WebsocketPolicyServer
BEHAVIOR_1K_PATH = "/shared_work/physical_intelligence/BEHAVIOR-1K"
sys.path.insert(0, os.path.join(BEHAVIOR_1K_PATH, "OmniGibson"))

# =============================================================================
# IMPORTS
# =============================================================================
# Import our wrapper (this is the file we just documented)
from web_infer_utils.ge_act_b1k_wrapper import GEActB1KWrapper

# Try to import BEHAVIOR-1K components
# These may not be available if BEHAVIOR-1K is not installed
try:
    from omnigibson.learning.utils.network_utils import WebsocketPolicyServer
    from omnigibson.learning.datas import BehaviorLerobotDatasetMetadata
    HAS_BEHAVIOR_1K = True
    print("✓ BEHAVIOR-1K imports successful")
except ImportError as e:
    print(f"✗ Could not import BEHAVIOR-1K: {e}")
    HAS_BEHAVIOR_1K = False

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    force=True
)
logger = logging.getLogger(__name__)

# Suppress noisy WebSocket handshake errors (these occur when browsers/monitors ping the port)
# They don't affect actual BEHAVIOR-1K client connections
logging.getLogger("websockets.server").setLevel(logging.WARNING)


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def get_args():
    """
    Parse command-line arguments.
    
    All arguments have sensible defaults for the BEHAVIOR-1K + Calvin model setup.
    """
    parser = argparse.ArgumentParser(
        description="Serve GE-ACT policy for BEHAVIOR-1K evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # Show defaults in help
    )
    
    # -------------------------------------------------------------------------
    # Required arguments (model files)
    # -------------------------------------------------------------------------
    parser.add_argument(
        "-c", "--config",
        type=str,
        required=True,
        help="Path to the GE-ACT config YAML file (e.g., action_model_b1k.yaml)"
    )
    
    parser.add_argument(
        "-w", "--weight",
        type=str,
        required=True,
        help="Path to the model weights .safetensors file"
    )
    
    # -------------------------------------------------------------------------
    # Task configuration
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--task_name",
        type=str,
        default="turning_on_radio",
        help="BEHAVIOR-1K task name (used to fetch the task prompt)"
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="Testing Prompt",
        help="Override task prompt manually (if not provided, fetched from dataset)"
    )
    
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/scr/behavior/2025-challenge-demos",
        help="Path to BEHAVIOR-1K dataset for metadata lookup"
    )
    
    # -------------------------------------------------------------------------
    # Server configuration
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind (0.0.0.0 = accept from any IP)"
    )
    
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8000,
        help="Port number for WebSocket server"
    )
    
    # -------------------------------------------------------------------------
    # Model configuration
    # -------------------------------------------------------------------------
    # Domain name specifies which robot platform's normalization statistics to use.
    # The model uses mean/std values to normalize input states and denormalize output actions.
    # Different robot platforms (behavior1k, calvin, libero, etc.) have different action ranges,
    # so each needs its own statistics. E.g., "behavior1k" looks up "behavior1k_delta_joint"
    # and "behavior1k_state_joint" from the statistics JSON file.
    parser.add_argument(
        "--domain_name",
        type=str,
        default="behavior1k",
        help="Domain name for statistics lookup in config"
    )
    
    # NOTE: action_dim = 22 because we're using Calvin-trained weights.
    # If using a GE-ACT model trained on BEHAVIOR-1K, this would be 23 (matching R1Pro robot)
    # and no padding would be needed in the wrapper.
    parser.add_argument(
        "--action_dim",
        type=int,
        default=22,
        help="GE-ACT model's action dimension (Calvin = 22: full action prediction for dual-arm or EEF+state)"
    )
    
    parser.add_argument(
        "--denoise_steps",
        type=int,
        default=5,
        help="Number of diffusion denoising steps (more = better but slower)"
    )
    
    # NOTE: Using 256x256 to match Calvin model training data.
    # If using a GE-ACT model trained on BEHAVIOR-1K, use 224x224 instead
    # (see b1k-baselines/baselines/openpi/src/openpi/shared/eval_b1k_wrapper.py:8)
    parser.add_argument(
        "--image_height",
        type=int,
        default=256,
        help="Target image height for model input (Calvin=256, B1K=224)"
    )
    
    parser.add_argument(
        "--image_width",
        type=int,
        default=256,
        help="Target image width for model input (Calvin=256, B1K=224)"
    )
    
    parser.add_argument(
        "--save_debug_images",
        action="store_true",
        help="Save received images to disk for debugging"
    )
    
    parser.add_argument(
        "--debug_image_dir",
        type=str,
        default="/shared_work/physical_intelligence/ruiheng/tmp/ge_act_b1k_debug_images",
        help="Directory to save debug images"
    )
    
    parser.add_argument(
        "--test_mode",
        action="store_true",
        help="Test mode: return zero actions without running the model (for testing infrastructure)"
    )
    
    return parser.parse_args()


# =============================================================================
# TASK PROMPT LOOKUP
# =============================================================================
# NOTE: This function is only called if --prompt is NOT specified.
# If you provide --prompt="Your custom prompt", this is skipped entirely.

def get_task_prompt(task_name: str, dataset_root: str) -> str:
    """
    Get the natural language task prompt for a given task name.
    
    BEHAVIOR-1K tasks have associated text descriptions like:
    - "Turn on the radio receiver that's on the table in the living room."
    - "Pick up the trash on the floor and put it in the trash can."
    
    These prompts are stored in the dataset metadata.
    
    Args:
        task_name: Task identifier (e.g., "turning_on_radio")
        dataset_root: Path to BEHAVIOR-1K dataset
        
    Returns:
        Task description string
    """
    # Fallback prompts in case metadata lookup fails
    FALLBACK_PROMPTS = {
        "turning_on_radio": "Turn on the radio receiver that's on the table in the living room.",
        "picking_up_trash": "Pick up the trash on the floor and put it in the trash can.",
        # Add more as needed
    }
    
    if not HAS_BEHAVIOR_1K:
        logger.warning("BEHAVIOR-1K not installed, using fallback prompt")
        return FALLBACK_PROMPTS.get(task_name, f"Complete the {task_name} task.")
    
    try:
        # Load metadata from dataset
        metadata = BehaviorLerobotDatasetMetadata(
            repo_id="behavior-1k/2025-challenge-demos",
            root=dataset_root,
            tasks=[task_name],
            modalities=[],
            cameras=[],
        )
        prompt = list(metadata.tasks.values())[0]
        logger.info(f"Loaded prompt from metadata: {prompt}")
        return prompt
        
    except Exception as e:
        logger.warning(f"Could not get prompt from metadata: {e}")
        return FALLBACK_PROMPTS.get(task_name, f"Complete the {task_name} task.")


# =============================================================================
# VALIDATION FUNCTIONS
# =============================================================================

def validate_config(args) -> None:
    """
    Validate that all required files exist.
    
    Args:
        args: Parsed arguments
        
    Raises:
        FileNotFoundError: If required files are missing
    """
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Config file not found: {args.config}")
    
    if not os.path.exists(args.weight):
        raise FileNotFoundError(f"Weight file not found: {args.weight}")
    
    logger.info("✓ Config and weight files found")


def print_banner(args, prompt: str) -> None:
    """
    Print a nice banner with server configuration.
    """
    hostname = socket.gethostname()
    try:
        local_ip = socket.gethostbyname(hostname)
    except socket.gaierror:
        local_ip = "unknown"
    
    print("\n" + "=" * 60)
    print("GE-ACT Server for BEHAVIOR-1K")
    print("=" * 60)
    print(f"  Task:        {args.task_name}")
    print(f"  Prompt:      {prompt[:50]}..." if len(prompt) > 50 else f"  Prompt:      {prompt}")
    print(f"  Config:      {os.path.basename(args.config)}")
    print(f"  Weights:     {os.path.basename(args.weight)}")
    print(f"  Action dim:  {args.action_dim}")
    print(f"  Image size:  {args.image_height} x {args.image_width}")
    print(f"  Host:        {args.host}:{args.port}")
    print(f"  Hostname:    {hostname}")
    print(f"  Local IP:    {local_ip}")
    print("=" * 60 + "\n")


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """
    Main entry point.
    
    Flow:
    1. Parse arguments
    2. Validate files exist
    3. Get task prompt
    4. Create GE-ACT wrapper (loads model)
    5. Create WebSocket server
    6. Start serving forever
    """
    # =========================================================================
    # STEP 1: Parse arguments
    # =========================================================================
    args = get_args()
    
    # =========================================================================
    # STEP 2: Validate configuration
    # =========================================================================
    validate_config(args)
    
    # =========================================================================
    # STEP 3: Get task prompt
    # =========================================================================
    if args.prompt:
        # User provided prompt manually via --prompt argument
        prompt = args.prompt
        logger.info("Using user-provided prompt")
    else:
        # Fetch from dataset metadata (or fallback if metadata unavailable)
        prompt = get_task_prompt(args.task_name, args.dataset_root)
        # Note: get_task_prompt() logs whether it used metadata or fallback
    
    # Print configuration banner
    print_banner(args, prompt)
    
    # =========================================================================
    # STEP 4: Create GE-ACT policy wrapper
    # =========================================================================
    # This is where the model is loaded (takes a few seconds)
    logger.info("Creating GEActB1KWrapper (this loads the model)...")
    
    try:
        policy = GEActB1KWrapper(
            config_file=args.config,
            weight_file=args.weight,
            text_prompt=prompt,
            domain_name=args.domain_name,
            image_height=args.image_height,
            image_width=args.image_width,
            action_dim=args.action_dim,
            num_inference_steps=args.denoise_steps,
            save_debug_images=args.save_debug_images,
            debug_image_dir=args.debug_image_dir,
            test_mode=args.test_mode,
        )
        logger.info("✓ GEActB1KWrapper created successfully")
        logger.info("✓ GE-ACT model loaded and ready")
    except Exception as e:
        logger.error(f"✗ Failed to create GEActB1KWrapper: {e}")
        logger.error("Check that config and weight files are correct")
        raise
    
    # =========================================================================
    # STEP 5: Create WebSocket server
    # =========================================================================
    if not HAS_BEHAVIOR_1K:
        logger.error("=" * 60)
        logger.error("BEHAVIOR-1K is not installed!")
        logger.error("Cannot start WebSocket server without it.")
        logger.error("")
        logger.error("To fix, run:")
        logger.error("  conda activate genie_envisioner")
        logger.error("  cd /shared_work/physical_intelligence/BEHAVIOR-1K")
        logger.error("  pip install -e OmniGibson")
        logger.error("=" * 60)
        sys.exit(1)
    
    logger.info("Creating WebsocketPolicyServer...")
    try:
        server = WebsocketPolicyServer(
            policy=policy,
            host=args.host,
            port=args.port,
            metadata={
                "model": "GE-ACT",
                "task": args.task_name,
                "action_dim": args.action_dim,
            },
        )
        logger.info("✓ WebsocketPolicyServer created successfully")
    except Exception as e:
        logger.error(f"✗ Failed to create WebsocketPolicyServer: {e}")
        raise
    
    # =========================================================================
    # STEP 6: Start serving
    # =========================================================================
    logger.info("=" * 60)
    logger.info(f"Server ready! Listening on {args.host}:{args.port}")
    logger.info("Waiting for connections from BEHAVIOR-1K...")
    logger.info("Press Ctrl+C to stop.")
    logger.info("=" * 60)
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("\nServer stopped by user.")


if __name__ == "__main__":
    main()
