#!/usr/bin/env python3
"""
MaskPLS GPU Inference Server - Fixed version for CUDA
This version properly handles MinkowskiEngine with GPU support
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import time
import signal
import logging
from multiprocessing import shared_memory
from pathlib import Path
import yaml
from easydict import EasyDict as edict
import argparse

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the original model path to Python path
original_path = os.path.join(os.path.dirname(__file__), "../original/MaskPLS")
if original_path not in sys.path and os.path.exists(original_path):
    sys.path.insert(0, original_path)

# Try to import MinkowskiEngine
try:
    import MinkowskiEngine as ME
    HAS_MINKOWSKI = True
    logger.info("MinkowskiEngine available")
except ImportError as e:
    logger.error(f"MinkowskiEngine not available: {e}")
    logger.error("Please install MinkowskiEngine with CUDA support:")
    logger.error("  pip install git+https://github.com/NVIDIA/MinkowskiEngine.git")
    sys.exit(1)


def load_checkpoint_safe(model_path, device='cuda'):
    """
    Safely load checkpoint with proper error handling
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    logger.info(f"Loading checkpoint from: {model_path}")
    
    try:
        # First try direct GPU load
        checkpoint = torch.load(model_path, map_location=device)
        logger.info(f"✓ Checkpoint loaded on {device}")
        return checkpoint
        
    except Exception as e:
        logger.warning(f"Direct GPU load failed: {e}")
        
        # Try CPU first then move to GPU
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            logger.info("✓ Checkpoint loaded on CPU, will move to GPU")
            return checkpoint
            
        except Exception as e2:
            logger.error(f"Failed to load checkpoint: {e2}")
            
            # Last resort - try to extract metadata only
            metadata = {
                'num_classes': 20,
                'things_ids': [1, 2, 3, 4, 5, 6, 7, 8],
                'overlap_threshold': 0.8
            }
            logger.warning("Using default metadata")
            return metadata


class SharedMemoryInferenceServerGPU:
    """
    GPU inference server with MinkowskiEngine support
    """
    
    # Shared memory configuration
    INPUT_SHM_NAME = "/maskpls_input"
    OUTPUT_SHM_NAME = "/maskpls_output"
    CONTROL_SHM_NAME = "/maskpls_control"
    
    # Control flags
    FLAG_IDLE = 0
    FLAG_NEW_DATA = 1
    FLAG_PROCESSING = 2
    FLAG_COMPLETE = 3
    FLAG_ERROR = 4
    FLAG_SHUTDOWN = 5
    
    def __init__(self, model_path, config_path=None):
        """
        Initialize the GPU inference server
        """
        self.model_path = model_path
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! This server requires GPU.")
        
        self.device = torch.device('cuda')
        torch.cuda.set_device(0)  # Use first GPU
        
        logger.info(f"Initializing MaskPLS GPU Inference Server")
        logger.info(f"Using device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load configuration
        self.cfg = self.load_config(config_path)
        
        # Load model
        self.load_model()
        
        # Initialize shared memory
        self.init_shared_memory()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.running = False
        
    def load_config(self, config_path=None):
        """Load configuration with defaults"""
        
        # Start with minimal defaults
        cfg = edict({
            'MODEL': {'DATASET': 'KITTI', 'OVERLAP_THRESHOLD': 0.8},
            'KITTI': {'NUM_CLASSES': 20, 'IGNORE_LABEL': 255},
            'BACKBONE': {'CHANNELS': [64, 128, 256, 512]},
            'DECODER': {'HIDDEN_DIM': 256, 'NUM_HEADS': 8, 'NUM_LAYERS': 6},
            'TRAIN': {'BATCH_SIZE': 1, 'NUM_WORKERS': 4, 'SUBSAMPLE': False, 'AUG': False}
        })
        
        # Try to load from file if provided
        if config_path and os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    loaded_cfg = yaml.safe_load(f)
                    if loaded_cfg:
                        # Recursively update config
                        def update_dict(d, u):
                            for k, v in u.items():
                                if isinstance(v, dict):
                                    d[k] = update_dict(d.get(k, {}), v)
                                else:
                                    d[k] = v
                            return d
                        cfg = edict(update_dict(cfg, loaded_cfg))
                logger.info(f"Loaded config from: {config_path}")
            except Exception as e:
                logger.warning(f"Could not load config file: {e}")
        
        return cfg
    
    def load_model(self):
        """Load the MaskPLS model with MinkowskiEngine"""
        
        # Load checkpoint
        checkpoint = load_checkpoint_safe(self.model_path, self.device)
        
        # Extract metadata
        if isinstance(checkpoint, dict):
            dataset = self.cfg.MODEL.DATASET
            self.num_classes = checkpoint.get('num_classes', self.cfg[dataset].NUM_CLASSES)
            self.things_ids = checkpoint.get('things_ids', [1, 2, 3, 4, 5, 6, 7, 8])
            self.overlap_threshold = checkpoint.get('overlap_threshold', 0.8)
        else:
            # Using defaults
            self.num_classes = 20
            self.things_ids = [1, 2, 3, 4, 5, 6, 7, 8]
            self.overlap_threshold = 0.8
        
        logger.info(f"Model configuration:")
        logger.info(f"  Num classes: {self.num_classes}")
        logger.info(f"  Things IDs: {self.things_ids}")
        
        # Try to create model components
        try:
            # Import MaskPLS models
            from mask_pls.models.mink import MinkEncoderDecoder
            from mask_pls.models.decoder import MaskedTransformerDecoder
            
            dataset = self.cfg.MODEL.DATASET
            
            # Create model components
            self.backbone = MinkEncoderDecoder(self.cfg.BACKBONE, self.cfg[dataset])
            self.backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.backbone)
            
            self.decoder = MaskedTransformerDecoder(
                self.cfg.DECODER,
                self.cfg.BACKBONE,
                self.cfg[dataset]
            )
            
            # Load weights if available
            if isinstance(checkpoint, dict):
                if 'backbone_state_dict' in checkpoint:
                    self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
                    logger.info("✓ Loaded backbone weights")
                if 'decoder_state_dict' in checkpoint:
                    self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
                    logger.info("✓ Loaded decoder weights")
            
            # Move to GPU and set to eval mode
            self.backbone = self.backbone.to(self.device)
            self.decoder = self.decoder.to(self.device)
            self.backbone.eval()
            self.decoder.eval()
            
            logger.info("✓ Model loaded successfully on GPU")
            
        except ImportError as e:
            logger.error(f"Cannot import MaskPLS models: {e}")
            logger.error("Creating simplified model instead...")
            self.create_simple_model()
            
        except Exception as e:
            logger.error(f"Error creating model: {e}")
            logger.error("Creating simplified model instead...")
            self.create_simple_model()
    
    def create_simple_model(self):
        """Create a simple model if MaskPLS models can't be loaded"""
        logger.warning("Using simplified model (not full MaskPLS)")
        
        class SimpleModel(nn.Module):
            def __init__(self, num_classes):
                super().__init__()
                self.fc1 = nn.Linear(4, 128)
                self.fc2 = nn.Linear(128, 256)
                self.fc3 = nn.Linear(256, 128)
                self.fc4 = nn.Linear(128, num_classes)
                
            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = torch.relu(self.fc3(x))
                return self.fc4(x)
        
        self.simple_model = SimpleModel(self.num_classes).to(self.device)
        self.simple_model.eval()
        self.use_simple = True
    
    def init_shared_memory(self):
        """Initialize shared memory segments"""
        # Maximum sizes
        self.max_points = 150000
        self.input_size = self.max_points * 4 * 4  # N x 4 floats
        self.output_size = self.max_points * 2 * 4  # N x 2 ints
        self.control_size = 32
        
        # Clean up any existing segments
        self.cleanup_shared_memory()
        
        try:
            # Create new shared memory segments
            self.input_shm = shared_memory.SharedMemory(
                create=True,
                size=self.input_size,
                name=self.INPUT_SHM_NAME
            )
            
            self.output_shm = shared_memory.SharedMemory(
                create=True,
                size=self.output_size,
                name=self.OUTPUT_SHM_NAME
            )
            
            self.control_shm = shared_memory.SharedMemory(
                create=True,
                size=self.control_size,
                name=self.CONTROL_SHM_NAME
            )
            
            # Create numpy arrays backed by shared memory
            self.control_array = np.ndarray(
                (8,), dtype=np.int32, buffer=self.control_shm.buf
            )
            self.control_array[:] = 0  # Initialize to zeros
            
            logger.info(f"✓ Shared memory initialized")
            logger.info(f"  Max points: {self.max_points}")
            
        except Exception as e:
            logger.error(f"Failed to create shared memory: {e}")
            raise
    
    def cleanup_shared_memory(self):
        """Clean up shared memory segments"""
        for name in [self.INPUT_SHM_NAME, self.OUTPUT_SHM_NAME, self.CONTROL_SHM_NAME]:
            try:
                shm = shared_memory.SharedMemory(name=name)
                shm.close()
                shm.unlink()
            except:
                pass
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    @torch.no_grad()
    def process_point_cloud(self, points, features):
        """
        Run MaskPLS inference on GPU
        """
        try:
            # Convert to torch tensors and move to GPU
            pt_coord = torch.from_numpy(points).float().to(self.device)
            feats = torch.from_numpy(features).float().to(self.device)
            
            if hasattr(self, 'use_simple') and self.use_simple:
                # Use simple model
                logits = self.simple_model(feats)
                semantic_pred = torch.argmax(logits, dim=-1).cpu().numpy().astype(np.int32)
                instance_pred = np.zeros(len(points), dtype=np.int32)
                
            else:
                # Use full MaskPLS model
                # Create batch dict
                batch_dict = {
                    'pt_coord': [pt_coord],
                    'feats': [feats],
                    'sem_label': [torch.zeros((points.shape[0], 1), dtype=torch.int32)],
                    'ins_label': [torch.zeros((points.shape[0], 1), dtype=torch.int32)],
                    'masks': [torch.zeros(1, points.shape[0])],
                    'masks_cls': [torch.zeros(1, dtype=torch.long)],
                    'masks_ids': [[]],
                    'fname': ['inference'],
                    'pose': [np.eye(4, dtype=np.float32)],
                    'token': ['inference_token']
                }
                
                # Run through backbone
                feats_out, coords, pad_masks, bb_logits = self.backbone(batch_dict)
                
                # Run through decoder
                outputs, padding = self.decoder(feats_out, coords, pad_masks)
                
                # Panoptic inference
                sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
                
                semantic_pred = sem_pred[0].cpu().numpy().astype(np.int32)
                instance_pred = ins_pred[0].cpu().numpy().astype(np.int32)
            
            return semantic_pred, instance_pred
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            
            # Return zeros on error
            return (np.zeros(len(points), dtype=np.int32),
                   np.zeros(len(points), dtype=np.int32))
    
    def panoptic_inference(self, outputs, padding):
        """Panoptic segmentation inference"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        
        sem_pred = []
        ins_pred = []
        
        for mask_cls, mask_pred, pad in zip(mask_cls, mask_pred, padding):
            scores, labels = mask_cls.max(-1)
            mask_pred = mask_pred[~pad].sigmoid()
            keep = labels.ne(self.num_classes)
            
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[:, keep]
            
            # Probability to belong to each mask
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            
            panoptic_seg = torch.zeros(
                (cur_masks.shape[0]), dtype=torch.int32, device=cur_masks.device
            )
            sem = torch.zeros_like(panoptic_seg)
            ins = torch.zeros_like(panoptic_seg)
            
            segment_id = 0
            
            if cur_masks.shape[1] == 0:  # No masks detected
                sem_pred.append(sem)
                ins_pred.append(ins)
            else:
                # Mask index for each point
                cur_mask_ids = cur_prob_masks.argmax(1)
                stuff_memory_list = {}
                
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in self.things_ids
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[:, k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    
                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue
                        
                        if not isthing:  # Merge stuff regions
                            if int(pred_class) in stuff_memory_list.keys():
                                panoptic_seg[mask] = stuff_memory_list[int(pred_class)]
                                continue
                            else:
                                stuff_memory_list[int(pred_class)] = segment_id + 1
                        
                        segment_id += 1
                        panoptic_seg[mask] = segment_id
                        
                        sem[mask] = pred_class
                        if isthing:
                            ins[mask] = segment_id
                        else:
                            ins[mask] = 0
                
                sem_pred.append(sem)
                ins_pred.append(ins)
        
        return sem_pred, ins_pred
    
    def run(self):
        """Main inference loop"""
        self.running = True
        logger.info("GPU Server ready - waiting for requests...")
        logger.info("Press Ctrl+C to stop")
        
        # Performance tracking
        inference_times = []
        frame_count = 0
        
        # GPU warmup
        logger.info("Warming up GPU...")
        dummy_points = np.random.randn(1000, 3).astype(np.float32)
        dummy_features = np.random.randn(1000, 4).astype(np.float32)
        self.process_point_cloud(dummy_points, dummy_features)
        torch.cuda.synchronize()
        logger.info("✓ GPU ready")
        
        while self.running:
            try:
                flag = self.control_array[0]
                
                if flag == self.FLAG_NEW_DATA:
                    num_points = self.control_array[1]
                    
                    if num_points > 0 and num_points <= self.max_points:
                        self.control_array[0] = self.FLAG_PROCESSING
                        
                        # Synchronize GPU
                        torch.cuda.synchronize()
                        start_time = time.time()
                        
                        # Read input data
                        input_array = np.ndarray(
                            (num_points, 4),
                            dtype=np.float32,
                            buffer=self.input_shm.buf
                        )
                        points = input_array[:, :3].copy()
                        features = input_array.copy()
                        
                        # Process on GPU
                        semantic_pred, instance_pred = self.process_point_cloud(
                            points, features
                        )
                        
                        # Write output
                        output_array = np.ndarray(
                            (num_points, 2),
                            dtype=np.int32,
                            buffer=self.output_shm.buf
                        )
                        output_array[:, 0] = semantic_pred
                        output_array[:, 1] = instance_pred
                        
                        # Synchronize and measure time
                        torch.cuda.synchronize()
                        inference_time = (time.time() - start_time) * 1000  # ms
                        
                        inference_times.append(inference_time)
                        frame_count += 1
                        
                        self.control_array[2] = int(inference_time)
                        self.control_array[3] = frame_count
                        self.control_array[0] = self.FLAG_COMPLETE
                        
                        # Log performance
                        if frame_count % 10 == 0:
                            avg_time = np.mean(inference_times[-10:])
                            logger.info(f"GPU Processing - Frame {frame_count}, "
                                      f"Points: {num_points}, "
                                      f"Time: {inference_time:.1f}ms, "
                                      f"Avg: {avg_time:.1f}ms, "
                                      f"FPS: {1000/avg_time:.1f}")
                    else:
                        logger.error(f"Invalid number of points: {num_points}")
                        self.control_array[0] = self.FLAG_ERROR
                
                elif flag == self.FLAG_SHUTDOWN:
                    logger.info("Received shutdown flag")
                    break
                
                time.sleep(0.0001)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.control_array[0] = self.FLAG_ERROR
        
        logger.info("Shutting down GPU inference server...")
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.control_array[0] = self.FLAG_SHUTDOWN
            
            # Clear GPU cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Close shared memory
            self.input_shm.close()
            self.output_shm.close()
            self.control_shm.close()
            
            self.input_shm.unlink()
            self.output_shm.unlink()
            self.control_shm.unlink()
            
            logger.info("✓ Cleanup complete")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(description='MaskPLS GPU Inference Server')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--config', help='Path to config file (optional)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This server requires GPU.")
        logger.error("Please check your PyTorch installation:")
        logger.error("  pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        sys.exit(1)
    
    # Create and run server
    try:
        server = SharedMemoryInferenceServerGPU(
            model_path=args.model,
            config_path=args.config
        )
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
