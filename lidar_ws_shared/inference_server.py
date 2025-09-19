#!/usr/bin/env python3
"""
MaskPLS GPU Inference Server - Fixed to load converted models properly
This version defines the model class before loading to avoid unpickling errors
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

# CRITICAL: Add the original model path to Python path BEFORE any imports
original_path = os.path.join(os.path.dirname(__file__), "../original/MaskPLS")
if os.path.exists(original_path):
    sys.path.insert(0, original_path)
    logger.info(f"Added to path: {original_path}")
else:
    logger.warning(f"Original MaskPLS path not found: {original_path}")

# Import MinkowskiEngine FIRST
try:
    import MinkowskiEngine as ME
    HAS_MINKOWSKI = True
    logger.info("✓ MinkowskiEngine imported successfully")
except ImportError as e:
    logger.error(f"MinkowskiEngine not available: {e}")
    sys.exit(1)

# Import MaskPLS models BEFORE defining the model class
try:
    from mask_pls.models.mink import MinkEncoderDecoder
    from mask_pls.models.decoder import MaskedTransformerDecoder
    logger.info("✓ MaskPLS models imported successfully")
except ImportError as e:
    logger.error(f"Cannot import MaskPLS models: {e}")
    logger.error("Make sure the original MaskPLS code is in the correct path")
    sys.exit(1)


# CRITICAL: Define the EXACT SAME model class used in the converter
class OriginalStandaloneModel(nn.Module):
    """Standalone inference model using original MinkowskiEngine architecture"""
    
    def __init__(self, cfg, things_ids):
        super().__init__()
        
        dataset = cfg.MODEL.DATASET
        self.cfg = cfg
        self.num_classes = cfg[dataset].NUM_CLASSES
        self.ignore_label = cfg[dataset].IGNORE_LABEL
        self.things_ids = things_ids
        self.overlap_threshold = cfg.MODEL.OVERLAP_THRESHOLD
        
        # Create the ORIGINAL components
        backbone = MinkEncoderDecoder(cfg.BACKBONE, cfg[dataset])
        self.backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(backbone)
        
        self.decoder = MaskedTransformerDecoder(
            cfg.DECODER,
            cfg.BACKBONE,
            cfg[dataset]
        )
        
        logger.info(f"OriginalStandaloneModel initialized:")
        logger.info(f"  Num classes: {self.num_classes}")
        logger.info(f"  Things IDs: {self.things_ids}")
    
    @torch.no_grad()
    def forward(self, batch_dict):
        """Forward pass matching the original training model"""
        feats, coords, pad_masks, bb_logits = self.backbone(batch_dict)
        outputs, padding = self.decoder(feats, coords, pad_masks)
        return outputs, padding, bb_logits
    
    def panoptic_inference(self, outputs, padding):
        """Panoptic segmentation inference"""
        mask_cls = outputs["pred_logits"]
        mask_pred = outputs["pred_masks"]
        things_ids = self.things_ids
        num_classes = self.num_classes
        
        sem_pred = []
        ins_pred = []
        
        for mask_cls, mask_pred, pad in zip(mask_cls, mask_pred, padding):
            scores, labels = mask_cls.max(-1)
            mask_pred = mask_pred[~pad].sigmoid()
            keep = labels.ne(num_classes)
            
            cur_scores = scores[keep]
            cur_classes = labels[keep]
            cur_masks = mask_pred[:, keep]
            
            cur_prob_masks = cur_scores.unsqueeze(0) * cur_masks
            
            panoptic_seg = torch.zeros(
                (cur_masks.shape[0]), dtype=torch.int32, device=cur_masks.device
            )
            sem = torch.zeros_like(panoptic_seg)
            ins = torch.zeros_like(panoptic_seg)
            
            segment_id = 0
            
            if cur_masks.shape[1] == 0:
                # FIX: Always convert to CPU and numpy
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
            else:
                cur_mask_ids = cur_prob_masks.argmax(1)
                stuff_memory_list = {}
                
                for k in range(cur_classes.shape[0]):
                    pred_class = cur_classes[k].item()
                    isthing = pred_class in things_ids
                    mask_area = (cur_mask_ids == k).sum().item()
                    original_area = (cur_masks[:, k] >= 0.5).sum().item()
                    mask = (cur_mask_ids == k) & (cur_masks[:, k] >= 0.5)
                    
                    if mask_area > 0 and original_area > 0 and mask.sum().item() > 0:
                        if mask_area / original_area < self.overlap_threshold:
                            continue
                        if not isthing:
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
                            
                # FIX: Always convert to CPU and numpy
                sem_pred.append(sem.cpu().numpy())
                ins_pred.append(ins.cpu().numpy())
        
        return sem_pred, ins_pred


class SharedMemoryInferenceServerGPU:
    """
    GPU inference server with proper model loading
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
        """Initialize the GPU inference server"""
        
        self.model_path = model_path
        
        # Check CUDA availability
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA not available! This server requires GPU.")
        
        self.device = torch.device('cuda')
        torch.cuda.set_device(0)
        
        logger.info(f"Initializing MaskPLS GPU Inference Server")
        logger.info(f"Using device: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        
        # Load model (this must happen before shared memory to ensure model loads properly)
        self.load_model_properly()
        
        # Initialize shared memory
        self.init_shared_memory()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
        
        self.running = False
    
    def load_model_properly(self):
        """
        Load the converted model properly to avoid unpickling errors
        """
        logger.info(f"Loading model from: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        try:
            # Load the checkpoint - use weights_only=False to allow EasyDict
            logger.info("Loading checkpoint...")
            checkpoint = torch.load(self.model_path, map_location='cpu', weights_only=False)
            
            # Extract configuration and metadata
            if not isinstance(checkpoint, dict):
                raise ValueError("Checkpoint is not a dictionary. Was the model converted properly?")
            
            # Check if this is a converted model
            if 'model_class' in checkpoint and checkpoint['model_class'] != 'OriginalStandaloneModel':
                raise ValueError(f"Unexpected model class: {checkpoint['model_class']}")
            
            # Get config and metadata
            cfg = checkpoint.get('config', self.get_default_config())
            # Convert to EasyDict if needed
            if not isinstance(cfg, edict):
                self.cfg = edict(cfg)
            else:
                self.cfg = cfg
                
            self.things_ids = checkpoint.get('things_ids', [1, 2, 3, 4, 5, 6, 7, 8])
            self.num_classes = checkpoint.get('num_classes', 20)
            self.overlap_threshold = checkpoint.get('overlap_threshold', 0.8)
            
            logger.info(f"Checkpoint loaded successfully:")
            logger.info(f"  Num classes: {self.num_classes}")
            logger.info(f"  Things IDs: {self.things_ids}")
            logger.info(f"  Architecture: {checkpoint.get('architecture', 'Unknown')}")
            
            # Create the model with the same configuration
            logger.info("Creating model...")
            self.model = OriginalStandaloneModel(self.cfg, self.things_ids)
            
            # Load the state dict
            if 'model_state_dict' in checkpoint:
                logger.info("Loading model weights...")
                self.model.load_state_dict(checkpoint['model_state_dict'])
                logger.info("✓ Model weights loaded successfully")
            elif 'backbone_state_dict' in checkpoint and 'decoder_state_dict' in checkpoint:
                logger.info("Loading backbone and decoder weights separately...")
                self.model.backbone.load_state_dict(checkpoint['backbone_state_dict'])
                self.model.decoder.load_state_dict(checkpoint['decoder_state_dict'])
                logger.info("✓ Weights loaded successfully")
            else:
                logger.warning("No model weights found in checkpoint, using random initialization")
            
            # Move model to GPU and set to eval mode
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.info("✓ Model loaded and ready on GPU")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            logger.error("Stack trace:", exc_info=True)
            
            # Try to provide helpful debugging info
            try:
                checkpoint_keys = torch.load(self.model_path, map_location='cpu', weights_only=False).keys()
                logger.info(f"Checkpoint contains keys: {list(checkpoint_keys)[:10]}...")
            except:
                pass
            
            raise RuntimeError(f"Cannot load model from {self.model_path}: {e}")
    
    def get_default_config(self):
        """Get default configuration if not in checkpoint"""
        return edict({
            'MODEL': {'DATASET': 'KITTI', 'OVERLAP_THRESHOLD': 0.8},
            'KITTI': {'NUM_CLASSES': 20, 'IGNORE_LABEL': 255},
            'BACKBONE': {'CHANNELS': [64, 128, 256, 512]},
            'DECODER': {'HIDDEN_DIM': 256, 'NUM_HEADS': 8, 'NUM_LAYERS': 6},
            'TRAIN': {'BATCH_SIZE': 1, 'NUM_WORKERS': 4, 'SUBSAMPLE': False, 'AUG': False}
        })
    
    def init_shared_memory(self):
        """Initialize shared memory segments"""
        self.max_points = 150000
        self.input_size = self.max_points * 4 * 4
        self.output_size = self.max_points * 2 * 4
        self.control_size = 32
        
        # Clean up any existing segments
        self.cleanup_shared_memory()
        
        try:
            self.input_shm = shared_memory.SharedMemory(
                create=True, size=self.input_size, name=self.INPUT_SHM_NAME
            )
            self.output_shm = shared_memory.SharedMemory(
                create=True, size=self.output_size, name=self.OUTPUT_SHM_NAME
            )
            self.control_shm = shared_memory.SharedMemory(
                create=True, size=self.control_size, name=self.CONTROL_SHM_NAME
            )
            
            self.control_array = np.ndarray(
                (8,), dtype=np.int32, buffer=self.control_shm.buf
            )
            self.control_array[:] = 0
            
            logger.info(f"✓ Shared memory initialized (max {self.max_points} points)")
            
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
        """Run MaskPLS inference on GPU"""
        try:
            # Move data to GPU
            pt_coord = torch.from_numpy(points).float().to(self.device)
            feats = torch.from_numpy(features).float().to(self.device)
            
            # Create batch dict for MaskPLS
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
            
            # Run through model
            outputs, padding, bb_logits = self.model(batch_dict)
            
            # Panoptic inference - already returns numpy arrays
            sem_pred, ins_pred = self.model.panoptic_inference(outputs, padding)
            
            # Now sem_pred and ins_pred are already numpy arrays
            semantic_pred = sem_pred[0].astype(np.int32)
            instance_pred = ins_pred[0].astype(np.int32)
            
            return semantic_pred, instance_pred
            
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            logger.error("Stack trace:", exc_info=True)
            
            # Return zeros on error
            return (np.zeros(len(points), dtype=np.int32),
                   np.zeros(len(points), dtype=np.int32))
    
    def run(self):
        """Main inference loop"""
        self.running = True
        logger.info("="*60)
        logger.info("GPU Server ready - waiting for requests...")
        logger.info("Press Ctrl+C to stop")
        logger.info("="*60)
        
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
                        
                        torch.cuda.synchronize()
                        start_time = time.time()
                        
                        # Read input
                        input_array = np.ndarray(
                            (num_points, 4), dtype=np.float32,
                            buffer=self.input_shm.buf
                        )
                        points = input_array[:, :3].copy()
                        features = input_array.copy()
                        
                        # Process
                        semantic_pred, instance_pred = self.process_point_cloud(points, features)
                        
                        # Write output
                        output_array = np.ndarray(
                            (num_points, 2), dtype=np.int32,
                            buffer=self.output_shm.buf
                        )
                        output_array[:, 0] = semantic_pred
                        output_array[:, 1] = instance_pred
                        
                        torch.cuda.synchronize()
                        inference_time = (time.time() - start_time) * 1000
                        
                        inference_times.append(inference_time)
                        frame_count += 1
                        
                        self.control_array[2] = int(inference_time)
                        self.control_array[3] = frame_count
                        self.control_array[0] = self.FLAG_COMPLETE
                        
                        # Log performance
                        if frame_count % 10 == 0:
                            avg_time = np.mean(inference_times[-10:])
                            logger.info(f"Frame {frame_count}: {num_points} points, "
                                      f"{inference_time:.1f}ms (avg: {avg_time:.1f}ms, "
                                      f"{1000/avg_time:.1f} FPS)")
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
        
        logger.info("Shutting down...")
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            self.control_array[0] = self.FLAG_SHUTDOWN
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
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
    parser = argparse.ArgumentParser(description='MaskPLS GPU Inference Server (Fixed)')
    parser.add_argument('--model', required=True, help='Path to converted model file')
    parser.add_argument('--config', help='Path to config file (optional)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Check GPU availability
    if not torch.cuda.is_available():
        logger.error("CUDA not available! This server requires GPU.")
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
