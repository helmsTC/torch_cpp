#!/usr/bin/env python3
"""
MaskPLS Inference Server using Shared Memory for C++ ROS2 Communication
This runs the full MaskPLS model with MinkowskiEngine
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
import struct

# Add the original model path to Python path
original_path = os.path.join(os.path.dirname(__file__), "../original/MaskPLS")
if original_path not in sys.path:
    sys.path.insert(0, original_path)

# Import the ORIGINAL models
try:
    from mask_pls.models.mink import MinkEncoderDecoder
    from mask_pls.models.decoder import MaskedTransformerDecoder
    from mask_pls.datasets.semantic_dataset import SemanticDatasetModule
    import MinkowskiEngine as ME
except ImportError as e:
    print(f"Error importing MaskPLS modules: {e}")
    print("Make sure the original MaskPLS code is in the correct location")
    sys.exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SharedMemoryInferenceServer:
    """
    Inference server that communicates with C++ ROS node via shared memory
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
    
    def __init__(self, model_path, config_path=None, use_cuda=True):
        """
        Initialize the inference server
        
        Args:
            model_path: Path to the converted .pt model file
            config_path: Optional path to config file
            use_cuda: Whether to use CUDA if available
        """
        self.model_path = model_path
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.device = torch.device('cuda' if self.use_cuda else 'cpu')
        
        logger.info(f"Initializing MaskPLS Inference Server")
        logger.info(f"Using device: {self.device}")
        
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
        """Load configuration for the model"""
        def getDir(obj):
            return os.path.dirname(os.path.abspath(obj))
        
        # Load from original config location or provided path
        if config_path:
            cfg = edict(yaml.safe_load(open(config_path)))
        else:
            config_base = os.path.join(getDir(__file__), "../original/MaskPLS/mask_pls/config")
            
            model_cfg = edict(yaml.safe_load(open(os.path.join(config_base, "model.yaml"))))
            backbone_cfg = edict(yaml.safe_load(open(os.path.join(config_base, "backbone.yaml"))))
            decoder_cfg = edict(yaml.safe_load(open(os.path.join(config_base, "decoder.yaml"))))
            
            cfg = edict({**model_cfg, **backbone_cfg, **decoder_cfg})
        
        # Set default parameters
        cfg.MODEL.DATASET = cfg.get('MODEL', {}).get('DATASET', 'KITTI')
        cfg.TRAIN.BATCH_SIZE = 1
        cfg.TRAIN.NUM_WORKERS = 4
        cfg.TRAIN.SUBSAMPLE = False
        cfg.TRAIN.AUG = False
        
        return cfg
    
    def load_model(self):
        """Load the MaskPLS model"""
        logger.info(f"Loading model from: {self.model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location='cpu')
        
        # Extract configuration and metadata
        dataset = self.cfg.MODEL.DATASET
        self.num_classes = checkpoint.get('num_classes', self.cfg[dataset].NUM_CLASSES)
        self.things_ids = checkpoint.get('things_ids', [1, 2, 3, 4, 5, 6, 7, 8])
        self.overlap_threshold = checkpoint.get('overlap_threshold', 0.8)
        
        # Create model components
        self.backbone = MinkEncoderDecoder(self.cfg.BACKBONE, self.cfg[dataset])
        self.backbone = ME.MinkowskiSyncBatchNorm.convert_sync_batchnorm(self.backbone)
        
        self.decoder = MaskedTransformerDecoder(
            self.cfg.DECODER,
            self.cfg.BACKBONE,
            self.cfg[dataset]
        )
        
        # Load weights
        if 'backbone_state_dict' in checkpoint:
            self.backbone.load_state_dict(checkpoint['backbone_state_dict'])
        if 'decoder_state_dict' in checkpoint:
            self.decoder.load_state_dict(checkpoint['decoder_state_dict'])
        
        # Move to device and set to eval mode
        self.backbone = self.backbone.to(self.device)
        self.decoder = self.decoder.to(self.device)
        self.backbone.eval()
        self.decoder.eval()
        
        logger.info(f"Model loaded successfully")
        logger.info(f"  Num classes: {self.num_classes}")
        logger.info(f"  Things IDs: {self.things_ids}")
        logger.info(f"  Device: {self.device}")
    
    def init_shared_memory(self):
        """Initialize shared memory segments"""
        # Maximum sizes
        self.max_points = 150000  # Maximum 150k points
        self.input_size = self.max_points * 4 * 4  # N x 4 floats (x,y,z,intensity)
        self.output_size = self.max_points * 2 * 4  # N x 2 ints (semantic, instance)
        self.control_size = 32  # Control structure
        
        try:
            # Try to create new shared memory segments
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
            
            logger.info("Created new shared memory segments")
            
        except FileExistsError:
            # Connect to existing segments
            logger.info("Connecting to existing shared memory segments")
            
            try:
                # Clean up existing segments first
                self.cleanup_shared_memory()
                
                # Create new ones
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
                
            except Exception as e:
                logger.error(f"Failed to create shared memory: {e}")
                raise
        
        # Create numpy arrays backed by shared memory
        self.control_array = np.ndarray(
            (8,), dtype=np.int32, buffer=self.control_shm.buf
        )
        self.control_array[:] = 0  # Initialize to zeros
        
        logger.info(f"Shared memory initialized:")
        logger.info(f"  Input size: {self.input_size / 1024 / 1024:.2f} MB")
        logger.info(f"  Output size: {self.output_size / 1024 / 1024:.2f} MB")
        logger.info(f"  Max points: {self.max_points}")
    
    def cleanup_shared_memory(self):
        """Clean up shared memory segments"""
        try:
            shm = shared_memory.SharedMemory(name=self.INPUT_SHM_NAME)
            shm.close()
            shm.unlink()
        except:
            pass
        
        try:
            shm = shared_memory.SharedMemory(name=self.OUTPUT_SHM_NAME)
            shm.close()
            shm.unlink()
        except:
            pass
        
        try:
            shm = shared_memory.SharedMemory(name=self.CONTROL_SHM_NAME)
            shm.close()
            shm.unlink()
        except:
            pass
    
    def signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.running = False
    
    def read_input_data(self, num_points):
        """Read input point cloud from shared memory"""
        # Create array view of shared memory
        input_array = np.ndarray(
            (num_points, 4),
            dtype=np.float32,
            buffer=self.input_shm.buf
        )
        
        # Copy data (important to avoid race conditions)
        points = input_array[:, :3].copy()
        features = input_array.copy()
        
        return points, features
    
    def write_output_data(self, semantic_pred, instance_pred):
        """Write predictions to shared memory"""
        num_points = len(semantic_pred)
        
        # Create array view of shared memory
        output_array = np.ndarray(
            (num_points, 2),
            dtype=np.int32,
            buffer=self.output_shm.buf
        )
        
        # Write predictions
        output_array[:, 0] = semantic_pred
        output_array[:, 1] = instance_pred
    
    @torch.no_grad()
    def process_point_cloud(self, points, features):
        """
        Run MaskPLS inference on point cloud
        
        Args:
            points: Nx3 numpy array of point coordinates
            features: Nx4 numpy array of point features
        
        Returns:
            semantic_pred: N-length array of semantic labels
            instance_pred: N-length array of instance IDs
        """
        try:
            # Convert to torch tensors
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
            
            # Run through backbone
            feats, coords, pad_masks, bb_logits = self.backbone(batch_dict)
            
            # Run through decoder
            outputs, padding = self.decoder(feats, coords, pad_masks)
            
            # Panoptic inference
            sem_pred, ins_pred = self.panoptic_inference(outputs, padding)
            
            # Convert to numpy
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
        """Panoptic segmentation inference (from original model)"""
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
        logger.info("Starting inference server loop...")
        logger.info("Waiting for requests from C++ node...")
        
        # Performance tracking
        inference_times = []
        frame_count = 0
        
        while self.running:
            try:
                # Check control flags
                flag = self.control_array[0]
                
                if flag == self.FLAG_NEW_DATA:
                    # New data available
                    num_points = self.control_array[1]
                    
                    if num_points > 0 and num_points <= self.max_points:
                        # Mark as processing
                        self.control_array[0] = self.FLAG_PROCESSING
                        
                        # Read input data
                        start_time = time.time()
                        points, features = self.read_input_data(num_points)
                        
                        # Process point cloud
                        semantic_pred, instance_pred = self.process_point_cloud(
                            points, features
                        )
                        
                        # Write output
                        self.write_output_data(semantic_pred, instance_pred)
                        
                        # Update timing stats
                        inference_time = (time.time() - start_time) * 1000  # ms
                        inference_times.append(inference_time)
                        frame_count += 1
                        
                        # Store timing info in control array
                        self.control_array[2] = int(inference_time)  # Current time
                        self.control_array[3] = frame_count  # Frame count
                        
                        # Mark as complete
                        self.control_array[0] = self.FLAG_COMPLETE
                        
                        # Log periodically
                        if frame_count % 10 == 0:
                            avg_time = np.mean(inference_times[-10:])
                            logger.info(f"Processed {frame_count} frames, "
                                      f"Last: {inference_time:.1f}ms, "
                                      f"Avg(10): {avg_time:.1f}ms")
                    else:
                        logger.error(f"Invalid number of points: {num_points}")
                        self.control_array[0] = self.FLAG_ERROR
                
                elif flag == self.FLAG_SHUTDOWN:
                    logger.info("Received shutdown flag")
                    break
                
                # Small sleep to prevent CPU spinning
                time.sleep(0.0001)  # 0.1ms
                
            except KeyboardInterrupt:
                logger.info("Keyboard interrupt received")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                self.control_array[0] = self.FLAG_ERROR
                import traceback
                traceback.print_exc()
        
        # Cleanup
        logger.info("Shutting down inference server...")
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Signal shutdown to C++ node
            self.control_array[0] = self.FLAG_SHUTDOWN
            
            # Close and unlink shared memory
            self.input_shm.close()
            self.output_shm.close()
            self.control_shm.close()
            
            self.input_shm.unlink()
            self.output_shm.unlink()
            self.control_shm.unlink()
            
            logger.info("Shared memory cleaned up")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


def main():
    parser = argparse.ArgumentParser(description='MaskPLS Inference Server')
    parser.add_argument('--model', required=True, help='Path to model file')
    parser.add_argument('--config', help='Path to config file (optional)')
    parser.add_argument('--cuda', action='store_true', help='Use CUDA if available')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Create and run server
    server = SharedMemoryInferenceServer(
        model_path=args.model,
        config_path=args.config,
        use_cuda=args.cuda
    )
    
    try:
        server.run()
    except Exception as e:
        logger.error(f"Server error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        server.cleanup()


if __name__ == "__main__":
    main()
