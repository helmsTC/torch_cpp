#!/usr/bin/env python3
"""
Semantic-KITTI Dataset Analysis Tool
Analyzes semantic and instance label distributions in your dataset
"""

import numpy as np
import yaml
import os
from pathlib import Path
from collections import defaultdict
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

class SemanticKittiAnalyzer:
    def __init__(self, dataset_path, config_path):
        self.dataset_path = Path(dataset_path)
        self.config_path = config_path
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.learning_map = self.config['learning_map']
        self.learning_map_inv = self.config['learning_map_inv']
        self.class_names = self.config['labels']
        self.learning_ignore = self.config.get('learning_ignore', {})
        
        # Determine things vs stuff
        self.things = self._get_things()
        self.stuff = self._get_stuff()
        
        # Initialize statistics
        self.stats = {
            'total_points': 0,
            'total_frames': 0,
            'frames_with_instances': 0,
            'frames_without_instances': 0,
            'class_point_counts': defaultdict(int),
            'class_frame_counts': defaultdict(int),
            'instance_counts': defaultdict(int),
            'points_per_instance': [],
            'instances_per_frame': [],
            'instances_per_class': defaultdict(list),
            'empty_frames': 0,
            'ignored_points': 0
        }
        
        # Create lookup table for remapping
        maxkey = max(self.learning_map.keys())
        self.lut = np.zeros((maxkey + 100), dtype=np.int32)
        self.lut[list(self.learning_map.keys())] = list(self.learning_map.values())
        
    def _get_things(self):
        """Identify 'thing' classes (instances)"""
        # Common thing classes for KITTI
        things_names = [
            'car', 'truck', 'bicycle', 'motorcycle', 'other-vehicle',
            'person', 'bicyclist', 'motorcyclist', 'bus', 'trailer',
            'construction_vehicle', 'pedestrian'
        ]
        things = []
        for name in things_names:
            if name in self.class_names.values():
                for k, v in self.class_names.items():
                    if v == name:
                        things.append(name)
                        break
        return things
    
    def _get_stuff(self):
        """Identify 'stuff' classes (non-instances)"""
        stuff = []
        for class_id, name in self.class_names.items():
            if name not in self.things and name != 'unlabeled':
                stuff.append(name)
        return stuff
    
    def analyze_sequence(self, sequence_path):
        """Analyze a single sequence"""
        labels_path = sequence_path / 'labels'
        
        if not labels_path.exists():
            print(f"Warning: Labels not found for {sequence_path}")
            return
        
        label_files = sorted(labels_path.glob('*.label'))
        
        for label_file in tqdm(label_files, desc=f"Processing {sequence_path.name}", leave=False):
            self.analyze_frame(label_file)
    
    def analyze_frame(self, label_file):
        """Analyze a single frame"""
        # Load label file
        label = np.fromfile(label_file, dtype=np.uint32)
        label = label.reshape((-1))
        
        if len(label) == 0:
            self.stats['empty_frames'] += 1
            return
        
        # Split semantic and instance labels
        sem_label = label & 0xFFFF  # Lower 16 bits
        inst_label = label >> 16     # Upper 16 bits
        
        # Remap semantic labels
        sem_label_mapped = self.lut[sem_label]
        
        # Update statistics
        self.stats['total_frames'] += 1
        self.stats['total_points'] += len(sem_label)
        
        # Count points per class
        unique_classes, counts = np.unique(sem_label_mapped, return_counts=True)
        for cls, cnt in zip(unique_classes, counts):
            if cls in self.learning_ignore and self.learning_ignore[cls]:
                self.stats['ignored_points'] += cnt
            else:
                self.stats['class_point_counts'][cls] += cnt
                self.stats['class_frame_counts'][cls] += 1
        
        # Analyze instances
        instances_in_frame = 0
        for cls in unique_classes:
            if cls == 0:  # Skip unlabeled
                continue
                
            # Check if this is a thing class
            class_name = self.class_names.get(self.learning_map_inv.get(cls, 0), 'unknown')
            if class_name in self.things:
                # Get instances for this semantic class
                mask = sem_label_mapped == cls
                unique_instances = np.unique(inst_label[mask])
                unique_instances = unique_instances[unique_instances > 0]  # Remove 0 (no instance)
                
                instances_in_frame += len(unique_instances)
                self.stats['instance_counts'][cls] += len(unique_instances)
                self.stats['instances_per_class'][cls].append(len(unique_instances))
                
                # Count points per instance
                for inst_id in unique_instances:
                    inst_mask = (sem_label_mapped == cls) & (inst_label == inst_id)
                    self.stats['points_per_instance'].append(np.sum(inst_mask))
        
        # Track frames with/without instances
        if instances_in_frame > 0:
            self.stats['frames_with_instances'] += 1
            self.stats['instances_per_frame'].append(instances_in_frame)
        else:
            self.stats['frames_without_instances'] += 1
            self.stats['instances_per_frame'].append(0)
    
    def analyze_dataset(self):
        """Analyze the entire dataset"""
        # Find all sequences
        sequences = []
        
        # Check for standard KITTI structure
        for split in ['train', 'valid', 'test']:
            split_path = self.dataset_path / 'dataset' / 'sequences'
            if split_path.exists():
                sequences.extend([d for d in split_path.iterdir() if d.is_dir()])
                break
        
        # Alternative: sequences directly in dataset_path
        if not sequences:
            sequences = [d for d in self.dataset_path.iterdir() 
                        if d.is_dir() and d.name.isdigit()]
        
        if not sequences:
            print(f"No sequences found in {self.dataset_path}")
            return
        
        print(f"Found {len(sequences)} sequences")
        
        for sequence in tqdm(sequences, desc="Analyzing sequences"):
            self.analyze_sequence(sequence)
    
    def generate_report(self):
        """Generate analysis report"""
        print("\n" + "="*80)
        print("SEMANTIC-KITTI DATASET ANALYSIS REPORT")
        print("="*80)
        
        print(f"\nConfiguration: {self.config_path}")
        print(f"Dataset Path: {self.dataset_path}")
        
        # General statistics
        print("\n" + "-"*40)
        print("GENERAL STATISTICS")
        print("-"*40)
        print(f"Total frames: {self.stats['total_frames']:,}")
        print(f"Total points: {self.stats['total_points']:,}")
        print(f"Empty frames: {self.stats['empty_frames']:,}")
        print(f"Ignored points: {self.stats['ignored_points']:,}")
        print(f"Average points per frame: {self.stats['total_points'] / max(1, self.stats['total_frames']):,.0f}")
        
        # Instance statistics
        print("\n" + "-"*40)
        print("INSTANCE STATISTICS")
        print("-"*40)
        print(f"Frames with instances: {self.stats['frames_with_instances']:,} "
              f"({100 * self.stats['frames_with_instances'] / max(1, self.stats['total_frames']):.1f}%)")
        print(f"Frames without instances: {self.stats['frames_without_instances']:,} "
              f"({100 * self.stats['frames_without_instances'] / max(1, self.stats['total_frames']):.1f}%)")
        
        if self.stats['instances_per_frame']:
            print(f"Average instances per frame: {np.mean(self.stats['instances_per_frame']):.2f}")
            print(f"Max instances in a frame: {np.max(self.stats['instances_per_frame'])}")
            print(f"Frames with >10 instances: "
                  f"{sum(1 for x in self.stats['instances_per_frame'] if x > 10)}")
        
        if self.stats['points_per_instance']:
            print(f"Average points per instance: {np.mean(self.stats['points_per_instance']):.0f}")
            print(f"Min points per instance: {np.min(self.stats['points_per_instance'])}")
            print(f"Max points per instance: {np.max(self.stats['points_per_instance'])}")
            print(f"Instances with <50 points: "
                  f"{sum(1 for x in self.stats['points_per_instance'] if x < 50)}")
        
        # Class distribution
        print("\n" + "-"*40)
        print("CLASS DISTRIBUTION")
        print("-"*40)
        print(f"{'Class':<20} {'Type':<8} {'Points':<15} {'Frames':<10} {'Instances':<12} {'Avg Inst/Frame':<15}")
        print("-"*80)
        
        for cls_id in sorted(self.stats['class_point_counts'].keys()):
            class_name = self.class_names.get(self.learning_map_inv.get(cls_id, 0), 'unknown')
            if class_name == 'unlabeled':
                continue
                
            cls_type = 'thing' if class_name in self.things else 'stuff'
            points = self.stats['class_point_counts'][cls_id]
            frames = self.stats['class_frame_counts'][cls_id]
            instances = self.stats['instance_counts'].get(cls_id, 0)
            
            avg_inst = 0
            if cls_id in self.stats['instances_per_class'] and self.stats['instances_per_class'][cls_id]:
                avg_inst = np.mean(self.stats['instances_per_class'][cls_id])
            
            print(f"{class_name:<20} {cls_type:<8} {points:<15,} {frames:<10,} {instances:<12,} {avg_inst:<15.2f}")
        
        # Things vs Stuff summary
        print("\n" + "-"*40)
        print("THINGS vs STUFF SUMMARY")
        print("-"*40)
        
        thing_points = sum(self.stats['class_point_counts'][cls] 
                          for cls in self.stats['class_point_counts']
                          if self.class_names.get(self.learning_map_inv.get(cls, 0), '') in self.things)
        stuff_points = sum(self.stats['class_point_counts'][cls] 
                          for cls in self.stats['class_point_counts']
                          if self.class_names.get(self.learning_map_inv.get(cls, 0), '') in self.stuff)
        
        total_labeled = thing_points + stuff_points
        if total_labeled > 0:
            print(f"Thing points: {thing_points:,} ({100*thing_points/total_labeled:.1f}%)")
            print(f"Stuff points: {stuff_points:,} ({100*stuff_points/total_labeled:.1f}%)")
        
        print(f"\nThing classes: {', '.join(self.things)}")
        print(f"Stuff classes: {', '.join(self.stuff)}")
        
        # Warnings and recommendations
        print("\n" + "-"*40)
        print("WARNINGS & RECOMMENDATIONS")
        print("-"*40)
        
        if self.stats['frames_without_instances'] > 0.5 * self.stats['total_frames']:
            print("⚠️  More than 50% of frames have no instances - consider:")
            print("   - Checking if instance labels are correctly loaded")
            print("   - Verifying the learning_map configuration")
            print("   - Using balanced sampling during training")
        
        if self.stats['empty_frames'] > 0:
            print(f"⚠️  Found {self.stats['empty_frames']} empty frames - these may cause training issues")
        
        small_instances = sum(1 for x in self.stats['points_per_instance'] if x < 50)
        if small_instances > 0.1 * len(self.stats['points_per_instance']):
            print(f"⚠️  {small_instances} instances have <50 points - consider:")
            print("   - Filtering small instances during training")
            print("   - Adjusting min_points threshold in evaluator")
        
        # Check for missing classes
        expected_classes = set(self.learning_map_inv.keys()) - {0}  # Exclude unlabeled
        found_classes = set(self.stats['class_point_counts'].keys())
        missing_classes = expected_classes - found_classes
        
        if missing_classes:
            print(f"⚠️  Classes defined but not found in data: {missing_classes}")
            for cls_id in missing_classes:
                class_name = self.class_names.get(self.learning_map_inv.get(cls_id, 0), 'unknown')
                print(f"   - {class_name} (ID: {cls_id})")
    
    def plot_distributions(self, save_path=None):
        """Create visualization plots"""
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('Semantic-KITTI Dataset Distribution Analysis', fontsize=16)
        
        # 1. Class point distribution
        ax = axes[0, 0]
        class_names = []
        point_counts = []
        for cls_id in sorted(self.stats['class_point_counts'].keys()):
            name = self.class_names.get(self.learning_map_inv.get(cls_id, 0), 'unknown')
            if name != 'unlabeled':
                class_names.append(name[:10])  # Truncate long names
                point_counts.append(self.stats['class_point_counts'][cls_id])
        
        ax.bar(range(len(class_names)), point_counts)
        ax.set_xticks(range(len(class_names)))
        ax.set_xticklabels(class_names, rotation=45, ha='right')
        ax.set_ylabel('Point Count')
        ax.set_title('Points per Class')
        ax.set_yscale('log')
        
        # 2. Instances per frame histogram
        ax = axes[0, 1]
        if self.stats['instances_per_frame']:
            ax.hist(self.stats['instances_per_frame'], bins=30, edgecolor='black')
            ax.set_xlabel('Instances per Frame')
            ax.set_ylabel('Frame Count')
            ax.set_title('Instance Distribution per Frame')
        
        # 3. Points per instance histogram
        ax = axes[0, 2]
        if self.stats['points_per_instance']:
            ax.hist(self.stats['points_per_instance'], bins=50, edgecolor='black')
            ax.set_xlabel('Points per Instance')
            ax.set_ylabel('Instance Count')
            ax.set_title('Point Distribution per Instance')
            ax.set_yscale('log')
        
        # 4. Thing vs Stuff pie chart
        ax = axes[1, 0]
        thing_points = sum(self.stats['class_point_counts'][cls] 
                          for cls in self.stats['class_point_counts']
                          if self.class_names.get(self.learning_map_inv.get(cls, 0), '') in self.things)
        stuff_points = sum(self.stats['class_point_counts'][cls] 
                          for cls in self.stats['class_point_counts']
                          if self.class_names.get(self.learning_map_inv.get(cls, 0), '') in self.stuff)
        
        if thing_points > 0 or stuff_points > 0:
            ax.pie([thing_points, stuff_points], labels=['Things', 'Stuff'], 
                   autopct='%1.1f%%', startangle=90)
            ax.set_title('Things vs Stuff Distribution')
        
        # 5. Frames with/without instances
        ax = axes[1, 1]
        ax.bar(['With Instances', 'Without Instances'], 
               [self.stats['frames_with_instances'], self.stats['frames_without_instances']])
        ax.set_ylabel('Frame Count')
        ax.set_title('Frames with/without Instances')
        
        # 6. Instance count per thing class
        ax = axes[1, 2]
        thing_classes = []
        instance_counts = []
        for cls_id in sorted(self.stats['instance_counts'].keys()):
            name = self.class_names.get(self.learning_map_inv.get(cls_id, 0), 'unknown')
            if name in self.things:
                thing_classes.append(name[:10])
                instance_counts.append(self.stats['instance_counts'][cls_id])
        
        if thing_classes:
            ax.bar(range(len(thing_classes)), instance_counts)
            ax.set_xticks(range(len(thing_classes)))
            ax.set_xticklabels(thing_classes, rotation=45, ha='right')
            ax.set_ylabel('Instance Count')
            ax.set_title('Total Instances per Thing Class')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plots saved to {save_path}")
        else:
            plt.show()


def main():
    parser = argparse.ArgumentParser(description='Analyze Semantic-KITTI dataset')
    parser.add_argument('dataset_path', type=str, help='Path to Semantic-KITTI dataset')
    parser.add_argument('config_path', type=str, help='Path to semantic-kitti-custom.yaml')
    parser.add_argument('--plot', action='store_true', help='Generate distribution plots')
    parser.add_argument('--save-plot', type=str, help='Path to save plots')
    
    args = parser.parse_args()
    
    analyzer = SemanticKittiAnalyzer(args.dataset_path, args.config_path)
    
    print("Starting dataset analysis...")
    analyzer.analyze_dataset()
    
    print("\nGenerating report...")
    analyzer.generate_report()
    
    if args.plot or args.save_plot:
        print("\nGenerating plots...")
        analyzer.plot_distributions(args.save_plot)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
