#!/usr/bin/env python3
"""
Script to visualize saved semantic point clouds
Supports NPZ, PCD, and PLY formats
"""

import numpy as np
import open3d as o3d
import argparse
from pathlib import Path
import glob

def load_npz_cloud(filepath):
    """Load cloud from NPZ file"""
    data = np.load(filepath)
    
    points = data['points']
    semantic_labels = data.get('semantic_labels', None)
    instance_labels = data.get('instance_labels', None)
    colors = data.get('colors', None)
    
    return points, semantic_labels, instance_labels, colors

def create_colored_point_cloud(points, semantic_labels=None, instance_labels=None, colors=None):
    """Create Open3D point cloud with colors"""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    
    if colors is not None:
        # Use provided colors
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)
    elif semantic_labels is not None:
        # Generate colors from semantic labels
        colors = semantic_to_colors(semantic_labels)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Default gray color
        colors = np.ones((points.shape[0], 3)) * 0.5
        pcd.colors = o3d.utility.Vector3dVector(colors)
    
    return pcd

def semantic_to_colors(labels):
    """Convert semantic labels to colors using SemanticKITTI color scheme"""
    # SemanticKITTI color map (RGB normalized)
    color_map = {
        0: [0.0, 0.0, 0.0],          # unlabeled - black
        1: [0.39, 0.59, 0.96],       # car - blue
        2: [0.96, 0.90, 0.39],       # bicycle - yellow
        3: [0.12, 0.24, 0.59],       # motorcycle - dark blue
        4: [0.31, 0.12, 0.71],       # truck - purple
        5: [0.0, 0.0, 1.0],          # other-vehicle - bright blue
        6: [1.0, 0.12, 0.12],        # person - red
        7: [1.0, 0.16, 0.78],        # bicyclist - pink
        8: [0.59, 0.12, 0.35],       # motorcyclist - maroon
        9: [1.0, 0.0, 1.0],          # road - magenta
        10: [1.0, 0.59, 1.0],        # parking - light magenta
        11: [0.29, 0.0, 0.29],       # sidewalk - dark purple
        12: [0.29, 0.0, 0.69],       # other-ground - blue-purple
        13: [1.0, 0.78, 0.0],        # building - orange
        14: [0.98, 0.47, 0.20],      # fence - orange-red
        15: [0.0, 0.69, 0.0],        # vegetation - green
        16: [0.53, 0.24, 0.0],       # trunk - brown
        17: [0.59, 0.94, 0.31],      # terrain - light green
        18: [1.0, 0.94, 0.59],       # pole - light yellow
        19: [1.0, 0.0, 0.0],         # traffic-sign - red
    }
    
    colors = np.zeros((labels.shape[0], 3))
    for i, label in enumerate(labels):
        if label in color_map:
            colors[i] = color_map[label]
        else:
            # Generate color for unknown labels
            hue = (label * 0.618033988749895) % 1.0
            colors[i] = hsv_to_rgb(hue, 0.8, 0.9)
    
    return colors

def instance_to_colors(labels):
    """Generate distinct colors for instance IDs"""
    colors = np.zeros((labels.shape[0], 3))
    unique_instances = np.unique(labels)
    
    for inst_id in unique_instances:
        if inst_id == 0:
            continue  # Skip background
        
        # Generate color using golden ratio
        hue = (inst_id * 0.618033988749895) % 1.0
        color = hsv_to_rgb(hue, 0.9, 0.9)
        colors[labels == inst_id] = color
    
    return colors

def hsv_to_rgb(h, s, v):
    """Convert HSV to RGB"""
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)
    
    if h_i == 0:
        return [v, t, p]
    elif h_i == 1:
        return [q, v, p]
    elif h_i == 2:
        return [p, v, t]
    elif h_i == 3:
        return [p, q, v]
    elif h_i == 4:
        return [t, p, v]
    else:
        return [v, p, q]

def visualize_single_cloud(filepath, show_instances=False):
    """Visualize a single point cloud file"""
    print(f"Loading: {filepath}")
    
    if filepath.endswith('.npz'):
        points, semantic_labels, instance_labels, colors = load_npz_cloud(filepath)
        
        if show_instances and instance_labels is not None:
            print(f"Showing instance segmentation")
            colors = instance_to_colors(instance_labels)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            print(f"Showing semantic segmentation")
            pcd = create_colored_point_cloud(points, semantic_labels, instance_labels, colors)
    
    elif filepath.endswith('.pcd') or filepath.endswith('.ply'):
        pcd = o3d.io.read_point_cloud(filepath)
    else:
        print(f"Unsupported format: {filepath}")
        return
    
    # Print statistics
    print(f"Points: {len(pcd.points)}")
    
    if filepath.endswith('.npz'):
        if semantic_labels is not None:
            unique_classes = np.unique(semantic_labels)
            print(f"Semantic classes: {len(unique_classes)} - {unique_classes}")
        if instance_labels is not None:
            unique_instances = np.unique(instance_labels)
            print(f"Instances: {len(unique_instances)} (max ID: {unique_instances.max()})")
    
    # Create visualization window
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud: {Path(filepath).name}")
    vis.add_geometry(pcd)
    
    # Set render options
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    # Set view
    view_ctl = vis.get_view_control()
    view_ctl.set_zoom(0.5)
    
    print("\nControls:")
    print("  Mouse: Rotate/Pan/Zoom")
    print("  '+'/'-': Increase/decrease point size")
    print("  'R': Reset view")
    print("  'Q': Close window")
    
    vis.run()
    vis.destroy_window()

def visualize_sequence(directory, pattern="*.npz", show_instances=False, delay=100):
    """Visualize a sequence of point clouds"""
    files = sorted(glob.glob(str(Path(directory) / pattern)))
    
    if not files:
        print(f"No files found matching {pattern} in {directory}")
        return
    
    print(f"Found {len(files)} files")
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="Point Cloud Sequence")
    
    # Load first cloud
    if files[0].endswith('.npz'):
        points, semantic_labels, instance_labels, colors = load_npz_cloud(files[0])
        if show_instances and instance_labels is not None:
            colors = instance_to_colors(instance_labels)
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            pcd = create_colored_point_cloud(points, semantic_labels, instance_labels, colors)
    else:
        pcd = o3d.io.read_point_cloud(files[0])
    
    vis.add_geometry(pcd)
    
    # Set render options
    opt = vis.get_render_option()
    opt.point_size = 2.0
    opt.background_color = np.array([0.1, 0.1, 0.1])
    
    print("\nPlaying sequence...")
    print("Press 'Q' to stop")
    
    for i, filepath in enumerate(files[1:], 1):
        print(f"\rFrame {i+1}/{len(files)}", end="")
        
        # Load next cloud
        if filepath.endswith('.npz'):
            points, semantic_labels, instance_labels, colors = load_npz_cloud(filepath)
            if show_instances and instance_labels is not None:
                colors = instance_to_colors(instance_labels)
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
            else:
                pcd.points = o3d.utility.Vector3dVector(points)
                if semantic_labels is not None:
                    colors = semantic_to_colors(semantic_labels)
                    pcd.colors = o3d.utility.Vector3dVector(colors)
        else:
            new_pcd = o3d.io.read_point_cloud(filepath)
            pcd.points = new_pcd.points
            pcd.colors = new_pcd.colors
        
        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        
        # Small delay between frames
        import time
        time.sleep(delay / 1000.0)
        
        if not vis.poll_events():
            break
    
    print("\nDone")
    vis.destroy_window()

def main():
    parser = argparse.ArgumentParser(description='Visualize saved point clouds')
    parser.add_argument('input', help='Input file or directory')
    parser.add_argument('--sequence', action='store_true', 
                       help='Play as sequence (for directories)')
    parser.add_argument('--pattern', default='*.pcd',
                       help='File pattern for sequence mode (default: *.npz)')
    parser.add_argument('--instances', action='store_true',
                       help='Show instance segmentation instead of semantic')
    parser.add_argument('--delay', type=int, default=100,
                       help='Delay between frames in ms (sequence mode)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        visualize_single_cloud(str(input_path), args.instances)
    elif input_path.is_dir():
        if args.sequence:
            visualize_sequence(str(input_path), args.pattern, args.instances, args.delay)
        else:
            # Visualize first file in directory
            files = sorted(glob.glob(str(input_path / args.pattern)))
            if files:
                visualize_single_cloud(files[0], args.instances)
            else:
                print(f"No files found matching {args.pattern}")
    else:
        print(f"Input path not found: {input_path}")

if __name__ == '__main__':
    main()
