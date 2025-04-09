""" Main script for loading and visualizing KITTI object detection data.

Allows visualizing images with 2D/3D boxes, LiDAR point clouds with 3D boxes (GT and predictions),
and projecting LiDAR onto images or BEV maps.

Run from XPillars/tools:
python kitti_vis/kitti_object.py --dir ../data/kitti --vis --show_lidar_with_boxes --show_image_with_boxes --ind 0 [--pred]
"""
from __future__ import print_function

import os
import sys
import numpy as np
import cv2
import argparse
from PIL import Image # Used potentially for viewing images if cv2 fails

from . import kitti_util as utils
from . import viz_util

TARGET_CLASSES = ["Car", "Pedestrian", "Cyclist"]   # classes to visualize

# --- Dataset Loading Class ---
class KittiObjectDataset(object):
    """ Loads KITTI object detection dataset samples (images, lidar, calibration, labels). """

    def __init__(self, root_dir, split='training', args=None):
        """
        Args:
            root_dir (str): Path to the root KITTI dataset directory (e.g., containing 'training', 'testing').
            split (str): Which split to load ('training' or 'testing').
            args (argparse.Namespace, optional): Command-line arguments to override default subdirs.
        """
        self.root_dir = root_dir
        self.split = split
        self.split_dir = os.path.join(self.root_dir, self.split)

        if not os.path.isdir(self.split_dir):
             raise FileNotFoundError(f"Dataset split directory not found: {self.split_dir}")

        # Determine number of samples
        # This relies on the presence of image_2 files to count
        self.image_dir = os.path.join(self.split_dir, "image_2")
        if not os.path.isdir(self.image_dir):
             raise FileNotFoundError(f"Image directory not found: {self.image_dir}")

        try:
            self.num_samples = len([name for name in os.listdir(self.image_dir) if name.endswith('.png')])
            if self.num_samples == 0:
                 print(f"Warning: No samples found in {self.image_dir}")

        except FileNotFoundError:
            print(f"Error: Image directory not found or inaccessible: {self.image_dir}")
            self.num_samples = 0 # Or handle error differently

        # Define standard subdirectories
        lidar_subdir = "velodyne"
        depth_subdir = "depth_2" # Commonly named depth_2 or similar for stereo depth
        pred_subdir = "pred"     # Default name for prediction labels

        # Override subdirs if provided in args
        if args is not None:
            lidar_subdir = args.lidar if hasattr(args, 'lidar') else lidar_subdir
            depth_subdir = args.depthdir if hasattr(args, 'depthdir') else depth_subdir
            pred_subdir = args.preddir if hasattr(args, 'preddir') else pred_subdir

        # Construct full paths
        self.label_dir = os.path.join(self.split_dir, "label_2")
        self.calib_dir = os.path.join(self.split_dir, "calib")
        self.lidar_dir = os.path.join(self.split_dir, lidar_subdir)
        self.depth_dir = os.path.join(self.split_dir, depth_subdir)
        self.pred_dir = os.path.join(self.split_dir, pred_subdir)
        # Optional: Depth Point Cloud directory (if pre-generated)
        self.depthpc_dir = os.path.join(self.split_dir, "depth_pc")

        # Pre-check essential directories
        if not os.path.isdir(self.calib_dir):
            print(f"Warning: Calibration directory not found: {self.calib_dir}")
        if not os.path.isdir(self.lidar_dir):
            print(f"Warning: LiDAR directory not found: {self.lidar_dir}")
        if split == 'training' and not os.path.isdir(self.label_dir):
            print(f"Warning: Label directory not found: {self.label_dir}")


    def __len__(self):
        return self.num_samples

    def get_image(self, idx):
        """ Loads the image for the given index. """
        if idx >= self.num_samples: raise IndexError("Index out of bounds")
        img_filename = os.path.join(self.image_dir, "%06d.png" % idx)
        if not os.path.exists(img_filename):
            print(f"Warning: Image file not found {img_filename}")
            return None # Or return a placeholder black image?
        return utils.load_image(img_filename)

    def get_lidar(self, idx, dtype=np.float32, n_vec=4):
        """ Loads the LiDAR point cloud for the given index. """
        if idx >= self.num_samples: raise IndexError("Index out of bounds")
        lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % idx)
        return utils.load_velo_scan(lidar_filename, dtype, n_vec)

    def get_calibration(self, idx):
        """ Loads the calibration data for the given index. """
        if idx >= self.num_samples: raise IndexError("Index out of bounds")
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % idx)
        if not os.path.exists(calib_filename):
            print(f"Warning: Calibration file not found {calib_filename}")
            return None
        return utils.Calibration(calib_filename)

    def get_label_objects(self, idx):
        """ Loads the ground truth label objects for the given index (training split only). """
        if idx >= self.num_samples: raise IndexError("Index out of bounds")
        if self.split != 'training':
            # print("Warning: Labels are typically only available for the 'training' split.")
            return [] # Return empty list for non-training splits

        label_filename = os.path.join(self.label_dir, "%06d.txt" % idx)
        if not os.path.exists(label_filename):
            # print(f"Warning: Label file not found {label_filename}")
            return [] # Return empty list if no label file
        return utils.read_label(label_filename)

    def get_pred_objects(self, idx):
        """ Loads the predicted label objects for the given index. """
        if idx >= self.num_samples: raise IndexError("Index out of bounds")
        pred_filename = os.path.join(self.pred_dir, "%06d.txt" % idx)
        if not os.path.exists(pred_filename):
            # It's normal for prediction files to not exist for all frames
            return None
        try:
            # Assuming predictions use the same format as GT labels
            return utils.read_label(pred_filename)
        except Exception as e:
            print(f"Error reading prediction file {pred_filename}: {e}")
            return None # Return None if file format is incorrect

    def get_depth(self, idx):
        """ Loads the depth map for the given index. """
        if idx >= self.num_samples: raise IndexError("Index out of bounds")
        # Try standard KITTI depth naming convention first
        depth_filename = os.path.join(self.depth_dir, "%06d.png" % idx)
        if not os.path.exists(depth_filename):
             # Fallback to potentially different naming or location if needed
             # print(f"Warning: Depth file not found {depth_filename}")
             return None, False
        return utils.load_depth(depth_filename) # Returns (depth_map, is_exist)

    def get_depth_pc(self, idx):
        """ Loads pre-generated depth point cloud if available. """
        if idx >= self.num_samples: raise IndexError("Index out of bounds")
        depthpc_filename = os.path.join(self.depthpc_dir, "%06d.bin" % idx)
        if not os.path.exists(depthpc_filename):
            return None, False
        # Assuming depth pc is stored like velo scan (e.g., x,y,z,intensity)
        # Adjust dtype and n_vec if the format is different
        return utils.load_velo_scan(depthpc_filename, dtype=np.float32, n_vec=4), True


# --- Visualization Functions ---

def show_image_with_boxes(img, objects_gt, calib, objects_pred=None, show_3d=True, show_2d_gt=True, show_2d_pred=False):
    """ Shows image with 2D and/or 3D bounding boxes, filtering for TARGET_CLASSES. """
    if img is None:
        print("Error: Cannot display boxes, image is None.")
        return None, None

    img_disp = np.copy(img)
    img_3d = np.copy(img) if show_3d else None

    # --- Define OpenCV BGR Colors ---
    CV2_BOX_COLOR_MAP = { # Use BGR for OpenCV
        "Car": (0, 255, 0),        # Green
        "Pedestrian": (255, 255, 0), # Cyan
        "Cyclist": (0, 255, 255),   # Yellow
        "Van": (128, 128, 0),      # Teal-like
        "Truck": (128, 0, 128),    # Purple-like
        "Misc": (200, 200, 200),   # Light Gray
        "Tram": (0, 128, 255),     # Orange-like
        "Person_sitting": (255, 0, 255), # Magenta
        "DontCare": (128, 128, 128) # Gray
    }
    CV2_PREDICTION_COLOR = (0, 0, 255) # Red (BGR)
    CV2_DEFAULT_COLOR = (255, 255, 255) # White (BGR)
    # --- END Define OpenCV BGR Colors ---


    # Draw GT Boxes
    for obj in objects_gt:
        # +++ Filter GT objects +++
        if obj.type not in TARGET_CLASSES:
            continue
        # +++++++++++++++++++++++++

        color_cv2 = CV2_BOX_COLOR_MAP.get(obj.type, CV2_DEFAULT_COLOR) # Use CV2 specific colors

        # Draw 2D GT Box
        if show_2d_gt:
            cv2.rectangle(img_disp, (int(obj.xmin), int(obj.ymin)),
                          (int(obj.xmax), int(obj.ymax)), color_cv2, 2)

        # Compute and Draw 3D GT Box
        if show_3d and img_3d is not None:
            box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P2)
            if box3d_pts_2d is not None:
                # Pass the CV2 color to the drawing util
                img_3d = utils.draw_projected_box3d(img_3d, box3d_pts_2d, color=color_cv2, thickness=2)

    # Draw Predicted Boxes
    if objects_pred is not None:
        for obj in objects_pred:
            # +++ Filter Predicted objects +++
            if obj.type not in TARGET_CLASSES:
                 continue
            # ++++++++++++++++++++++++++++++

            # Draw 2D Predicted Box
            if show_2d_pred:
                 cv2.rectangle(img_disp, (int(obj.xmin), int(obj.ymin)),
                               (int(obj.xmax), int(obj.ymax)), CV2_PREDICTION_COLOR, 1) # Thinner line for preds

            # Compute and Draw 3D Predicted Box
            if show_3d and img_3d is not None:
                box3d_pts_2d, _ = utils.compute_box_3d(obj, calib.P2)
                if box3d_pts_2d is not None:
                    img_3d = utils.draw_projected_box3d(img_3d, box3d_pts_2d, color=CV2_PREDICTION_COLOR, thickness=1)

    # Display the images
    if show_2d_gt or show_2d_pred:
        cv2.imshow("2D Boxes", img_disp)
    if show_3d and img_3d is not None:
        cv2.imshow("3D Boxes", img_3d)

    return img_disp, img_3d


def get_lidar_in_image_fov(pc_velo, calib, img_width, img_height, clip_distance=2.0):
    """ Filters LiDAR points to keep only those potentially visible in the image FOV. """
    if pc_velo is None or calib is None: return None, None

    # Project points to image coordinates
    pts_2d = calib.project_velo_to_image(pc_velo[:, 0:3])

    # Basic FOV check (inside image bounds)
    fov_inds = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < img_width) & \
               (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < img_height)

    # Distance check (remove points too close to the sensor)
    fov_inds = fov_inds & (pc_velo[:, 0] > clip_distance) # Check X distance in velo coords

    return pc_velo[fov_inds, :], pts_2d[fov_inds, :]


def show_lidar_on_image(pc_velo, img, calib):
    """ Projects LiDAR points onto an image and visualizes them with color-coded depth. """
    if img is None or pc_velo is None or calib is None:
        print("Error: Cannot project lidar onto image, missing data.")
        return None

    img_lidar_proj = np.copy(img)
    img_height, img_width, _ = img.shape

    # Filter points to FOV and get their 2D coordinates
    imgfov_pc_velo, imgfov_pts_2d = get_lidar_in_image_fov(pc_velo, calib, img_width, img_height)

    if imgfov_pc_velo is None or imgfov_pc_velo.shape[0] == 0:
        print("No LiDAR points found in image FOV.")
        cv2.imshow("LiDAR Projection", img_lidar_proj)
        return img_lidar_proj

    # Get depth (distance for coloring) - use velo X-coord (forward distance)
    depths = imgfov_pc_velo[:, 0]

    # Create a colormap (e.g., viridis, jet)
    import matplotlib.pyplot as plt
    cmap = plt.cm.get_cmap('jet') # Or 'viridis', 'hsv' etc.
    max_depth = np.max(depths) if len(depths) > 0 else 70.0 # Avoid issues with empty depths
    min_depth = np.min(depths) if len(depths) > 0 else 0.0
    # Normalize depths for colormap
    norm_depths = (depths - min_depth) / (max_depth - min_depth) if (max_depth - min_depth) > 0 else np.zeros_like(depths)
    colors = (cmap(norm_depths)[:, :3] * 255).astype(np.uint8) # Get RGB colors (0-255)

    # Draw circles on the image
    for i in range(imgfov_pts_2d.shape[0]):
        pt = (int(np.round(imgfov_pts_2d[i, 0])), int(np.round(imgfov_pts_2d[i, 1])))
        color_tuple = tuple(c.item() for c in colors[i]) # Convert np array row to tuple
        cv2.circle(img_lidar_proj, pt, radius=2, color=color_tuple, thickness=-1) # Filled circle

    cv2.imshow("LiDAR Projection", img_lidar_proj)
    return img_lidar_proj


def show_lidar_with_boxes(pc_velo, objects_gt, calib, fig, objects_pred=None,
                           img_fov=False, img_width=None, img_height=None,
                           show_reflectance=False, pc_label_channel=None,
                           show_gt=True, show_pred=True, show_orientation=False):
    """ Visualizes LiDAR point cloud and 3D bounding boxes using Mayavi. """
    # Import mayavi here to avoid hard dependency if not used
    try:
        import mayavi.mlab as mlab
    except ImportError:
        print("Mayavi not installed. Skipping 3D LiDAR visualization.")
        print("Install Mayavi: pip install mayavi")
        return None # Indicate failure

    if pc_velo is None or calib is None:
        print("Error: Cannot show LiDAR, missing point cloud or calibration.")
        return fig # Return existing fig or None

    # --- Filter points to Image FOV if requested ---
    if img_fov and img_width is not None and img_height is not None:
        pc_velo_filtered, _ = get_lidar_in_image_fov(pc_velo, calib, img_width, img_height)
        if pc_velo_filtered is None or pc_velo_filtered.shape[0] == 0:
            print("Warning: No points left after FOV filtering.")
            # Draw empty scene or just return
            # Let's still draw the axes and origin
            pc_to_draw = np.zeros((1, pc_velo.shape[1])) # Dummy point if all filtered
        else:
             pc_to_draw = pc_velo_filtered
    else:
        pc_to_draw = pc_velo

    # --- Draw LiDAR Points ---
    fig = viz_util.draw_lidar(pc_to_draw, fig=fig,
                              color_by_reflectance=show_reflectance,
                              label_channel=pc_label_channel)

    # --- Prepare GT Boxes ---
    boxes3d_gt_velo = []
    box_labels_gt = []
    orientations_gt = []
    if show_gt and objects_gt:
        for obj in objects_gt:
            # +++ Filter GT objects +++
            if obj.type not in TARGET_CLASSES:
                 continue
            # +++++++++++++++++++++++++
            if obj.type == "DontCare": continue # Already filtered by TARGET_CLASSES, but good practice
            _, box3d_pts_3d_cam = utils.compute_box_3d(obj, calib.P2)
            if box3d_pts_3d_cam is not None:
                box3d_velo = calib.project_rect_to_velo(box3d_pts_3d_cam)
                boxes3d_gt_velo.append(box3d_velo)
                box_labels_gt.append(obj.type)
                if show_orientation:
                     _, ori3d_cam = utils.compute_orientation_3d(obj, calib.P2)
                     if ori3d_cam is not None:
                         orientations_gt.append(calib.project_rect_to_velo(ori3d_cam))
                     else:
                          orientations_gt.append(None)

    # --- Prepare Predicted Boxes ---
    boxes3d_pred_velo = []
    box_labels_pred = []
    orientations_pred = []
    if show_pred and objects_pred:
         for obj in objects_pred:
             # +++ Filter Predicted objects +++
             if obj.type not in TARGET_CLASSES:
                 continue
             # ++++++++++++++++++++++++++++++
             if obj.type == "DontCare": continue # Optional filtering
             _, box3d_pts_3d_cam = utils.compute_box_3d(obj, calib.P2)
             if box3d_pts_3d_cam is not None:
                 box3d_velo = calib.project_rect_to_velo(box3d_pts_3d_cam)
                 boxes3d_pred_velo.append(box3d_velo)
                 box_labels_pred.append(obj.type) # Assuming pred format has type
                 if show_orientation:
                     _, ori3d_cam = utils.compute_orientation_3d(obj, calib.P2)
                     if ori3d_cam is not None:
                         orientations_pred.append(calib.project_rect_to_velo(ori3d_cam))
                     else:
                         orientations_pred.append(None)


    # --- Draw Boxes and Orientations ---
    if boxes3d_gt_velo:
        fig = viz_util.draw_gt_boxes3d(boxes3d_gt_velo, fig, labels=box_labels_gt, is_pred=False)
        if show_orientation:
            for i, ori in enumerate(orientations_gt):
                 if ori is not None:
                      color = viz_util.BOX_COLOR_MAP.get(box_labels_gt[i], viz_util.DEFAULT_COLOR)
                      viz_util.draw_orientation_arrow(ori, fig, color=color)

    if boxes3d_pred_velo:
        fig = viz_util.draw_gt_boxes3d(boxes3d_pred_velo, fig, labels=box_labels_pred, is_pred=True)
        if show_orientation:
            for i, ori in enumerate(orientations_pred):
                 if ori is not None:
                     viz_util.draw_orientation_arrow(ori, fig, color=viz_util.PREDICTION_COLOR)

    mlab.show(1) # Show the plot briefly (or remove to keep window open)
    return fig


def show_lidar_topview_with_boxes(pc_velo, objects_gt, calib, objects_pred=None):
    """ Generates and displays a BEV map with projected 3D boxes. """
    if pc_velo is None or calib is None:
        print("Error: Cannot show BEV, missing point cloud or calibration.")
        return None

    # Generate BEV representation (Height, Intensity, Density channels)
    bev_map = utils.lidar_to_top(pc_velo) # H x W x C

    # Create a displayable image from BEV (e.g., using density or max height)
    top_image = utils.draw_top_image(bev_map) # H x W x 3 (uint8)

    # --- Prepare GT Boxes for BEV ---
    boxes3d_gt_velo = []
    box_labels_gt = []
    if objects_gt:
        for obj in objects_gt:
            # +++ Filter GT objects +++
            if obj.type not in TARGET_CLASSES:
                 continue
            # +++++++++++++++++++++++++
            if obj.type == "DontCare": continue
            _, box3d_pts_3d_cam = utils.compute_box_3d(obj, calib.P2)
            if box3d_pts_3d_cam is not None:
                 box3d_velo = calib.project_rect_to_velo(box3d_pts_3d_cam)
                 boxes3d_gt_velo.append(box3d_velo)
                 box_labels_gt.append(obj.type)

    # --- Prepare Predicted Boxes for BEV ---
    boxes3d_pred_velo = []
    box_labels_pred = []
    pred_scores = [] # Assuming prediction format might have scores
    if objects_pred:
        for obj in objects_pred:
             # +++ Filter Predicted objects +++
             if obj.type not in TARGET_CLASSES:
                 continue
             # ++++++++++++++++++++++++++++++
             if obj.type == "DontCare": continue
             _, box3d_pts_3d_cam = utils.compute_box_3d(obj, calib.P2)
             if box3d_pts_3d_cam is not None:
                 box3d_velo = calib.project_rect_to_velo(box3d_pts_3d_cam)
                 boxes3d_pred_velo.append(box3d_velo)
                 box_labels_pred.append(obj.type)
                 # Extract score if available in prediction object (e.g., obj.score)
                 # score = getattr(obj, 'score', None) # Example: get score attribute if it exists
                 # pred_scores.append(score)


    # --- Draw Boxes on BEV Image ---
    if boxes3d_gt_velo:
        top_image = utils.draw_box3d_on_top(top_image, boxes3d_gt_velo,
                                             text_labels=box_labels_gt, is_gt=True)

    if boxes3d_pred_velo:
        top_image = utils.draw_box3d_on_top(top_image, boxes3d_pred_velo,
                                             text_labels=box_labels_pred,
                                             scores=pred_scores if pred_scores else None,
                                             is_gt=False, color=(255, 0, 0)) # Red for preds

    cv2.imshow("Top View (BEV)", top_image)
    return top_image


# --- Main Execution Logic ---
def dataset_viz(args):
    """ Main function to load data and trigger visualizations based on args. """
    dataset = KittiObjectDataset(args.dir, split=args.split, args=args)

    if len(dataset) == 0:
        print("No samples found in the dataset. Exiting.")
        return

    # Initialize Mayavi figure only if needed
    mlab_fig = None
    if args.show_lidar_with_boxes or args.show_lidar_topview_with_boxes:
         # Conditional import to avoid hard dependency
         try:
             import mayavi.mlab as mlab
             if args.show_lidar_with_boxes: # Only create fig if using Mayavi 3D plot
                 mlab_fig = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1200, 800))
         except ImportError:
             print("Mayavi not installed. Skipping 3D LiDAR/BEV visualizations.")
             print("Install Mayavi: pip install mayavi")
             args.show_lidar_with_boxes = False # Disable flags if mlab failed
             # We can still show BEV image with cv2 even if mlab fails

    # Determine range of indices to visualize
    if args.ind is not None and args.ind >= 0:
        indices = range(args.ind, min(args.ind + 1, len(dataset))) # Single index
    elif args.range is not None:
        start, end = args.range
        indices = range(max(0, start), min(end, len(dataset))) # Range of indices
    else:
        indices = range(len(dataset)) # All indices

    for data_idx in indices:
        print(f"\n--- Loading Sample Index: {data_idx:06d} ---")

        # --- Load Data ---
        objects_gt = dataset.get_label_objects(data_idx) # Ground truth
        objects_pred = dataset.get_pred_objects(data_idx) if args.pred else None # Predictions
        calib = dataset.get_calibration(data_idx)
        pc_velo = dataset.get_lidar(data_idx, n_vec=4) # Load X,Y,Z,Reflectance
        img = dataset.get_image(data_idx)
        depth_map, _ = dataset.get_depth(data_idx) if args.depth else (None, False)

        if calib is None:
             print("Skipping frame due to missing calibration.")
             continue
        if img is None:
             print("Skipping frame due to missing image.")
             continue
        # Velo can be None, handled by viz functions

        img_height, img_width, _ = img.shape if img is not None else (0, 0, 0)

        # Print info
        print(f"Image shape: {img.shape if img is not None else 'None'}")
        print(f"LiDAR points: {pc_velo.shape if pc_velo is not None else 'None'}")
        # print(f"GT Objects: {len(objects_gt)}")
        # if objects_pred: print(f"Pred Objects: {len(objects_pred)}")

        # --- Perform Visualizations ---

        # 1. Image with 2D/3D Boxes
        if args.show_image_with_boxes:
            show_image_with_boxes(img, objects_gt, calib, objects_pred=objects_pred, show_3d=True)

        # 2. LiDAR point cloud with 3D boxes (Mayavi)
        if args.show_lidar_with_boxes and mlab_fig is not None:
            mlab.clf(mlab_fig) # Clear previous plot
            mlab_fig = show_lidar_with_boxes(
                pc_velo, objects_gt, calib, mlab_fig,
                objects_pred=objects_pred,
                img_fov=args.img_fov, img_width=img_width, img_height=img_height,
                show_reflectance=args.show_reflectance,
                show_orientation=True # Show orientation arrows
            )

        # 3. LiDAR points projected onto image (OpenCV)
        if args.show_lidar_on_image:
            show_lidar_on_image(pc_velo, img, calib)

        # 4. Top-view BEV map with 3D boxes (OpenCV)
        if args.show_lidar_topview_with_boxes:
            show_lidar_topview_with_boxes(pc_velo, objects_gt, calib, objects_pred=objects_pred)


        # --- Wait for User Input ---
        print("Press any key in an OpenCV window to continue to next frame (or Q to quit)...")
        key = cv2.waitKey(0) & 0xFF # Wait indefinitely until a key is pressed
        if key == ord('q') or key == 27: # Quit if 'q' or ESC is pressed
             print("Exiting.")
             break
        if key == ord('k'): # Kill all windows (useful if script hangs)
             cv2.destroyAllWindows()
             break

    # Clean up OpenCV windows
    cv2.destroyAllWindows()
    if mlab_fig is not None:
        try:
             import mayavi.mlab as mlab
             mlab.close(all=True)
        except ImportError:
            pass # Already handled

# --- Argument Parser ---
def parse_args():
    parser = argparse.ArgumentParser(description="KITTI Object Detection Visualization Tool")
    parser.add_argument(
        "-d", "--dir", type=str, default="../data/kitti",
        help="Root directory of the KITTI dataset (default: ../data/kitti relative to OpenPCDet/tools)"
    )
    parser.add_argument(
        "--split", type=str, default="training", choices=['training', 'testing'],
        help="Dataset split to visualize (default: training)"
    )
    parser.add_argument(
        "-i", "--ind", type=int, default=None, metavar="N",
        help="Visualize a single specific sample index (default: None, visualize all or range)"
    )
    parser.add_argument(
        "--range", type=int, nargs=2, metavar=('START', 'END'),
        help="Visualize sample indices within a range [START, END)"
    )
    parser.add_argument(
        "-p", "--pred", action="store_true",
        help="Load and display prediction results (from default 'pred' directory)"
    )
    parser.add_argument(
        "--vis", action="store_true",
        help="Enable visualization windows (required to see output)"
    )
    parser.add_argument(
        "--depth", action="store_true",
        help="Load and potentially use depth maps (currently not explicitly visualized)"
    )
    # Visualization type flags
    parser.add_argument(
        "--show_image_with_boxes", action="store_true", help="Show image with 2D/3D boxes (OpenCV)"
    )
    parser.add_argument(
        "--show_lidar_with_boxes", action="store_true", help="Show LiDAR point cloud with 3D boxes (Mayavi)"
    )
    parser.add_argument(
        "--show_lidar_on_image", action="store_true", help="Show LiDAR points projected onto image (OpenCV)"
    )
    parser.add_argument(
        "--show_lidar_topview_with_boxes", action="store_true", help="Show BEV map with 3D boxes (OpenCV)"
    )
    # LiDAR visualization options
    parser.add_argument(
        "--img_fov", action="store_true", help="Filter LiDAR points to image FOV in Mayavi view"
    )
    parser.add_argument(
        "--show_reflectance", action="store_true", help="Color LiDAR points by reflectance in Mayavi"
    )
    # Directory overrides (optional)
    parser.add_argument(
        "--lidar", type=str, default="velodyne", metavar="SUBDIR",
        help="Subdirectory name for LiDAR data (default: velodyne)"
    )
    parser.add_argument(
        "--depthdir", type=str, default="depth_2", metavar="SUBDIR",
        help="Subdirectory name for depth data (default: depth_2)"
    )
    parser.add_argument(
        "--preddir", type=str, default="pred", metavar="SUBDIR",
        help="Subdirectory name for prediction labels (default: pred)"
    )

    args = parser.parse_args()

    if not args.vis:
        print("Warning: --vis flag not set. No visualization windows will be shown.")
        # Disable all show flags if --vis is not set
        args.show_image_with_boxes = False
        args.show_lidar_with_boxes = False
        args.show_lidar_on_image = False
        args.show_lidar_topview_with_boxes = False

    if not (args.show_image_with_boxes or args.show_lidar_with_boxes or \
            args.show_lidar_on_image or args.show_lidar_topview_with_boxes) and args.vis:
         print("Warning: --vis flag set, but no specific visualization type selected (--show_...). Nothing to display.")

    # Check if prediction dir exists if --pred is used
    if args.pred:
        pred_full_dir = os.path.join(args.dir, args.split, args.preddir)
        if not os.path.isdir(pred_full_dir):
             print(f"Warning: Prediction directory '{pred_full_dir}' not found, but --pred flag is set.")
             # Decide if this should be a fatal error or just a warning
             # args.pred = False # Optionally disable prediction loading if dir not found

    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        dataset_viz(args)
    except FileNotFoundError as e:
        print(f"\nError: A required directory or file was not found.")
        print(e)
        print("Please ensure the KITTI dataset is correctly structured at the specified path:")
        print(f"  {args.dir}")
        print(f"And that the '{args.split}' split exists with necessary subdirectories (image_2, velodyne, calib, label_2).")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()