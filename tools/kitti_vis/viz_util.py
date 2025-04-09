""" Visualization code for point clouds and 3D bounding boxes with Mayavi.

Original Author Ref: https://github.com/hengck23/didi-udacity-2017/blob/master/baseline-04/kitti_data/draw.py
Modified by Charles R. Qi
Refactored: [Fu/2025]
"""

import numpy as np
import mayavi.mlab as mlab

BOX_COLOR_MAP = {
    "Car": (0, 1, 0),         # Green
    "Pedestrian": (0, 1, 1),  # Cyan
    "Cyclist": (1, 1, 0),     # Yellow
    "Van": (0, 0.5, 0.5),     # Teal
    "Truck": (0.5, 0, 0.5),   # Purple
    "Misc": (0.8, 0.8, 0.8),  # Light Gray
    "Tram": (1, 0.5, 0),      # Orange
    "Person_sitting": (1, 0, 1), # Magenta
    "DontCare": (0.5, 0.5, 0.5) # Gray
}

PREDICTION_COLOR = (1, 0, 0) # Red for predictions
DEFAULT_COLOR = (1, 1, 1)    # White as default


def draw_lidar(pc, color=None, fig=None, bgcolor=(0, 0, 0), pts_scale=1.0,
               pts_mode='point', pts_color=None, color_by_reflectance=False,
               label_channel=None):
    """ Draws LiDAR points using Mayavi.

    Args:
        pc (np.ndarray): Point cloud data (N, 3+). Needs at least x, y, z.
        color (np.ndarray, optional): N-dimensional array for coloring points (e.g., z-coordinate, reflectance).
        fig (Mayavi figure, optional): Existing figure to draw on. If None, creates a new one.
        bgcolor (tuple, optional): Background color (R, G, B) tuple (0-1).
        pts_scale (float, optional): Scale factor for the points.
        pts_mode (str, optional): Mayavi drawing mode ('point', 'sphere', etc.).
        pts_color (tuple, optional): Fixed color for all points (R, G, B) if `color` array is not used. Overrides `color`.
        color_by_reflectance (bool, optional): If True, uses 4th column of pc for color. Overrides `color`.
        label_channel (int, optional): If specified, uses this column index from pc for coloring (e.g., 4 for semantic label).

    Returns:
        Mayavi figure: The figure object used or created.
    """
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bgcolor, fgcolor=None, engine=None, size=(1000, 800))

    # Determine the scalar values for coloring
    if pts_color is not None:
        # Use fixed color, need dummy scalars
        scalars = np.ones(pc.shape[0])
        pts_color_tuple = pts_color # Already a tuple
    else:
        if label_channel is not None and pc.shape[1] > label_channel:
            scalars = pc[:, label_channel]
            pts_color_tuple = None # Use colormap based on labels
        elif color_by_reflectance and pc.shape[1] >= 4:
            scalars = pc[:, 3] # Use reflectance
            pts_color_tuple = None # Use colormap based on reflectance
        elif color is not None:
            scalars = color # Use provided color array (e.g., z-value)
            pts_color_tuple = None # Use colormap based on this array
        else:
            scalars = pc[:, 2] # Default to Z-coordinate for coloring
            pts_color_tuple = None # Use colormap based on height

    # Draw points
    mlab.points3d(pc[:, 0], pc[:, 1], pc[:, 2], scalars,
                  mode=pts_mode,
                  colormap='gnuplot', # Common colormap, others: 'viridis', 'jet', 'coolwarm'
                  scale_factor=pts_scale,
                  figure=fig,
                  color=pts_color_tuple) # color is None if using colormap

    # Draw origin
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode='sphere', scale_factor=0.3)

    # Draw axes
    axes = np.array([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]]) # Length 2 axes
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]], color=(1, 0, 0), tube_radius=None, figure=fig) # X: Red
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]], color=(0, 1, 0), tube_radius=None, figure=fig) # Y: Green
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]], color=(0, 0, 1), tube_radius=None, figure=fig) # Z: Blue

    # Default view point
    mlab.view(azimuth=180, elevation=70, distance=60.0, focalpoint=[12.0, 0.0, -2.0], figure=fig)
    # Adjust focal point and distance based on typical KITTI scene scale

    return fig


def draw_gt_boxes3d(gt_boxes3d, fig, color=None, line_width=2, draw_text=True,
                    text_scale=(0.5, 0.5, 0.5), color_list=None, labels=None, is_pred=False):
    """ Draws 3D bounding boxes in a Mayavi figure.

    Args:
        gt_boxes3d (list or np.ndarray): List or array of boxes. Each box is (8, 3) array of corner coordinates (world/velo).
        fig (Mayavi figure): The figure to draw on.
        color (tuple, optional): Default color (R, G, B) for boxes if color_list or labels are not provided.
        line_width (int, optional): Width of the box lines.
        draw_text (bool, optional): If True, draws labels near the boxes.
        text_scale (tuple, optional): Scale of the text labels.
        color_list (list, optional): List of colors, one for each box. Overrides default color.
        labels (list of str, optional): List of labels (e.g., object types) for each box. Used for text and color mapping.
        is_pred (bool, optional): If True, uses the prediction color.
    """
    num_boxes = len(gt_boxes3d)
    for i in range(num_boxes):
        box_corners = gt_boxes3d[i] # Shape (8, 3)

        # Determine color
        current_color = DEFAULT_COLOR
        if is_pred:
            current_color = PREDICTION_COLOR
        elif color_list is not None and i < len(color_list):
            current_color = color_list[i]
        elif labels is not None and i < len(labels) and labels[i] in BOX_COLOR_MAP:
            current_color = BOX_COLOR_MAP[labels[i]]
        elif color is not None:
            current_color = color

        # Draw text label
        label_text = ""
        if draw_text and labels is not None and i < len(labels):
            label_text = labels[i]
            # Position text near a corner (e.g., corner 4 - top-front-left relative to standard order)
            text_pos = box_corners[4, :]
            # Small offset to prevent overlap with box lines
            mlab.text3d(text_pos[0], text_pos[1], text_pos[2] + 0.2, label_text,
                        scale=text_scale, color=current_color, figure=fig)

        # Draw the 12 lines of the box
        for k in range(0, 4):
            # Front face (0-1, 1-2, 2-3, 3-0) assuming standard corner indexing
            i1, j1 = k, (k + 1) % 4
            mlab.plot3d([box_corners[i1, 0], box_corners[j1, 0]],
                        [box_corners[i1, 1], box_corners[j1, 1]],
                        [box_corners[i1, 2], box_corners[j1, 2]],
                        color=current_color, tube_radius=None, line_width=line_width, figure=fig)

            # Back face (4-5, 5-6, 6-7, 7-4)
            i2, j2 = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([box_corners[i2, 0], box_corners[j2, 0]],
                        [box_corners[i2, 1], box_corners[j2, 1]],
                        [box_corners[i2, 2], box_corners[j2, 2]],
                        color=current_color, tube_radius=None, line_width=line_width, figure=fig)

            # Connecting lines (0-4, 1-5, 2-6, 3-7)
            i3, j3 = k, k + 4
            mlab.plot3d([box_corners[i3, 0], box_corners[j3, 0]],
                        [box_corners[i3, 1], box_corners[j3, 1]],
                        [box_corners[i3, 2], box_corners[j3, 2]],
                        color=current_color, tube_radius=None, line_width=line_width, figure=fig)

    return fig


def draw_orientation_arrow(ori3d_pts_3d_velo, fig, color=(1, 1, 1)):
    """ Draws an arrow representing the orientation (forward direction) of a box.

    Args:
        ori3d_pts_3d_velo (np.ndarray): (2, 3) array [start_point, end_point] in velo coords.
        fig (Mayavi figure): Figure to draw on.
        color (tuple): Color of the arrow line.
    """
    if ori3d_pts_3d_velo is None or ori3d_pts_3d_velo.shape[0] != 2:
        return # Cannot draw if points are invalid

    start_pt = ori3d_pts_3d_velo[0, :]
    end_pt = ori3d_pts_3d_velo[1, :]

    mlab.plot3d([start_pt[0], end_pt[0]],
                [start_pt[1], end_pt[1]],
                [start_pt[2], end_pt[2]],
                color=color, tube_radius=0.05, line_width=1, figure=fig) # Small tube radius for arrow line


# Note: draw_lidar_simple, xyzwhl2eight, draw_xyzwhl, rotation helpers
# are omitted as they seemed less critical or redundant with draw_gt_boxes3d
# and the rotation matrices in kitti_util.py. Add them back if needed.