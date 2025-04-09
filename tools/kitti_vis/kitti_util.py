""" Helper methods for loading and parsing KITTI data.

Original Authors: Charles R. Qi, Kui Xu
Date: September 2017/2018
Refactored: [Fu/2025]
"""
from __future__ import print_function

import numpy as np
import cv2
import os
from PIL import Image # Used potentially for viewing images if cv2 fails

# --- Constants for Top-View Projection ---
TOP_Y_MIN = -30.0
TOP_Y_MAX = 30.0
TOP_X_MIN = 0.0
TOP_X_MAX = 70.4  # Adjusted to match cbox
TOP_Z_MIN = -3.0  # Adjusted to match cbox
TOP_Z_MAX = 1.0   # Adjusted to match cbox

TOP_X_DIVISION = 0.1 # Reduced for finer grid to match common practice
TOP_Y_DIVISION = 0.1 # Reduced for finer grid
TOP_Z_DIVISION = 0.2 # Reduced for finer grid

# Default Voxel Grid boundary (matches VoxelNet/SECOND common practice)
# Slightly different from original TOP_X/Y/Z_MIN/MAX, standardized here
cbox = np.array([[0, 70.4], [-40, 40], [-3, 1]]) # x, y, z range


class KittiObject3d(object):
    """ 3d object label """

    def __init__(self, label_file_line):
        data = label_file_line.split(" ")
        data[1:] = [float(x) for x in data[1:]]

        # extract label, truncation, occlusion
        self.type = data[0]  # 'Car', 'Pedestrian', ...
        self.truncation = data[1]  # truncated pixel ratio [0..1]
        self.occlusion = int(
            data[2]
        )  # 0=visible, 1=partly occluded, 2=fully occluded, 3=unknown
        self.alpha = data[3]  # object observation angle [-pi..pi]

        # extract 2d bounding box in 0-based coordinates
        self.xmin = data[4]  # left
        self.ymin = data[5]  # top
        self.xmax = data[6]  # right
        self.ymax = data[7]  # bottom
        self.box2d = np.array([self.xmin, self.ymin, self.xmax, self.ymax])

        # extract 3d bounding box information
        self.h = data[8]  # box height
        self.w = data[9]  # box width
        self.l = data[10]  # box length (in meters)
        self.t = (data[11], data[12], data[13])  # location (x,y,z) in camera coord.
        self.ry = data[14]  # yaw angle (around Y-axis in camera coordinates) [-pi..pi]

    def estimate_difficulty(self):
        """ Estimates difficulty based on KITTI criteria. Returns str: Easy, Moderate, Hard, Unknown """
        bb_height = np.abs(self.ymax - self.ymin) # Corrected: ymax-ymin for height

        if bb_height >= 40 and self.occlusion == 0 and self.truncation <= 0.15:
            return "Easy"
        elif bb_height >= 25 and self.occlusion in [0, 1] and self.truncation <= 0.30:
            return "Moderate"
        elif bb_height >= 25 and self.occlusion in [0, 1, 2] and self.truncation <= 0.50:
            return "Hard"
        else:
            return "Unknown"

    def print_object(self):
        print(
            "Type, truncation, occlusion, alpha: %s, %.2f, %d, %.2f"
            % (self.type, self.truncation, self.occlusion, self.alpha)
        )
        print(
            "2d bbox (x0,y0,x1,y1): %.2f, %.2f, %.2f, %.2f"
            % (self.xmin, self.ymin, self.xmax, self.ymax)
        )
        print("3d bbox h,w,l: %.2f, %.2f, %.2f" % (self.h, self.w, self.l))
        print(
            "3d bbox location (cam rect), ry: (%.2f, %.2f, %.2f), %.2f"
            % (self.t[0], self.t[1], self.t[2], self.ry)
        )
        print("Difficulty: %s" % (self.estimate_difficulty()))


class Calibration(object):
    """ Calibration matrices and projection utilities.
        Handles transformations between Velodyne, Ref Cam, Rect Cam, and Image coords.

        3d XYZ in <label>.txt are in rect camera coord (x-right, y-down, z-front).
        2d box xy are in image2 coord (u-right, v-down).
        Points in <lidar>.bin are in Velodyne coord (x-front, y-left, z-up).

        Reference: http://www.cvlibs.net/publications/Geiger2013IJRR.pdf
                   http://www.cvlibs.net/datasets/kitti/setup.php
    """

    def __init__(self, calib_filepath):
        calibs = self._read_calib_file(calib_filepath)
        # Projection matrix from rect camera coord to image2 coord P2 = K * [R | t] (3x4)
        self.P2 = calibs['P2']
        self.P2 = np.reshape(self.P2, [3, 4])

        # Rigid transform from Velodyne coord to reference camera coord Tr = [R | t] (3x4)
        self.V2C = calibs['Tr_velo_to_cam']
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_trans(self.V2C)

        # Rotation from reference camera coord to rect camera coord (3x3)
        self.R0 = calibs['R0_rect']
        self.R0 = np.reshape(self.R0, [3, 3])

        # ---------------
        # # Compute transformation matrix from velodyne lidar points to camera (rect) coords
        # # x_rect = P2 * R0 * Tr * x_velo
        # self.V2R = np.dot(self.R0, self.V2C) # R0 * Tr_velo_to_cam
        # self.V2R = np.vstack((self.V2R, [0, 0, 0, 1])) # Make 4x4

        # # Compute transformation matrix from velodyne lidar points to image coords
        # # y_image = P2 * R0 * Tr * x_velo
        # self.V2I = np.dot(self.P2, np.dot(self.R0, self.V2C)) # P2 * R0 * Tr_velo_to_cam
        
        # Make Tr_velo_to_cam 4x4
        tr_v2c_padded = np.identity(4)
        tr_v2c_padded[:3, :4] = self.V2C # Fill R|t part -> 4x4

        # Make R0_rect 4x4
        r0_padded = np.identity(4)
        r0_padded[:3, :3] = self.R0 # Fill 3x3 rotation part -> 4x4

        # Compute Velo -> Rect transformation (homogeneous)
        # x_rect_hom = R0_padded * Tr_V2C_padded * x_velo_hom
        self.V2R_hom = np.dot(r0_padded, tr_v2c_padded) # (4x4) * (4x4) -> 4x4

        # Compute Velo -> Image transformation (homogeneous input -> homogeneous output)
        # y_image_hom = P2 * x_rect_hom = P2 * V2R_hom * x_velo_hom
        self.V2I = np.dot(self.P2, self.V2R_hom) # (3x4) * (4x4) -> (3, 4)
        # ---------------

        # Camera intrinsics and baseline (for stereo)
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu) # Baseline (relative to cam0)
        self.ty = self.P2[1, 3] / (-self.fv)

    def _read_calib_file(self, filepath):
        """ Reads a calibration file and parses into a dictionary. """
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    # Skip lines like 'calib_time', etc.
                    pass
        # Ensure standard matrices are present
        if 'P2' not in data or 'R0_rect' not in data or 'Tr_velo_to_cam' not in data:
             raise IOError(f"Required calibration matrices (P2, R0_rect, Tr_velo_to_cam) not found in {filepath}")

        # Handle potential missing optional matrices if needed (e.g., P0, P1, P3, Tr_imu_to_velo)
        # For simplicity, we assume the required ones exist for standard tasks.

        # Compatibility: if R0_rect is identity and Tr_velo_to_cam has last row [0,0,0,1]
        # adjust them to be 3x3 and 3x4
        if 'R0_rect' in data and data['R0_rect'].size == 12: # check if it's 3x4 (padded)
             data['R0_rect'] = data['R0_rect'].reshape(3,4)[:3,:3]
        elif 'R0_rect' in data and data['R0_rect'].size == 9: # check if it's 3x3
             data['R0_rect'] = data['R0_rect'].reshape(3,3)

        if 'Tr_velo_to_cam' in data and data['Tr_velo_to_cam'].size == 16: # check if it's 4x4
             data['Tr_velo_to_cam'] = data['Tr_velo_to_cam'].reshape(4,4)[:3,:]
        elif 'Tr_velo_to_cam' in data and data['Tr_velo_to_cam'].size == 12: # check if it's 3x4
             data['Tr_velo_to_cam'] = data['Tr_velo_to_cam'].reshape(3,4)

        return data

    def cart2hom(self, pts_3d):
        """ Converts Nx3 points to Nx4 homogeneous coordinates. """
        n = pts_3d.shape[0]
        pts_3d_hom = np.hstack((pts_3d, np.ones((n, 1))))
        return pts_3d_hom

    # ====== Coordinate Transformations ======

    def project_velo_to_ref(self, pts_3d_velo):
        """ Projects Velodyne points (Nx3) to Reference Camera coordinates (Nx3). """
        pts_3d_velo = self.cart2hom(pts_3d_velo)  # Nx4
        return np.dot(pts_3d_velo, np.transpose(self.V2C)) # (Nx4) * (4x3) -> Nx3

    def project_ref_to_velo(self, pts_3d_ref):
        """ Projects Reference Camera points (Nx3) to Velodyne coordinates (Nx3). """
        pts_3d_ref = self.cart2hom(pts_3d_ref)  # Nx4
        return np.dot(pts_3d_ref, np.transpose(self.C2V)) # (Nx4) * (4x3) -> Nx3

    def project_rect_to_ref(self, pts_3d_rect):
        """ Projects Rectified Camera points (Nx3) to Reference Camera coordinates (Nx3). """
        # inverse rotation: pts_ref = R0_inv * pts_rect
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(pts_3d_rect)))

    def project_ref_to_rect(self, pts_3d_ref):
        """ Projects Reference Camera points (Nx3) to Rectified Camera coordinates (Nx3). """
        # pts_rect = R0 * pts_ref
        return np.transpose(np.dot(self.R0, np.transpose(pts_3d_ref)))

    def project_rect_to_velo(self, pts_3d_rect):
        """ Projects Rectified Camera points (Nx3) to Velodyne coordinates (Nx3). """
        pts_3d_ref = self.project_rect_to_ref(pts_3d_rect)
        return self.project_ref_to_velo(pts_3d_ref)

    def project_velo_to_rect(self, pts_3d_velo):
        """ Projects Velodyne points (Nx3) to Rectified Camera coordinates (Nx3). """
        pts_3d_ref = self.project_velo_to_ref(pts_3d_velo)
        return self.project_ref_to_rect(pts_3d_ref)

    # ====== Projections to Image ======

    def project_rect_to_image(self, pts_3d_rect):
        """ Projects Rectified Camera points (Nx3) to Image coordinates (Nx2). """
        pts_3d_rect = self.cart2hom(pts_3d_rect) # Nx4
        pts_2d = np.dot(pts_3d_rect, np.transpose(self.P2))  # (Nx4) * (4x3) -> Nx3
        pts_2d[:, 0] /= pts_2d[:, 2] # Normalize u
        pts_2d[:, 1] /= pts_2d[:, 2] # Normalize v
        return pts_2d[:, 0:2]

    def project_velo_to_image(self, pts_3d_velo):
        """ Projects Velodyne points (Nx3) to Image coordinates (Nx2). """
        pts_3d_rect = self.project_velo_to_rect(pts_3d_velo)
        return self.project_rect_to_image(pts_3d_rect)

    # ====== Projections from Image ======

    def project_image_to_rect(self, uv_depth):
        """ Projects Image points + depth (Nx3) [u, v, Z] to Rectified Camera coordinates (Nx3).
            Z is depth in the rectified camera coordinate system.
        """
        n = uv_depth.shape[0]
        u = uv_depth[:, 0]
        v = uv_depth[:, 1]
        z_rect = uv_depth[:, 2]

        x_rect = ((u - self.cu) * z_rect) / self.fu + self.tx # Include baseline offset if P2 includes it
        y_rect = ((v - self.cv) * z_rect) / self.fv + self.ty
        # z_rect is already given

        pts_3d_rect = np.vstack((x_rect, y_rect, z_rect)).transpose()
        return pts_3d_rect

    def project_image_to_velo(self, uv_depth):
        """ Projects Image points + depth (Nx3) [u, v, Z] to Velodyne coordinates (Nx3). """
        pts_3d_rect = self.project_image_to_rect(uv_depth)
        return self.project_rect_to_velo(pts_3d_rect)

    # ====== Depth Map Processing ======
    def project_depth_to_velo(self, depth_map, constraint_box=True):
        """ Projects a depth map to Velodyne coordinates.
        Args:
            depth_map: HxW numpy array with depth values (Z_rect).
            constraint_box: If True, filters points outside the predefined `cbox`.
        Returns:
            Nx3 numpy array of points in Velodyne coordinates.
        """
        img_height, img_width = depth_map.shape
        u_coords, v_coords = np.meshgrid(np.arange(img_width), np.arange(img_height))

        # Flatten coordinates and depth
        u_flat = u_coords.flatten()
        v_flat = v_coords.flatten()
        depth_flat = depth_map.flatten()

        # Create uv_depth array (Nx3)
        # Filter out invalid depth values (e.g., 0 or very large values if applicable)
        valid_indices = depth_flat > 0.1 # Example threshold, adjust if needed
        uv_depth = np.vstack((u_flat[valid_indices], v_flat[valid_indices], depth_flat[valid_indices])).transpose()

        # Project to rectified camera coordinates
        pts_3d_rect = self.project_image_to_rect(uv_depth)

        # Project to velodyne coordinates
        pts_3d_velo = self.project_rect_to_velo(pts_3d_rect)

        if constraint_box:
            # Filter points outside the predefined cbox volume
            box_fov_inds = (
                (pts_3d_velo[:, 0] < cbox[0, 1]) & (pts_3d_velo[:, 0] >= cbox[0, 0]) &
                (pts_3d_velo[:, 1] < cbox[1, 1]) & (pts_3d_velo[:, 1] >= cbox[1, 0]) &
                (pts_3d_velo[:, 2] < cbox[2, 1]) & (pts_3d_velo[:, 2] >= cbox[2, 0])
            )
            pts_3d_velo = pts_3d_velo[box_fov_inds]

        return pts_3d_velo


# === Rotation Matrices ===
def rotx(t):
    """ 3D Rotation about the x-axis. """
    c = np.cos(t); s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

def roty(t):
    """ Rotation about the y-axis. """
    c = np.cos(t); s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

def rotz(t):
    """ Rotation about the z-axis. """
    c = np.cos(t); s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])

# === Transformation Matrix Helpers ===
def transform_from_rot_trans(R, t):
    """ Creates a 4x4 transformation matrix from a 3x3 rotation matrix and a 3x1 translation vector. """
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def inverse_rigid_trans(Tr):
    """ Inverts a 3x4 rigid body transformation matrix [R|t]. Returns a 3x4 matrix. """
    inv_Tr = np.zeros_like(Tr)  # 3x4
    inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3])
    inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3])
    return inv_Tr

# === File Loading Functions ===
def read_label(label_filename):
    """ Reads a KITTI label file (e.g., label_2/*.txt).
    Returns:
        list of KittiObject3d objects.
    """
    lines = [line.rstrip() for line in open(label_filename)]
    objects = [KittiObject3d(line) for line in lines]
    return objects

def load_image(img_filename):
    """ Loads an image using OpenCV. Returns BGR image. """
    return cv2.imread(img_filename)

def load_depth(img_filename):
    """ Loads a depth map from a 16-bit PNG file (standard KITTI format).
    Returns:
        depth_map (HxW float32): Depth in meters.
        is_exist (bool): True if file exists, False otherwise.
    """
    if not os.path.exists(img_filename):
        # Determine expected shape (this might need adjustment based on dataset variant)
        # Common KITTI shapes: (370, 1224), (375, 1242), etc.
        # Let's assume a common shape, but ideally, this should be known or passed.
        print(f"Warning: Depth file not found {img_filename}, returning zeros.")
        # Need a default shape if file doesn't exist
        # Example: return np.zeros((375, 1242), dtype=np.float32), False
        # Returning None is safer if shape is unknown
        return None, False

    depth_img = cv2.imread(img_filename, cv2.IMREAD_UNCHANGED)
    if depth_img is None:
        print(f"Warning: Failed to load depth file {img_filename}, returning None.")
        return None, False

    depth_map = depth_img.astype(np.float32) / 256.0
    return depth_map, True


def load_velo_scan(velo_filename, dtype=np.float32, n_vec=4):
    """ Loads a Velodyne scan file (.bin).
    Args:
        velo_filename (str): Path to the .bin file.
        dtype (np.dtype): Data type (float32 or float64).
        n_vec (int): Number of channels (typically 4: x, y, z, intensity).
    Returns:
        Nxn_vec numpy array, or None if file not found.
    """
    if not os.path.exists(velo_filename):
        print(f"Warning: Velodyne file not found {velo_filename}, returning None.")
        return None
    scan = np.fromfile(velo_filename, dtype=dtype)
    scan = scan.reshape((-1, n_vec))
    return scan


# === Top View (BEV) Generation ===
def lidar_to_top_coords(x, y):
    """ Converts Velodyne coordinates (x, y) to top-view image coordinates (pixel row, col). """
    # Note: Image origin (0,0) is top-left. Velodyne (0,0) is sensor center.
    # BEV image y-axis corresponds to Velodyne negative x-axis
    # BEV image x-axis corresponds to Velodyne negative y-axis

    # Check boundaries
    if x < TOP_X_MIN or x > TOP_X_MAX or y < TOP_Y_MIN or y > TOP_Y_MAX:
        return None, None

    pixel_x = int(np.floor((y - TOP_Y_MIN) / TOP_Y_DIVISION))
    pixel_y = int(np.floor((TOP_X_MAX - x) / TOP_X_DIVISION)) # Y pixel coord increases downwards (opposite of velo x)

    # Calculate image size
    height = int(np.ceil((TOP_X_MAX - TOP_X_MIN) / TOP_X_DIVISION))
    width = int(np.ceil((TOP_Y_MAX - TOP_Y_MIN) / TOP_Y_DIVISION))

    # Clamp coords to be within image bounds
    pixel_x = np.clip(pixel_x, 0, width - 1)
    pixel_y = np.clip(pixel_y, 0, height - 1)

    # Original implementation had swapped x/y in return compared to common BEV mapping
    # Returning (row, col) convention where row ~ velo x, col ~ velo y
    return pixel_y, pixel_x # (row, col)


def lidar_to_top(lidar):
    """ Creates a top-view (BEV) representation from LiDAR data using height/intensity/density maps.

    Args:
        lidar (Nx4 array): Velodyne points (x, y, z, intensity).

    Returns:
        numpy array (Height, Width, Channels): BEV map. Channels represent:
            - Height slices (based on Z divisions)
            - Max Intensity in cell
            - Density in cell (log scale)
    """
    # Filter points within the desired X, Y, Z range
    idx = np.where(
        (lidar[:, 0] >= TOP_X_MIN) & (lidar[:, 0] < TOP_X_MAX) &
        (lidar[:, 1] >= TOP_Y_MIN) & (lidar[:, 1] < TOP_Y_MAX) &
        (lidar[:, 2] >= TOP_Z_MIN) & (lidar[:, 2] < TOP_Z_MAX)
    )[0]
    lidar_filt = lidar[idx]

    if lidar_filt.shape[0] == 0:
         print("Warning: No lidar points found within the specified BEV range.")
         # Return an empty map with correct dimensions
         height = int(np.ceil((TOP_X_MAX - TOP_X_MIN) / TOP_X_DIVISION))
         width = int(np.ceil((TOP_Y_MAX - TOP_Y_MIN) / TOP_Y_DIVISION))
         num_z_slices = int(np.ceil((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION))
         channels = num_z_slices + 2 # Height slices + Intensity + Density
         return np.zeros(shape=(height, width, channels), dtype=np.float32)


    # Quantize coordinates
    # Note the coordinate mapping: BEV y ~ Velo x, BEV x ~ Velo y
    pxs = lidar_filt[:, 0]
    pys = lidar_filt[:, 1]
    pzs = lidar_filt[:, 2]
    prs = lidar_filt[:, 3] # Intensity/Reflectance

    # Quantized indices (col, row, z_slice)
    qxs = np.floor((pys - TOP_Y_MIN) / TOP_Y_DIVISION).astype(np.int32) # BEV X (Column)
    qys = np.floor((TOP_X_MAX - pxs) / TOP_X_DIVISION).astype(np.int32) # BEV Y (Row)
    qzs = np.floor((pzs - TOP_Z_MIN) / TOP_Z_DIVISION).astype(np.int32) # Z slice index

    # Calculate BEV map dimensions
    height = int(np.ceil((TOP_X_MAX - TOP_X_MIN) / TOP_X_DIVISION)) # Rows
    width = int(np.ceil((TOP_Y_MAX - TOP_Y_MIN) / TOP_Y_DIVISION))   # Cols
    num_z_slices = int(np.ceil((TOP_Z_MAX - TOP_Z_MIN) / TOP_Z_DIVISION))
    channels = num_z_slices + 2 # Height slices + Max Intensity + Density

    top = np.zeros(shape=(height, width, channels), dtype=np.float32)

    # Efficiently create the BEV map
    # Sort points by their quantized coordinates for faster processing (optional but can help)
    # sorted_indices = np.lexsort((qzs, qxs, qys))
    # qxs, qys, qzs, prs, pzs = qxs[sorted_indices], qys[sorted_indices], qzs[sorted_indices], prs[sorted_indices], pzs[sorted_indices]

    # Create a unique index for each (row, col) cell
    # This allows using np.unique and accumulators for speed
    cell_indices = qys * width + qxs # Unique ID for each (row, col) pixel

    # Get unique cell indices and their counts
    unique_cells, cell_inverse_indices, cell_counts = np.unique(
        cell_indices, return_inverse=True, return_counts=True
    )

    # Map unique cell indices back to (row, col)
    unique_rows = unique_cells // width
    unique_cols = unique_cells % width

    # --- Calculate Density Channel ---
    # Log density, clipped max value for stability
    log_density = np.log(cell_counts + 1) / np.log(64) # Normalize with log(64)
    log_density = np.clip(log_density, 0, 1.0)
    top[unique_rows, unique_cols, -1] = log_density # Last channel is density

    # --- Calculate Intensity Channel ---
    # Use scatter_max equivalent: group by cell and find max intensity
    # Create an array to store max intensity for each unique cell
    max_intensity_map = np.zeros(unique_cells.shape[0], dtype=np.float32)
    np.maximum.at(max_intensity_map, cell_inverse_indices, prs) # Efficiently find max intensity per cell
    top[unique_rows, unique_cols, -2] = max_intensity_map # Second to last channel is intensity

    # --- Calculate Height Channels (Slices) ---
    # Create an array to store max height *within each slice* for each unique cell
    max_height_in_slice_map = np.zeros((unique_cells.shape[0], num_z_slices), dtype=np.float32)

    # Relative height within the slice: pzs - (TOP_Z_MIN + qzs * TOP_Z_DIVISION)
    relative_pzs = pzs - (TOP_Z_MIN + qzs * TOP_Z_DIVISION)
    relative_pzs = np.clip(relative_pzs, 0, TOP_Z_DIVISION) # Height within slice

    # Combine cell index and z-slice index for unique identification
    cell_z_indices = cell_inverse_indices * num_z_slices + qzs

    # Find max relative height per (cell, z_slice) combination
    max_rel_height_per_cell_z = np.zeros(unique_cells.shape[0] * num_z_slices, dtype=np.float32)
    np.maximum.at(max_rel_height_per_cell_z, cell_z_indices, relative_pzs)

    # Reshape and place into the `top` array
    # max_rel_height_per_cell_z now contains max height for cell 0/slice 0, cell 0/slice 1, ..., cell 1/slice 0, ...
    max_height_map_reshaped = max_rel_height_per_cell_z.reshape((unique_cells.shape[0], num_z_slices))

    # Assign to the first `num_z_slices` channels of the `top` array
    # Normalize height by slice division? Original code did max(0, max_height - z), implying absolute height.
    # Let's keep relative height within slice [0, TOP_Z_DIVISION], maybe normalize later if needed.
    # Normalizing here:
    top[unique_rows, unique_cols, 0:num_z_slices] = max_height_map_reshaped / TOP_Z_DIVISION

    return top

# === Drawing Utilities ===

def draw_projected_box3d(image, qs, color=(0, 255, 0), thickness=2):
    """ Draws a projected 3D box onto an image (qs are projected 2D points).
        Input qs: (8, 2) array of vertices for the 3d box in image coords.
        Vertex order:
            1 -------- 0
           /|         /|
          2 -------- 3 .
          | |        | |
          . 5 -------- 4
          |/         |/
          6 -------- 7
    """
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Draw front face lines (0-1, 1-2, 2-3, 3-0)
        i, j = k, (k + 1) % 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
        # Draw back face lines (4-5, 5-6, 6-7, 7-4)
        i, j = k + 4, (k + 1) % 4 + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
        # Draw connecting lines (0-4, 1-5, 2-6, 3-7)
        i, j = k, k + 4
        cv2.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness, cv2.LINE_AA)
    return image

def draw_top_image(lidar_top_channels):
    """ Creates a visualizable RGB image from the BEV channels (e.g., max height). """
    # Example: Visualize the max height across all Z slices
    # You might want to visualize density, intensity, or specific height slices instead.
    # top_view_feature = np.max(lidar_top_channels[:, :, :-2], axis=2) # Max height across slices

    # Example: Visualize Density channel (last channel)
    top_view_feature = lidar_top_channels[:, :, -1] # Density channel

    # Normalize to 0-255
    top_image = top_view_feature - np.min(top_view_feature)
    divisor = np.max(top_image) - np.min(top_image)
    if divisor > 1e-6: # Avoid division by zero
         top_image = (top_image / divisor * 255)
    else:
         top_image = np.zeros_like(top_image) # Handle case of flat feature map

    top_image = np.dstack((top_image, top_image, top_image)).astype(np.uint8)
    return top_image

def draw_box3d_on_top(image, boxes3d_velo, color=(255, 0, 0), thickness=1,
                      text_labels=None, scores=None, is_gt=False):
    """ Draws 3D bounding boxes (already in Velo coords) onto the top-view BEV image.

    Args:
        image (HxWx3 uint8): The BEV image.
        boxes3d_velo (list of Nx8x3 arrays): List of boxes, each box has 8 corners in Velo coords.
        color (tuple): Default color.
        thickness (int): Line thickness.
        text_labels (list of str): Optional labels for each box.
        scores (list of float): Optional scores (used for color coding predictions).
        is_gt (bool): If True, uses green color and specific text position.
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1

    num_boxes = len(boxes3d_velo)
    # Determine starting y position for text based on whether GT or Pred
    start_y = 5 if is_gt else 25 * (len(text_labels) + 2) if text_labels else 85

    # Define heat map color function if needed (simplified version)
    def heat_map_rgb(min_val, max_val, value):
        ratio = 2 * (value - min_val) / (max_val - min_val)
        b = int(max(0, 255 * (1 - ratio)))
        r = int(max(0, 255 * (ratio - 1)))
        g = 255 - b - r
        return (r, g, b) # Return RGB tuple

    for i in range(num_boxes):
        box_velo = boxes3d_velo[i] # Shape (8, 3)

        # We only need the 4 ground corners (assuming standard KITTI format)
        # Order might be: 0,1,2,3 are bottom corners, 4,5,6,7 are top corners
        # The specific order depends on compute_box_3d. Let's assume we want the footprint.
        # Common BEV plots use corners 0, 1, 2, 3 (or sometimes 4, 5, 6, 7 projected)
        # Let's project corners 0, 1, 2, 3
        corners_velo = box_velo[0:4, :] # Get the bottom 4 corners (or whichever define the ground footprint)

        # Convert velo corners to BEV image coordinates
        corners_bev = []
        valid_corners = True
        for j in range(4):
            # Using velo x, y for projection
            row, col = lidar_to_top_coords(corners_velo[j, 0], corners_velo[j, 1])
            if row is None or col is None:
                valid_corners = False
                break # Skip drawing this box if any corner is outside BEV range
            corners_bev.append((col, row)) # (x, y) format for cv2.line

        if not valid_corners:
            continue # Skip this box

        corners_bev = np.array(corners_bev, dtype=np.int32)

        # Determine color
        box_color = (0, 255, 0) if is_gt else color
        if not is_gt and scores is not None and i < len(scores):
            # Use score for color, e.g., heatmap
             box_color = heat_map_rgb(0.0, 1.0, scores[i])

        # Draw the 4 lines on the BEV image
        cv2.line(image, corners_bev[0], corners_bev[1], box_color, thickness, cv2.LINE_AA)
        cv2.line(image, corners_bev[1], corners_bev[2], box_color, thickness, cv2.LINE_AA)
        cv2.line(image, corners_bev[2], corners_bev[3], box_color, thickness, cv2.LINE_AA)
        cv2.line(image, corners_bev[3], corners_bev[0], box_color, thickness, cv2.LINE_AA)

        # Add text label if provided
        if text_labels is not None and i < len(text_labels):
             label = text_labels[i]
             if scores is not None and i < len(scores) and not is_gt:
                 label = f"{label}: {scores[i]:.2f}"

             # Put text near the first corner (or center)
             text_pos = (corners_bev[0, 0] + 5, corners_bev[0, 1] + 5) # Offset from corner 0
             # Basic text positioning within image bounds
             text_x = max(5, min(image.shape[1] - 50, text_pos[0])) # Clamp x
             text_y = max(15, min(image.shape[0] - 5, text_pos[1])) # Clamp y

             cv2.putText(image, label, (text_x, text_y), font, font_scale, box_color, font_thickness, cv2.LINE_AA)

    return image


# === 3D Box Calculation ===
def compute_box_3d(obj: KittiObject3d, P: np.ndarray):
    """ Projects the 3d bounding box of a KittiObject3d into the image plane.
    Args:
        obj: The KittiObject3d object.
        P: 3x4 projection matrix (e.g., calib.P2).
    Returns:
        corners_2d (8, 2): Box corners in image coordinates. Returns None if box is behind camera.
        corners_3d (8, 3): Box corners in rectified camera coordinates.
    """
    # Compute rotation matrix around Y-axis in camera coordinates
    R = roty(obj.ry)

    # 3d bounding box dimensions
    l, w, h = obj.l, obj.w, obj.h

    # 3d bounding box corners (local coordinates relative to box center)
    # Standard practice: center is bottom center (x, y, z) = (0, -h/2, 0) in local frame?
    # KITTI definition: location obj.t is the *bottom center* of the 3D box.
    # Let's define corners relative to this bottom center (0, 0, 0).
    x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2] # Front/back relative to center
    y_corners = [0,   0,   0,    0,   -h,  -h,  -h,   -h ]  # Y is down, 0 is bottom, -h is top
    z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2] # Left/right relative to center

    corners_3d_local = np.vstack([x_corners, y_corners, z_corners]) # 3x8

    # Rotate corners
    corners_3d_rotated = np.dot(R, corners_3d_local) # 3x8

    # Translate to object location (center_bottom in camera coordinates)
    corners_3d = corners_3d_rotated + np.array(obj.t).reshape(3, 1) # Add translation vector (broadcasts)

    # Check if any corner is behind the camera (z <= 0.1 often used)
    if np.any(corners_3d[2, :] < 0.1):
        # print(f"Warning: Box for object type {obj.type} partially behind camera.")
        # Return None for 2D corners if you don't want to draw partially visible boxes
        # return None, corners_3d.transpose()
        pass # Allow drawing even if partially behind, projection handles clipping.


    # Project the 3d corners to the image plane
    corners_3d_h = np.vstack((corners_3d, np.ones((1, 8)))) # 4x8 (homogeneous)
    corners_2d_h = np.dot(P, corners_3d_h) # P(3x4) * corners_3d_h(4x8) -> 3x8

    # Normalize to get 2D image coordinates
    corners_2d = corners_2d_h[0:2, :] / corners_2d_h[2, :] # Divide by Z

    return corners_2d.transpose(), corners_3d.transpose() # Return (8, 2) and (8, 3)


def compute_orientation_3d(obj: KittiObject3d, P: np.ndarray):
    """ Projects the 3D orientation vector (forward direction) into the image plane.
    Args:
        obj: The KittiObject3d object.
        P: 3x4 projection matrix (e.g., calib.P2).
    Returns:
        orientation_2d (2, 2): Start and end points of orientation vector in image coords. None if behind camera.
        orientation_3d (2, 3): Start and end points in rectified camera coords.
    """
    R = roty(obj.ry)

    # Define orientation vector in local coordinates (e.g., from center_bottom forward along length)
    # Start point: center_bottom (0, 0, 0) local
    # End point: forward along L-axis (l, 0, 0) local (assuming L points along camera X in local frame before rotation)
    # Let's verify KITTI's convention: ry rotates around Y (down), L is along X (right), W is along Z (front)
    # If L is along X (local), then orientation vector is [l, 0, 0]
    orientation_3d_local = np.array([[0, obj.l], [0, 0], [0, 0]]) # 3x2 [start_local, end_local]

    # Rotate and translate
    orientation_3d = np.dot(R, orientation_3d_local) + np.array(obj.t).reshape(3, 1) # 3x2

    # Check if points are behind camera
    if np.any(orientation_3d[2, :] < 0.1):
        return None, orientation_3d.transpose()

    # Project to image
    orientation_3d_h = np.vstack((orientation_3d, np.ones((1, 2)))) # 4x2
    orientation_2d_h = np.dot(P, orientation_3d_h) # 3x2
    orientation_2d = orientation_2d_h[0:2, :] / orientation_2d_h[2, :] # 2x2

    return orientation_2d.transpose(), orientation_3d.transpose() # Return (2, 2) and (2, 3)


# --- Miscellaneous ---
# Keep the heat map function if needed elsewhere or for BEV coloring
def heat_map_rgb(min_val, max_val, value):
    """ Generates an RGB color from a value within a range using a basic heatmap logic. """
    if max_val <= min_val: # Avoid division by zero
        return (0, 255, 0) # Default to green if range is invalid
    ratio = 2 * (value - min_val) / (max_val - min_val)
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return (r, g, b) # RGB tuple