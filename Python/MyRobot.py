import os
import sys
import inspect
import math
import numpy as np
import time
import cv2
import CameraCalibration
import DectectSmarties as ds


# --- Dynamixel SDK path bootstrap -----------------------
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sdk_path = os.path.join(parentdir, "DynamixelSDK-3.7.31", "python", "src", "dynamixel_sdk")
if sdk_path not in sys.path:
    sys.path.insert(0, sdk_path)

from dynamixel_sdk import PortHandler, PacketHandler


def cosd(x_deg):  # cos with degrees
    return math.cos(math.radians(x_deg))


def sind(x_deg):  # sin with degrees
    return math.sin(math.radians(x_deg))

# ------------------------------- Camera ---------------------------------------
# Calibration file saved by CameraCalibration.py, next to this script


CALIBRATION_FILE = os.path.join(currentdir, "calibration_data.npz")

# Camera axes expressed in frame 4 (columns = x_c, y_c, z_c).
# z_c = x4 (optical axis points along the gripper). This is ONE plausible mounting -- verify on the robot!
R_EE_CAM = np.array([[ 0,  0, 1],
                     [-1,  0, 0],
                     [ 0, -1, 0]], dtype=float)

# Lens position in frame 4 [mm] (T04 -> T05): the lens is not centred on the gripper
CAM_OFFSET = np.array([-10.0, 25.0, 15.0])


def load_camera_calibration(path=CALIBRATION_FILE):
    """Return (K, dist) saved by CameraCalibration.py."""
    with np.load(path) as data:
        return data['mtx'], data['dist']


def camera_transform(T04):
    """Base-to-camera transform T05 from the end-effector transform T04."""
    T05 = np.eye(4)
    T05[:3, :3] = T04[:3, :3] @ R_EE_CAM
    T05[:3, 3] = T04[:3, :3] @ CAM_OFFSET + T04[:3, 3]
    return T05


def translation(x,y,z):
    return np.array([
        [1, 0, 0, x],
        [0, 1, 0, y],
        [0, 0, 1, z],
        [0, 0, 0, 1]
    ], dtype=float

    )

def pixel_to_3D(u, v, T04, K, dist, z_plane=0.0):
    """Project pixel (u, v) of the RAW (not undistorted) image onto the plane z = z_plane.

    T04: end-effector transform from MyRobot.forward(). Returns [x, y, z] in the base frame [mm].
    """
    # Undistort -> normalized image coordinates (xn, yn) = K^-1 [u, v, 1]

    
    T_05 = T04 @ translation(*CAM_OFFSET)

    R_5c = np.array([[0, 0, 1],
                     [0,-1, 0],
                     [1, 0, 0]], dtype=float)

    R_05 = T_05[:3,:3]
    R_0c = R_05 @ R_5c


    x_c = np.linalg.inv(K) @ np.array([u,v,1])
    x_w = R_0c @ x_c

    t_cam = T_05[:3,3]
    s = (z_plane - t_cam[2]) / x_w[2]
    
    x_world = t_cam + s*x_w

    return x_world


class MyRobot:
    # ----------------------------- Constants (AX-12A) -------------------------
    ADDR_MX_TORQUE_ENABLE = 24       # 1 byte
    ADDR_MX_GOAL_POSITION = 30       # 2 bytes
    ADDR_MX_MOVING_SPEED = 32        # 2 bytes
    ADDR_MX_TORQUE_LIMIT = 34        # 2 bytes
    ADDR_MX_PRESENT_POSITION = 36    # 2 bytes

    PROTOCOL_VERSION = 1.0
    BAUDRATE = 1000000
    DEVICENAME = "COM8"              # e.g., Windows: 'COM3' | Linux: '/dev/ttyUSB0' | Mac: '/dev/tty.usbserial-*'

    TORQUE_ENABLE = 1
    TORQUE_DISABLE = 0
    COMM_SUCCESS = 0
    DXL_MOVING_STATUS_THRESHOLD = 10  # [ticks], not used directly here
    TICKS_PER_DEG = 1.0 / 0.29        # ≈ 3.448275862 ticks per degree
    DEG_PER_TICK = 0.29

    # Link dimensions from the DH model [mm]
    D1 = 46.07
    A2 = 95.18
    A3 = 95.18
    A4 = 60.72

    # ------------------------------- Init -------------------------------------
    def __init__(self):
        # Robot + kinematics state
        self.motor_ids = [1, 2, 3, 4]
        self.gripper_motor_id = 5
        self.forward_transform = np.zeros((4, 4))
        self.joint_angles = np.array([0.0, 0.0, 0.0, 0.0])  # deg
        self.joint_pos = np.zeros((4, 4))
        self.draw_robot_flag = False
        self.use_smooth_speed_flag = False
        self.gripper_open_flag = True

        # deg limits: [-130,130], [-180,0], [-140,140], [-140,100]
        self.joint_limits = np.array([[-90, 90], [-180, 180], [-140, 140], [-140, 140]])

        # IK config (simple analytical below; no external toolbox)
        self.ik_weights = np.array([0.25, 0.25, 0.25, 1, 1, 1])

        # Motor offsets (deg)
        self.joint_offsets = np.array([150, 150, 150, 150], dtype=float)

        self.joint_angle_error = np.array([0.0, 0.0, 0.0, 0.0])
        self.init_status = 0

        self.movement_history = []  # list of dicts
        self.motor_speed = np.array([0.1, 0.1, 0.1, 0.1])   # [0..1]
        self.motor_torque = np.array([1.0, 1.0, 1.0, 1.0])  # [0..1]
        self.pitch = 0.0

        # --- SDK setup ---
        try:
            self.port_handler = PortHandler(self.DEVICENAME)
            self.packet_handler = PacketHandler(self.PROTOCOL_VERSION)

            if self.port_handler.openPort():
                print("Succeeded to open the port!")
            else:
                print("Failed to open the port! Reconnect Robot.")
                raise RuntimeError("openPort failed")

            if self.port_handler.setBaudRate(self.BAUDRATE):
                print("Succeeded to change the baudrate!")
            else:
                print("Failed to change the baudrate! Reconnect Robot.")
                raise RuntimeError("setBaudRate failed")

            # Initialize defaults
            self.set_speed(self.motor_speed, overwrite_speeds=True)
            self.set_torque_limit(self.motor_torque)
            print(self.read_joint_angles())
            self.move_j(0, 0, 0, 0)
            self.enable_motors()
            self.init_status = 1

        except Exception as e:
            print(f"Initialization error: {e}")
            self.init_status = 0

    # ----------------------------- Utilities ----------------------------------
    @staticmethod
    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    @classmethod
    def deg_to_ticks(cls, deg):
        return int(round(deg * cls.TICKS_PER_DEG))

    @classmethod
    def ticks_to_deg(cls, ticks):
        return ticks * cls.DEG_PER_TICK

    def _check_limits_one(self, deg, idx):
        lo, hi = self.joint_limits[idx]
        if not (lo <= deg <= hi):
            if idx == 0:
                raise ValueError("Angle Limits for first Axis Reached, Min/Max: ±130°")
            elif idx == 1:
                raise ValueError("Angle Limits for second Axis Reached, Min/Max: [−180°, 0°]")
            else:
                raise ValueError("Angle Limits Reached, Min/Max: ±100°")
        return deg

    # --------------------------- Motor I/O wrappers ---------------------------
    def _write1(self, dxl_id, addr, value):
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, addr, int(value))
        if dxl_comm_result != self.COMM_SUCCESS or dxl_error != 0:
            raise IOError(f"DXL write1 error id={dxl_id}, addr={addr}, result={dxl_comm_result}, error={dxl_error}")

    def _write2(self, dxl_id, addr, value):
        dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(self.port_handler, dxl_id, addr, int(value))
        if dxl_comm_result != self.COMM_SUCCESS or dxl_error != 0:
            raise IOError(f"DXL write2 error id={dxl_id}, addr={addr}, result={dxl_comm_result}, error={dxl_error}")

    def _read2(self, dxl_id, addr):
        value, dxl_comm_result, dxl_error = self.packet_handler.read2ByteTxRx(self.port_handler, dxl_id, addr)
        if dxl_comm_result != self.COMM_SUCCESS or dxl_error != 0:
            raise IOError(f"DXL read2 error id={dxl_id}, addr={addr}, result={dxl_comm_result}, error={dxl_error}")
        return value

    # ------------------------------- Speeds/Torque ----------------------------
    def smooth_speed(self, joint_angles_delta_deg):
        max_angle = float(np.max(np.abs(joint_angles_delta_deg)))
        if max_angle == 0:
            return
        speed_per_deg = max_angle / 100.0
        if speed_per_deg != 0:
            new_speeds = np.abs(joint_angles_delta_deg / speed_per_deg) * 0.01
            # preserve zeros by falling back to previous speeds
            for i in range(len(self.motor_speed)):
                if new_speeds[i] == 0:
                    new_speeds[i] = self.motor_speed[i]
                else:
                    new_speeds[i] = new_speeds[i] * self.motor_speed[i]
            self.set_speed(new_speeds, overwrite_speeds=False)

    def set_speed(self, speeds, overwrite_speeds=True):
        speeds = np.asarray(speeds, dtype=float)
        if overwrite_speeds:
            self.motor_speed = speeds.copy()

        for i, dxl_id in enumerate(self.motor_ids):
            s = float(speeds[i])
            if 0 < s <= 1:
                raw = int(round(s * 1023))
                self._write2(dxl_id, self.ADDR_MX_MOVING_SPEED, raw)
            else:
                print("Movement speed out of range, enter value between (0, 1].")

    def set_torque_limit(self, torques):
        torques = np.asarray(torques, dtype=float)
        self.motor_torque = torques.copy()
        for i, dxl_id in enumerate(self.motor_ids):
            t = float(torques[i])
            if 0 < t <= 1:
                self._write2(dxl_id, self.ADDR_MX_TORQUE_LIMIT, int(round(t * 1023)))
            else:
                print("Torque limit out of range, enter value between (0, 1].")

    # --------------------------------- Power ----------------------------------
    def enable_motors(self):
        for dxl_id in self.motor_ids:
            self._write1(dxl_id, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_ENABLE)
        print("Dynamixel torque enabled on all motors.")

    def disable_motors(self):
        for dxl_id in self.motor_ids:
            try:
                self._write1(dxl_id, self.ADDR_MX_TORQUE_ENABLE, self.TORQUE_DISABLE)
            except Exception as e:
                print(f"Disable motor {dxl_id} error: {e}")
        try:
            self.port_handler.closePort()
        finally:
            self.init_status = 0
        print("Dynamixels disabled and port closed.")

    # ------------------------------- Telemetry --------------------------------
    def get_position(self, motor_id):
        ticks = self._read2(motor_id, self.ADDR_MX_PRESENT_POSITION)
        return self.ticks_to_deg(ticks)

    def read_joint_angles(self):
        j_a = np.zeros(4, dtype=float)
        for i, dxl_id in enumerate(self.motor_ids):
            ticks = self._read2(dxl_id, self.ADDR_MX_PRESENT_POSITION)
            deg = self.ticks_to_deg(ticks) - self.joint_offsets[i]
            j_a[i] = deg
            self.joint_angle_error[i] = deg - self.joint_angles[i]
        return j_a

    # --------------------------- Kinematics (FK / IK) -------------------------
    def forward(self, j1_deg, j2_deg, j3_deg, j4_deg):
        """Return the base-to-end-effector transform for joint angles in degrees."""
        q1, q2, q3, q4 = np.radians([j1_deg, j2_deg, j3_deg, j4_deg])

        transform = np.array([
            [
                -math.cos(q1) * math.sin(q2 + q3 + q4),
                -math.cos(q1) * math.cos(q2 + q3 + q4),
                math.sin(q1),
                math.cos(q1) * (
                    -self.A2 * math.sin(q2)
                    - self.A3 * math.sin(q2 + q3)
                    - self.A4 * math.sin(q2 + q3 + q4)
                ),
            ],
            [
                -math.sin(q1) * math.sin(q2 + q3 + q4),
                -math.sin(q1) * math.cos(q2 + q3 + q4),
                -math.cos(q1),
                math.sin(q1) * (
                    -self.A2 * math.sin(q2)
                    - self.A3 * math.sin(q2 + q3)
                    - self.A4 * math.sin(q2 + q3 + q4)
                ),
            ],
            [
                math.cos(q2 + q3 + q4),
                -math.sin(q2 + q3 + q4),
                0.0,
                self.D1
                + self.A2 * math.cos(q2)
                + self.A3 * math.cos(q2 + q3)
                + self.A4 * math.cos(q2 + q3 + q4),
            ],
            [0.0, 0.0, 0.0, 1.0],
        ])
        self.forward_transform = transform
        return transform

    def inverse(self, x, y, z, pitch_deg):
        """Return elbow-up joint angles in degrees for a pose in millimetres."""
        psi = math.radians(pitch_deg)
        radius = math.hypot(x, y)
        # Directly above the base any base angle works; pick 0 to avoid jumping to ±180°
        q1 = 0.0 if radius < 1e-6 else math.atan2(y, x)
        wrist_radius = radius - self.A4 * math.cos(psi)
        wrist_height = z - self.D1 - self.A4 * math.sin(psi)
        elbow_cosine = (
            wrist_radius**2 + wrist_height**2 - self.A2**2 - self.A3**2
        ) / (2 * self.A2 * self.A3)
        if elbow_cosine < -1.0 - 1e-12 or elbow_cosine > 1.0 + 1e-12:
            raise ValueError("Target position is outside the robot's reachable workspace")

        elbow_cosine = self.clamp(elbow_cosine, -1.0, 1.0)
        q3 = -math.acos(elbow_cosine)
        q2_dh = math.atan2(wrist_height, wrist_radius) - math.atan2(
            self.A3 * math.sin(q3),
            self.A2 + self.A3 * math.cos(q3),
        )
        q4 = psi - q2_dh - q3
        q2 = q2_dh - math.pi / 2

        return np.degrees(np.array([q1, q2, q3, q4], dtype=float))


    # ------------------------------- Motion -----------------------------------
    def move_j(self, j1_deg, j2_deg, j3_deg, j4_deg, wait=True, tol_deg=2.0, poll_period=0.05, timeout_s=None):
        """Move to joint angles [deg]. With timeout_s, stop waiting after that many seconds (e.g. when blocked)."""
        # Limits
        desired = np.array([j1_deg, j2_deg, j3_deg, j4_deg], dtype=float)
        for i in range(4):
            desired[i] = self._check_limits_one(desired[i], i)

        # Smooth speed option
        if self.use_smooth_speed_flag:
            self.smooth_speed(desired - self.joint_angles)

        # Update internal & FK
        #self.joint_angles = desired.copy()
        #self.forward(desired)

        # Apply motor offsets for command
        cmd_deg = desired + self.joint_offsets
        for i, dxl_id in enumerate(self.motor_ids):
            self._write2(dxl_id, self.ADDR_MX_GOAL_POSITION, self.deg_to_ticks(cmd_deg[i]))

        if not wait:
            return


        # Wait until all joints within tolerance
        start = time.time()
        while True:
            if timeout_s is not None and time.time() - start > timeout_s:
                print("move_j: timeout, target not reached (blocked?)")
                break
            try:
                ja = self.read_joint_angles()
            except Exception:
                # If read fails transiently, just try again shortly.
                time.sleep(poll_period)
                continue
            if np.all(np.abs(ja - desired) < tol_deg):
                break
            time.sleep(poll_period)

    #def move_c(self, x, y, z, pitch_deg, **kwargs):
     #   j_deg = self.inverse(x, y, z, math.radians(pitch_deg))
      #  self.move_j(j_deg[0], j_deg[1], j_deg[2], j_deg[3], **kwargs)

    #def read_ee_position(self):
     #   j_a = self.read_joint_angles()
      #  return self.forward(j_a)
    # --------------------------- History / Macros ------------------------------
    def record_configuration(self):
        entry = {
            "j1": float(self.joint_angles[0]),
            "j2": float(self.joint_angles[1]),
            "j3": float(self.joint_angles[2]),
            "j4": float(self.joint_angles[3]),
            "speed": float(self.motor_speed[0]),
            "torque": float(self.motor_torque[0]),
            "gripper_open": bool(self.gripper_open_flag),
        }
        self.movement_history.append(entry)
        print(
            f"Recorded Speed: {entry['speed']:.3f}, Torque: {entry['torque']:.3f}\n"
            f"Joint Positions: {entry['j1']:.2f}, {entry['j2']:.2f}, {entry['j3']:.2f}, {entry['j4']:.2f}\n"
            f"Gripper open: {int(entry['gripper_open'])}"
        )

    def delete_last_recorded_configuration(self):
        if not self.movement_history:
            print("No last history position")
            return
        self.movement_history.pop()

    def play_configuration_history(self, dwell_after_move_s=1.0, dwell_after_grip_s=3.0):
        if not self.movement_history:
            print("No recorded configurations.")
            return
        for rec in self.movement_history:
            s = rec["speed"]
            t = rec["torque"]
            self.set_speed([s, s, s, s], overwrite_speeds=True)
            self.set_torque_limit([t, t, t, t])
            self.move_j(rec["j1"], rec["j2"], rec["j3"], rec["j4"])
            time.sleep(dwell_after_move_s)
            if self.gripper_open_flag != rec["gripper_open"]:
                self.actuate_gripper()
                time.sleep(dwell_after_grip_s)


# ------------------------------- Example usage --------------------------------
# Example 1:
# Demonstrates how to move the robot joints into desired joint positions [degrees]
# You might need to adjust the offsets in the code above to make the zero point suit your needs.
# Just comment out the code snippet, input desired joint angles in robot.move_j() and run.

def angle_between(v1, v2):
    v1_u = v1 / np.linalg.norm(v1)
    v2_u = v2 / np.linalg.norm(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

# ------------------------------- Smartie pipeline ------------------------------
# 1. Move to the camera pose and take a picture.
# 2. Detect red smarties (HSV pipeline from DectectSmarties.py) -> pixel centres.
# 3. Convert each pixel centre to base-frame [x, y, 0] with pixel_to_3D.
# 4. For each smartie: hover above it, go down to probe, go back up.
# Detection runs on the RAW frame: pixel_to_3D undistorts the points itself.

CAMERA_INDEX = 1
CAMERA_POSE = [0, 0, -61.55, -114.91]  # joint angles [deg] for taking the picture
PROBE_PITCH = -90.0             # gripper pointing straight down
# z = 0 is the robot base (frame 0, as in the DH drawing). The table is 63 mm below it.
TABLE_Z = -56.5                 # table height in the robot frame [mm]
# Measured error correction [mm]: (where the smartie really is) - (where the code says it is)
XY_CORRECTION = np.array([0.0, 0.0])
HOVER_Z = TABLE_Z + 40        # tip 60 mm above the table before/after probing
PROBE_Z = TABLE_Z + 13    # tip 30 mm above the table when probing -- lower a few mm per test
PROBE_TORQUE = [0.5, 0.5, 0.5, 0.3]  # torque limits [0..1] per joint while going down (wrist gives way first)
PROBE_TIMEOUT_S = 3.0           # stop pushing if the probe position isn't reached in time


x_offset = 4
y_offset = 1

def capture_frame(cap, flush=5):
    """Grab a fresh frame (drop buffered ones taken while the robot was moving)."""
    for _ in range(flush):
        cap.read()
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Could not read a frame from the camera")
    return frame


# Red in HSV (OpenCV hue 0-180). Red wraps around 0, so it needs two ranges.
# Measured through our camera: red smarties hue 6-11 with S ~200-255, orange hue 15-17,
# pink S 65-130 (reddish edge), paper S ~38 -> S >= 160 keeps pink and paper out.
RED_HSV_RANGES = [
    (np.array([0, 160, 40]), np.array([12, 255, 255])),
    (np.array([170, 160, 40]), np.array([180, 255, 255])),
]
SMARTIE_RADIUS_PX = 35  # smartie radius in the image at CAMERA_POSE (measured: ~70 px across)
MIN_AREA_PX = 300        # ignore specks smaller than this


def detect_red_smarties(frame):
    """Pixel centres [[u, v], ...] of red smarties in the raw frame (touching smarties are separated)."""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = np.zeros(hsv.shape[:2], np.uint8)
    for low, high in RED_HSV_RANGES:
        mask |= cv2.inRange(hsv, low, high)

    # Remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    single_area = math.pi * SMARTIE_RADIUS_PX**2
    centres = []
    for cont in contours:
        area = cv2.contourArea(cont)
        if area < MIN_AREA_PX:
            continue

        # How many smarties are in this blob, judged by its size (touching smarties form one blob)
        count = max(1, int(round(area / single_area)))

        if count == 1:
            # Centre of the outer outline (convex hull): ignores dents from highlights
            m = cv2.moments(cv2.convexHull(cont))
            centres.append([int(round(m["m10"] / m["m00"])), int(round(m["m01"] / m["m00"]))])
        else:
            # Split the blob's pixels into `count` groups and take each group's centre
            blob = np.zeros_like(mask)
            cv2.drawContours(blob, [cont], -1, 255, thickness=-1)
            ys, xs = np.nonzero(blob)
            pts = np.column_stack([xs, ys]).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
            _, _, group_centres = cv2.kmeans(pts, count, None, criteria, 5, cv2.KMEANS_PP_CENTERS)
            centres += [[int(round(u)), int(round(v))] for u, v in group_centres]
    return centres


def show_targets(frame, centres, targets):
    """Draw each detection with its world coordinates. Returns the pressed key ('p' or 'q')."""
    img = frame.copy()
    for i, ((u, v), xyz) in enumerate(zip(centres, targets)):
        cv2.circle(img, (u, v), 6, (0, 255, 0), 2)
        label = f"{i}: ({xyz[0]:.0f}, {xyz[1]:.0f})" if xyz is not None else f"{i}: unreachable"
        cv2.putText(img, label, (u + 8, v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    cv2.putText(img, "p = probe all, q = cancel", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.imshow("red smarties", img)
    while True:
        key = cv2.waitKey(0) & 0xFF
        if key in (ord('p'), ord('q')):
            return key


def find_smarties(robot, cap, K, dist):
    """Take a picture from the camera pose. Returns (frame, centres, targets); unreachable targets are None."""
    robot.move_j(*CAMERA_POSE)
    time.sleep(1.0)  # let the arm settle
    frame = capture_frame(cap)
    cv2.imwrite(os.path.join(currentdir, "imgs", "last_frame.png"), frame)  # raw frame, for tuning the colour filter

    # Use the measured angles: more accurate than the commanded ones
    T04 = robot.forward(*robot.read_joint_angles())

    centres = detect_red_smarties(frame)
    print(f"Found {len(centres)} red smarties")

    targets = []
    for u, v in centres:
        try:
            xyz = pixel_to_3D(u, v, T04, K, dist, z_plane=TABLE_Z)
            # Reachability + joint limit check for both probe heights
            for z in (HOVER_Z, PROBE_Z):
                q = robot.inverse(xyz[0], xyz[1], z+4, PROBE_PITCH)
                for i in range(4):
                    robot._check_limits_one(q[i], i)
            targets.append(xyz)
            print(f"  pixel ({u}, {v}) -> world ({xyz[0]:.1f}, {xyz[1]:.1f}, {xyz[2]:.1f}) mm")
        except ValueError as e:
            targets.append(None)
            print(f"  pixel ({u}, {v}) skipped: {e}")
    return frame, centres, targets


def probe(robot, x, y):
    """Hover above (x, y), go down to probe with reduced torque, go back up."""
    hover = robot.inverse(x, y, HOVER_Z, PROBE_PITCH)
    robot.move_j(*hover)

    normal_torque = robot.motor_torque.copy()
    robot.set_torque_limit(PROBE_TORQUE)
    try:
        robot.move_j(*robot.inverse(x, y, PROBE_Z, PROBE_PITCH), timeout_s=PROBE_TIMEOUT_S)
        time.sleep(0.5)
        robot.move_j(*hover, timeout_s=PROBE_TIMEOUT_S)
    finally:
        robot.set_torque_limit(normal_torque)  # full torque again for normal moves

def centers_to_world(centers, robot, K, dist):
    """Convert pixel centres to world coordinates using the current robot pose."""
    T04 = robot.forward(*robot.read_joint_angles())
    targets = []
    for u, v in centers:
        try:
            xyz = pixel_to_3D(u, v, T04, K, dist, z_plane=TABLE_Z)  # smarties lie on the table
            xyz[:2] += XY_CORRECTION
            targets.append(xyz)
        except ValueError as e:
            targets.append(None)
            print(f"Pixel ({u}, {v}) skipped: {e}")
    return targets

if __name__ == "__main__":
    K, dist = load_camera_calibration()
    robot = MyRobot()
    cap = cv2.VideoCapture(CAMERA_INDEX)
    # Move to the camera pose first, so the camera sees the table and can detect smarties
    robot.move_j(*CAMERA_POSE)
    time.sleep(1.0)  # let the arm stop swinging before taking the picture
    frame = capture_frame(cap)  # drops old buffered frames, then takes the picture
    cv2.imwrite(os.path.join(currentdir, "imgs", "last_frame.png"), frame)  # raw frame, for tuning
    centers = detect_red_smarties(frame)  # pixel centres of the red smarties

    world_coordinates_smarties = []

    # Show the detected smarties and their world coordinates
    if len(centers) == 0:
        print("No red smarties detected. Please place some on the table and try again.")
    else:
        world_coordinates_smarties = centers_to_world(centers, robot, K, dist)
        for (u, v), xyz in zip(centers, world_coordinates_smarties):
            if xyz is not None:
                print(f"pixel ({u}, {v}) -> world ({xyz[0]:.1f}, {xyz[1]:.1f}) mm")

    # Show the picture with the detections: p = probe, q = cancel
    if show_targets(frame, centers, world_coordinates_smarties) == ord('q'):
        print("Cancelled")
        world_coordinates_smarties = []

    # move to every detected smartie and probe it
    for world_coord in world_coordinates_smarties:
        if world_coord is not None:
            hover = robot.inverse(world_coord[0], world_coord[1], HOVER_Z, PROBE_PITCH)
            robot.move_j(*hover)
            normal_torque = robot.motor_torque.copy()
            robot.set_torque_limit(PROBE_TORQUE)
            try:
                robot.move_j(*robot.inverse(world_coord[0]+x_offset, world_coord[1]+y_offset+3, PROBE_Z, PROBE_PITCH), timeout_s=PROBE_TIMEOUT_S)
                angles_probe = robot.read_joint_angles()
                forward_probe = robot.forward(*angles_probe)
                diff = forward_probe[:3, 3] - np.array([world_coord[0], world_coord[1], PROBE_Z])
                print(f"Probe error: Δx={diff[0]:.1f} mm, Δy={diff[1]:.1f} mm, Δz={diff[2]:.1f} mm")
                print("x,y,z", forward_probe)
                time.sleep(3)
                robot.move_j(*hover, timeout_s=PROBE_TIMEOUT_S)
            finally:
                robot.set_torque_limit(normal_torque)  # full torque again for normal moves

        else:
            print("Unreachable smartie detected.")
     
    cap.release()
    cv2.destroyAllWindows()
            #robot.disable_motors()


# #Example 2:
# #Demonstrates how to move the robot joints into desired a list of joint positions [degrees]
# #You might need to adjust the offsets in the code above to make the zero point suit your needs.
# #Just comment out the code snippet, input desired joint angle sequence in robot.move_j() and run.

# '''
# if __name__ == "__main__":
# import time
# import numpy as np

# # Create robot instance
# robot = MyRobot()

# if robot.init_status:
#     try:
#         # ----------------- Define the sequence of joint angles -----------------
#         # Each row = [J1, J2, J3, J4] in DEGREES
#         # Adjust these based on your mechanical limits / workspace
#         joint_sequence = np.array([
#             [-12.02, -7.99, -95.76, 13.75],
#             [-11.75, -7.21, -93.14, 10.35],
#             [-11.24, -6.60, -90.47, 7.07],
#             [-10.49, -6.14, -87.85, 3.99],
#             [-9.33, -5.47, -85.76, 1.24],
#             [-7.69, -5.20, -83.63, -1.17],
#             [-5.96, -5.05, -81.82, -3.13],
#             [-4.25, -4.67, -80.78, -4.55],
#             [-2.04, -4.61, -79.95, -5.44],
#             [-0.00, -4.60, -79.66, -5.74],
#             [2.04, -4.61, -79.95, -5.44],
#             [4.25, -4.67, -80.78, -4.55],
#             [5.96, -5.05, -81.82, -3.13],
#             [7.69, -5.20, -83.63, -1.17],
#             [9.33, -5.47, -85.76, 1.24],
#             [10.49, -6.14, -87.85, 3.99],
#             [11.24, -6.60, -90.47, 7.07],
#             [11.75, -7.21, -93.14, 10.35],
#             [12.02, -7.99, -95.76, 13.75],
#             [11.75, -8.85, -98.32, 17.17],
#             [11.24, -9.82, -100.68, 20.50],
#             [10.49, -10.87, -102.80, 23.67],
#             [9.33, -11.60, -105.02, 26.62],
#             [7.69, -12.54, -106.61, 29.14],
#             [5.96, -13.37, -107.84, 31.21],
#             [4.25, -13.76, -109.06, 32.82],
#             [2.04, -14.16, -109.60, 33.76],
#             [-0.00, -14.30, -109.78, 34.08],
#             [-2.04, -14.16, -109.60, 33.76],
#             [-4.25, -13.76, -109.06, 32.82],
#             [-5.96, -13.37, -107.84, 31.21],
#             [-7.69, -12.54, -106.61, 29.14],
#             [-9.33, -11.60, -105.02, 26.62],
#             [-10.49, -10.87, -102.80, 23.67],
#             [-11.24, -9.82, -100.68, 20.50],
#             [-11.75, -8.85, -98.32, 17.17],
#             [-12.02, -7.99, -95.76, 13.75]
#             # return home
#         ])

#         # ----------------- Run the motion sequence -----------------
#         robot.enable_motors()
#         print("\nStarting joint sequence...")
#         time.sleep(1.0)

#         #Number of loops
#         loops = 5

#         for loop in range(loops):
#             print(f"\n🔁 Loop {loop + 1} of {loops} ----------------------------")
#             for idx, pose in enumerate(joint_sequence):
#                 print(f"\n➡️  Moving to pose {idx+1}: {pose}°")
#                 robot.move_j(*pose)
#                 robot.record_configuration()  # save to history
#                 # Read and print measured joint positions
#                 measured = robot.read_joint_angles()
#                 print("Measured (deg):", np.round(measured, 2))
#                 time.sleep(0)  # pause between poses



#     except Exception as e:
#         print("Error during sequence:", e)

#     finally:
#         robot.disable_motors()
#         print("\nMotors disabled. Program finished.")
# 