import argparse
import asyncio
import ctypes
import functools
import http.server
import json
import math
import queue
import socketserver
import sys
import threading
import time
from pathlib import Path

import glfw
import numpy as np
import openvr
from OpenGL import GL
import websockets

try:
    from openspace import Api as OpenSpaceApi
except ImportError:
    OpenSpaceApi = None

try:
    from windows_capture import WindowsCapture, Frame, InternalCaptureControl
except ImportError:
    WindowsCapture = None

try:
    import mss
except ImportError:
    mss = None


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "index_bindings.json"

DEFAULT_CONFIG = {
    "flight_mode": "body",  # "body" (orbital/body heading) or "gaze" (fly where you look)
    "walk_scale": 1.0,      # 1.0 = true 1:1 meters in room-scale playspace
    "invert_hmd": {
        "yaw": False,
        "pitch": False,
        "roll": False,
    },
    "mappings": [
        # --- Google Earth VR Inspired Flight & Orbit ---
        {
            "id": "m1",
            "source": "Right_Trigger",
            "mode": "continuous",
            "deadzone": 0.05,
            "min": 0.0,
            "max": 3.5,
            "lua": "openspace.navigation.addTruckMovement({VALUE});",
        },
        {
            "id": "m2",
            "source": "Left_Trigger",
            "mode": "continuous",
            "deadzone": 0.05,
            "min": 0.0,
            "max": -3.5,
            "lua": "openspace.navigation.addTruckMovement({VALUE});",
        },
        {
            "id": "m3",
            "source": "Left_Stick_Y",
            "mode": "continuous",
            "deadzone": 0.12,
            "min": -3.0,
            "max": 3.0,
            "lua": "openspace.navigation.addTruckMovement({VALUE});",
        },
        {
            "id": "m4",
            "source": "Left_Stick_X",
            "mode": "continuous",
            "deadzone": 0.12,
            "min": -2.0,
            "max": 2.0,
            "lua": "openspace.navigation.addGlobalRoll({VALUE});",
        },
        {
            "id": "m5",
            "source": "Right_Grip",
            "mode": "continuous",
            "deadzone": 0.10,
            "min": 0.0,
            "max": 2.0,
            "lua": "openspace.navigation.addGlobalRotation({VALUE}, 0.0);",
        },
        {
            "id": "m6",
            "source": "Left_Grip",
            "mode": "continuous",
            "deadzone": 0.10,
            "min": 0.0,
            "max": -2.0,
            "lua": "openspace.navigation.addGlobalRotation({VALUE}, 0.0);",
        },
        # --- Time Control (Right Stick X + Trackpad + B) ---
        {
            "id": "m7",
            "source": "Right_Stick_X",
            "mode": "step_repeat",
            "deadzone": 0.55,
            "min": -1.0,
            "max": 1.0,
            "lua": "if {VALUE} > 0 then openspace.time.interpolateNextDeltaTimeStep() else openspace.time.interpolatePreviousDeltaTimeStep() end",
        },
        {
            "id": "m8",
            "source": "Right_Stick_Y",
            "mode": "continuous",
            "deadzone": 0.12,
            "min": -2.0,
            "max": 2.0,
            "lua": "openspace.navigation.addGlobalRotation(0.0, {VALUE});",
        },
        {
            "id": "m9",
            "source": "Right_Trackpad_Click",
            "mode": "on_press",
            "deadzone": 0.0,
            "min": 0.0,
            "max": 1.0,
            "lua": "openspace.time.interpolateTogglePause();",
        },
        {
            "id": "m10",
            "source": "Right_Button_B",
            "mode": "on_press",
            "deadzone": 0.0,
            "min": 0.0,
            "max": 1.0,
            "lua": "openspace.time.interpolateDeltaTime(1.0, 0.5);",
        },
        # --- VR Comfort & Recentering ---
        {
            "id": "m11",
            "source": "Left_Button_A",
            "mode": "on_press",
            "deadzone": 0.0,
            "min": 0.0,
            "max": 1.0,
            "lua": "__BRIDGE_RECENTER__",
        },
        {
            "id": "m12",
            "source": "Left_Button_B",
            "mode": "on_press",
            "deadzone": 0.0,
            "min": 0.0,
            "max": 1.0,
            "lua": "__BRIDGE_TOGGLE_FLIGHT_MODE__",
        },
        {
            "id": "m13",
            "source": "Right_Button_A",
            "mode": "on_press",
            "deadzone": 0.0,
            "min": 0.0,
            "max": 1.0,
            "lua": "openspace.navigation.retargetAnchor();",
        },
    ],
}


# --- Quaternion & 3D Math Helpers ---
def matrix34_to_pose(m):
    """Extract ((x, y, z), (w, x, y, z)) from OpenVR 3x4 HmdMatrix34_t."""
    pos = (float(m[0][3]), float(m[1][3]), float(m[2][3]))
    r00, r01, r02 = m[0][0], m[0][1], m[0][2]
    r10, r11, r12 = m[1][0], m[1][1], m[1][2]
    r20, r21, r22 = m[2][0], m[2][1], m[2][2]

    trace = r00 + r11 + r22
    if trace > 0.0:
        s = 0.5 / math.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (r21 - r12) * s
        y = (r02 - r20) * s
        z = (r10 - r01) * s
    elif r00 > r11 and r00 > r22:
        s = 2.0 * math.sqrt(1.0 + r00 - r11 - r22)
        w = (r21 - r12) / s
        x = 0.25 * s
        y = (r01 + r10) / s
        z = (r02 + r20) / s
    elif r11 > r22:
        s = 2.0 * math.sqrt(1.0 + r11 - r00 - r22)
        w = (r02 - r20) / s
        x = (r01 + r10) / s
        y = 0.25 * s
        z = (r12 + r21) / s
    else:
        s = 2.0 * math.sqrt(1.0 + r22 - r00 - r11)
        w = (r10 - r01) / s
        x = (r02 + r20) / s
        y = (r12 + r21) / s
        z = 0.25 * s

    norm = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
    return pos, (w / norm, x / norm, y / norm, z / norm)


def quat_inv(q):
    return (q[0], -q[1], -q[2], -q[3])


def quat_mul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return (
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    )


def quat_rotate_vec(q, v):
    """Rotate 3D vector v by unit quaternion q."""
    qv = (0.0, v[0], v[1], v[2])
    res = quat_mul(quat_mul(q, qv), quat_inv(q))
    return (res[1], res[2], res[3])


def yaw_only_quat(yaw_rad):
    """Unit quaternion representing rotation around +Y (up) by yaw_rad."""
    half = 0.5 * yaw_rad
    return (math.cos(half), 0.0, math.sin(half), 0.0)


def quat_to_euler_rad(q):
    """Return (pitch, yaw, roll) in radians for OpenVR (+X right, +Y up, -Z forward)."""
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + x * x))
    sinp = max(-1.0, min(1.0, 2.0 * (w * x - z * y)))
    pitch = math.asin(sinp)
    roll = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (x * x + z * z))
    return pitch, yaw, roll


def quat_to_glm_xyz(q):
    """
    Decompose unit quaternion q into (rx, ry, rz) matching GLM's
    glm::quat(glm::vec3(rx, ry, rz)) = Q_x(rx) * Q_y(ry) * Q_z(rz),
    which OpenSpace RenderEngine.GlobalRotation uses for [Pitch, Yaw, Roll].
    """
    w, x, y, z = q
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r01 = 2.0 * (x * y - w * z)
    r02 = max(-1.0, min(1.0, 2.0 * (x * z + w * y)))
    r12 = 2.0 * (y * z - w * x)
    r22 = 1.0 - 2.0 * (x * x + y * y)

    ry = math.asin(r02)
    if abs(r02) < 0.99999:
        rx = math.atan2(-r12, r22)
        rz = math.atan2(-r01, r00)
    else:
        r11 = 1.0 - 2.0 * (x * x + z * z)
        r21 = 2.0 * (y * z + w * x)
        rx = math.atan2(r21, r11)
        rz = 0.0
    return rx, ry, rz


def compute_eye_texture_bounds(vr_sys, eye, rendered_hfov_deg=103.0, rendered_vfov_deg=104.0):
    left_tan, right_tan, top_tan, bottom_tan = vr_sys.getProjectionRaw(eye)
    h_tan_half = math.tan(math.radians(rendered_hfov_deg * 0.5))
    v_tan_half = math.tan(math.radians(rendered_vfov_deg * 0.5))

    u_min = max(0.0, min(1.0, 0.5 + (left_tan / (2.0 * h_tan_half))))
    u_max = max(0.0, min(1.0, 0.5 + (right_tan / (2.0 * h_tan_half))))
    v_top = max(0.0, min(1.0, 0.5 - (top_tan / (2.0 * v_tan_half))))
    v_bot = max(0.0, min(1.0, 0.5 - (bottom_tan / (2.0 * v_tan_half))))
    return openvr.VRTextureBounds_t(u_min, v_top, u_max, v_bot)


def build_local_translate_lua(dx, dy, dz):
    """
    Build zero-inertia Lua script moving NavigationState.Position in body-local space:
    dx = right (m), dy = up (m), dz = forward (m).
    """
    return (
        "local ns = openspace.navigation.getNavigationState();"
        "if ns and ns.Position and ns.Up then "
        "local px,py,pz = ns.Position[1],ns.Position[2],ns.Position[3];"
        "local d = math.sqrt(px*px+py*py+pz*pz);"
        "if d > 0.001 then "
        "local fx,fy,fz = -px/d,-py/d,-pz/d;"
        "local ux,uy,uz = ns.Up[1],ns.Up[2],ns.Up[3];"
        "local rx,ry,rz = fy*uz-fz*uy, fz*ux-fx*uz, fx*uy-fy*ux;"
        "local rl = math.sqrt(rx*rx+ry*ry+rz*rz);"
        "if rl > 1e-6 then "
        "rx,ry,rz = rx/rl,ry/rl,rz/rl;"
        "ux,uy,uz = ry*fz-rz*fy, rz*fx-rx*fz, rx*fy-ry*fx;"
        f"ns.Position[1] = px + rx*({dx:.6f}) + ux*({dy:.6f}) + fx*({dz:.6f});"
        f"ns.Position[2] = py + ry*({dx:.6f}) + uy*({dy:.6f}) + fy*({dz:.6f});"
        f"ns.Position[3] = pz + rz*({dx:.6f}) + uz*({dy:.6f}) + fz*({dz:.6f});"
        "openspace.navigation.setNavigationState(ns);"
        "end end end"
    )


def build_gaze_thrust_lua(thrust_val, gx, gy, gz):
    """
    Fly along the HMD gaze vector (gx right, gy up, gz forward in body space)
    scaled by distanceToFocus so speed matches OpenSpace's altitude scaling.
    """
    return (
        "local ns = openspace.navigation.getNavigationState();"
        "if ns and ns.Position and ns.Up then "
        "local px,py,pz = ns.Position[1],ns.Position[2],ns.Position[3];"
        "local d = math.sqrt(px*px+py*py+pz*pz);"
        "if d > 0.001 then "
        "local fx,fy,fz = -px/d,-py/d,-pz/d;"
        "local ux,uy,uz = ns.Up[1],ns.Up[2],ns.Up[3];"
        "local rx,ry,rz = fy*uz-fz*uy, fz*ux-fx*uz, fx*uy-fy*ux;"
        "local rl = math.sqrt(rx*rx+ry*ry+rz*rz);"
        "if rl > 1e-6 then "
        "rx,ry,rz = rx/rl,ry/rl,rz/rl;"
        "ux,uy,uz = ry*fz-rz*fy, rz*fx-rx*fz, rx*fy-ry*fx;"
        f"local step = ({thrust_val:.5f}) * 0.012 * d;"
        f"local dx,dy,dz = step*({gx:.5f}), step*({gy:.5f}), step*({gz:.5f});"
        "ns.Position[1] = px + rx*dx + ux*dy + fx*dz;"
        "ns.Position[2] = py + ry*dx + uy*dy + fy*dz;"
        "ns.Position[3] = pz + rz*dx + uz*dy + fz*dz;"
        "openspace.navigation.setNavigationState(ns);"
        "end end end"
    )


# --- Per-Window Stereo Frame Grabber ---
class StereoWindowGrabber:
    def __init__(self, window_title: str):
        self.window_title = window_title
        self._lock = threading.Lock()
        self._latest_frame = None
        self._active = False
        self._stop_event = threading.Event()
        self._control = None

    def start(self):
        if sys.platform.startswith("win") and WindowsCapture is not None:
            threading.Thread(target=self._wgc_supervisor, daemon=True).start()
            print(f"[Capture] Windows Graphics Capture enabled for window '{self.window_title}'.")
        elif mss is not None:
            threading.Thread(target=self._mss_fallback_loop, daemon=True).start()
            print(f"[Capture] Using mss window-region capture for '{self.window_title}'.")
        else:
            print("[Capture] Warning: Neither windows-capture nor mss is available.")

    def _wgc_supervisor(self):
        while not self._stop_event.is_set():
            if self._active:
                time.sleep(0.5)
                continue

            try:
                capture = WindowsCapture(
                    cursor_capture=False,
                    draw_border=False,
                    monitor_index=None,
                    window_name=self.window_title,
                )

                @capture.event
                def on_frame_arrived(frame: Frame, capture_control: InternalCaptureControl):
                    if self._stop_event.is_set():
                        capture_control.stop()
                        return
                    buf = frame.frame_buffer
                    if buf is not None and buf.size > 0:
                        with self._lock:
                            self._latest_frame = buf.copy()

                @capture.event
                def on_closed():
                    self._active = False
                    print(f"[Capture] Window '{self.window_title}' closed; waiting to reattach...")

                self._control = capture.start_free_threaded()
                self._active = True
                print(f"[Capture] Locked onto '{self.window_title}' window buffer.")
            except Exception:
                self._active = False
                time.sleep(1.0)

    def _mss_fallback_loop(self):
        with mss.mss() as sct:
            while not self._stop_event.is_set():
                rect = self._find_linux_rect()
                if rect:
                    try:
                        grab = sct.grab(rect)
                        arr = np.frombuffer(grab.bgra, dtype=np.uint8).reshape((grab.height, grab.width, 4))
                        with self._lock:
                            self._latest_frame = arr
                        self._active = True
                    except Exception:
                        self._active = False
                time.sleep(1.0 / 90.0)

    def _find_linux_rect(self):
        try:
            import subprocess

            out = subprocess.check_output(["xwininfo", "-name", self.window_title], stderr=subprocess.DEVNULL, text=True)
            vals = {}
            for line in out.splitlines():
                line = line.strip()
                if line.startswith("Absolute upper-left X:"):
                    vals["left"] = int(line.split(":")[1])
                elif line.startswith("Absolute upper-left Y:"):
                    vals["top"] = int(line.split(":")[1])
                elif line.startswith("Width:"):
                    vals["width"] = int(line.split(":")[1])
                elif line.startswith("Height:"):
                    vals["height"] = int(line.split(":")[1])
            if len(vals) == 4 and vals["width"] > 100 and vals["height"] > 100:
                return vals
        except Exception:
            pass
        return None

    def get_frame(self):
        with self._lock:
            frame = self._latest_frame
            self._latest_frame = None
            return frame

    def stop(self):
        self._stop_event.set()
        if self._control is not None:
            try:
                self._control.stop()
            except Exception:
                pass


# --- Bridge State ---
class BridgeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.config = self.load_config()
        self.openspace_connected = False
        self.steamvr_connected = False
        self.lua_queue = queue.Queue(maxsize=32)

        # Latest-wins zero-inertia HMD state
        self.latest_global_rot = None     # [pitch, yaw, roll] in radians for RenderEngine.GlobalRotation
        self.pending_walk_delta = [0.0, 0.0, 0.0]  # accumulated [right, up, forward] meters
        self.q_ref = None
        self.pos_origin = None
        self.pos_prev = None
        self.recenter_requested = True

        # Controller state tracking
        self.prev_buttons = {}
        self.step_timers = {}
        self.telemetry = {
            "hmd_euler_deg": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "hmd_pos_m": {"x": 0.0, "y": 0.0, "z": 0.0},
            "flight_mode": self.config["flight_mode"],
            "walk_scale": self.config["walk_scale"],
            "last_signal": {"source": "None", "value": 0.0},
            "openspace_connected": False,
            "steamvr_connected": False,
            "capture_rect": None,
        }

    def load_config(self):
        if CONFIG_FILE.exists():
            try:
                with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                merged = json.loads(json.dumps(DEFAULT_CONFIG))
                merged.update(data)
                return merged
            except Exception as exc:
                print(f"[Config] Failed to load {CONFIG_FILE}, using defaults: {exc}")
        return json.loads(json.dumps(DEFAULT_CONFIG))

    def save_config(self):
        with self.lock:
            with open(CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(self.config, f, indent=2)

    def enqueue_lua(self, script):
        if not script:
            return
        if self.lua_queue.full():
            try:
                self.lua_queue.get_nowait()
            except queue.Empty:
                pass
        self.lua_queue.put_nowait(script)


# --- Valve Index Input Reader ---
def read_controller_inputs(vr_sys):
    inputs = {}
    roles = {
        "Left": vr_sys.getTrackedDeviceIndexForControllerRole(openvr.TrackedControllerRole_LeftHand),
        "Right": vr_sys.getTrackedDeviceIndexForControllerRole(openvr.TrackedControllerRole_RightHand),
    }

    for side, idx in roles.items():
        if idx == openvr.k_unTrackedDeviceIndexInvalid:
            continue
        ok, state = vr_sys.getControllerState(idx)
        if not ok:
            continue

        inputs[f"{side}_Stick_X"] = float(state.rAxis[0].x)
        inputs[f"{side}_Stick_Y"] = float(state.rAxis[0].y)
        inputs[f"{side}_Trigger"] = float(state.rAxis[1].x)
        inputs[f"{side}_Grip"] = float(state.rAxis[2].x)
        inputs[f"{side}_Trackpad_X"] = float(state.rAxis[3].x)
        inputs[f"{side}_Trackpad_Y"] = float(state.rAxis[3].y)

        pressed = state.ulButtonPressed
        inputs[f"{side}_Button_A"] = 1.0 if (pressed & (1 << openvr.k_EButton_A)) else 0.0
        inputs[f"{side}_Button_B"] = 1.0 if (pressed & (1 << openvr.k_EButton_ApplicationMenu)) else 0.0
        inputs[f"{side}_Grip_Click"] = 1.0 if (pressed & (1 << openvr.k_EButton_Grip)) else 0.0
        inputs[f"{side}_Trackpad_Click"] = 1.0 if (pressed & (1 << openvr.k_EButton_SteamVR_Touchpad)) else 0.0
        inputs[f"{side}_Trigger_Click"] = 1.0 if (pressed & (1 << openvr.k_EButton_SteamVR_Trigger)) else 0.0

    return inputs


# --- SteamVR Compositor + Tracking Thread ---
def vr_loop(bridge: BridgeState, window_title: str, disable_compositor: bool):
    if not glfw.init():
        print("[VR] Failed to initialize GLFW.")
        return

    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    gl_window = glfw.create_window(64, 64, "OpenSpaceVR_GLContext", None, None)
    if not gl_window:
        print("[VR] Failed to create hidden OpenGL context.")
        glfw.terminate()
        return
    glfw.make_context_current(gl_window)

    try:
        vr_sys = openvr.init(openvr.VRApplication_Scene)
        compositor = openvr.VRCompositor()
        bridge.steamvr_connected = True
        bridge.telemetry["steamvr_connected"] = True
        print("[VR] Connected to SteamVR runtime (Valve Index ready).")
    except Exception as exc:
        print(f"[VR] Could not initialize SteamVR: {exc}")
        glfw.terminate()
        return

    grabber = None
    if not disable_compositor:
        grabber = StereoWindowGrabber(window_title)
        grabber.start()

    left_tex, right_tex = GL.glGenTextures(2)
    for tex in (left_tex, right_tex):
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)

    tex_dims = [0, 0]
    poses_t = openvr.TrackedDevicePose_t * openvr.k_unMaxTrackedDeviceCount
    poses = poses_t()

    bounds_left = compute_eye_texture_bounds(vr_sys, openvr.Eye_Left, 103.0, 104.0)
    bounds_right = compute_eye_texture_bounds(vr_sys, openvr.Eye_Right, 103.0, 104.0)

    tex_left_desc = openvr.Texture_t()
    tex_left_desc.handle = ctypes.cast(int(left_tex), ctypes.c_void_p)
    tex_left_desc.eType = openvr.TextureType_OpenGL
    tex_left_desc.eColorSpace = openvr.ColorSpace_Gamma

    tex_right_desc = openvr.Texture_t()
    tex_right_desc.handle = ctypes.cast(int(right_tex), ctypes.c_void_p)
    tex_right_desc.eType = openvr.TextureType_OpenGL
    tex_right_desc.eColorSpace = openvr.ColorSpace_Gamma

    try:
        while not glfw.window_should_close(gl_window):
            now = time.perf_counter()
            compositor.waitGetPoses(poses, None)

            # 1. Upload latest per-window stereo frame and submit to SteamVR Compositor
            if grabber is not None:
                frame = grabber.get_frame()
                if frame is not None and frame.shape[1] >= 200 and frame.shape[0] >= 100:
                    h, w = frame.shape[:2]
                    eye_w = w // 2
                    bridge.telemetry["capture_rect"] = {"left": 0, "top": 0, "width": int(w), "height": int(h)}

                    left_eye = np.ascontiguousarray(frame[:, :eye_w, :])
                    right_eye = np.ascontiguousarray(frame[:, eye_w : eye_w * 2, :])

                    if tex_dims[0] != eye_w or tex_dims[1] != h:
                        GL.glBindTexture(GL.GL_TEXTURE_2D, left_tex)
                        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, eye_w, h, 0, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, left_eye)
                        GL.glBindTexture(GL.GL_TEXTURE_2D, right_tex)
                        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, eye_w, h, 0, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, right_eye)
                        tex_dims = [eye_w, h]
                    else:
                        GL.glBindTexture(GL.GL_TEXTURE_2D, left_tex)
                        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, eye_w, h, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, left_eye)
                        GL.glBindTexture(GL.GL_TEXTURE_2D, right_tex)
                        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, eye_w, h, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, right_eye)

                if tex_dims[0] > 0:
                    compositor.submit(openvr.Eye_Left, tex_left_desc, bounds_left)
                    compositor.submit(openvr.Eye_Right, tex_right_desc, bounds_right)

            # 2. Process 6DoF HMD Pose (Zero-Inertia Direct Rotation + Room-Scale Translation)
            gaze_dir_body = (0.0, 0.0, 1.0)  # (right, up, forward) in body frame
            hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]

            if hmd_pose.bPoseIsValid:
                pos_curr, q_curr = matrix34_to_pose(hmd_pose.mDeviceToAbsoluteTracking)
                _, yaw_raw, _ = quat_to_euler_rad(q_curr)

                with bridge.lock:
                    inv = dict(bridge.config.get("invert_hmd", {}))
                    walk_scale = float(bridge.config.get("walk_scale", 1.0))
                    recenter = bridge.recenter_requested
                    bridge.recenter_requested = False

                if recenter or bridge.q_ref is None or bridge.pos_prev is None:
                    # Recenter forward yaw & room origin while keeping gravity horizon level
                    bridge.q_ref = yaw_only_quat(yaw_raw)
                    bridge.pos_origin = pos_curr
                    bridge.pos_prev = pos_curr

                # Head orientation relative to recentered forward (+X right, +Y up, -Z forward)
                q_offset = quat_mul(quat_inv(bridge.q_ref), q_curr)
                pitch_off, yaw_off, roll_off = quat_to_euler_rad(q_offset)

                # Gaze vector in body frame (right, up, forward)
                gx_vr, gy_vr, gz_vr = quat_rotate_vec(q_offset, (0.0, 0.0, -1.0))
                gaze_dir_body = (gx_vr, gy_vr, -gz_vr)

                # Room-scale delta since last VR frame rotated into recentered body frame
                dp_world = (
                    pos_curr[0] - bridge.pos_prev[0],
                    pos_curr[1] - bridge.pos_prev[1],
                    pos_curr[2] - bridge.pos_prev[2],
                )
                bridge.pos_prev = pos_curr
                dx_vr, dy_vr, dz_vr = quat_rotate_vec(quat_inv(bridge.q_ref), dp_world)

                # Relative room position for telemetry
                rel_world = (
                    pos_curr[0] - bridge.pos_origin[0],
                    pos_curr[1] - bridge.pos_origin[1],
                    pos_curr[2] - bridge.pos_origin[2],
                )
                rx_m, ry_m, rz_m = quat_rotate_vec(quat_inv(bridge.q_ref), rel_world)

                # OpenSpace RenderEngine.GlobalRotation rotates the scene relative to the camera
                # using GLM's [X=Pitch, Y=Yaw, Z=Roll] Euler order: decompose quat_inv(q_offset) directly.
                q_view = quat_inv(q_offset)
                os_pitch, os_yaw, os_roll = quat_to_glm_xyz(q_view)

                if inv.get("pitch", False):
                    os_pitch = -os_pitch
                if inv.get("yaw", False):
                    os_yaw = -os_yaw
                if inv.get("roll", False):
                    os_roll = -os_roll

                with bridge.lock:
                    bridge.latest_global_rot = [os_pitch, os_yaw, os_roll]
                    if walk_scale > 0.0:
                        bridge.pending_walk_delta[0] += dx_vr * walk_scale
                        bridge.pending_walk_delta[1] += dy_vr * walk_scale
                        bridge.pending_walk_delta[2] += (-dz_vr) * walk_scale

                    bridge.telemetry["hmd_euler_deg"] = {
                        "yaw": round(math.degrees(-yaw_off), 1),
                        "pitch": round(math.degrees(pitch_off), 1),
                        "roll": round(math.degrees(-roll_off), 1),
                    }
                    bridge.telemetry["hmd_pos_m"] = {
                        "x": round(rx_m, 2),
                        "y": round(ry_m, 2),
                        "z": round(-rz_m, 2),
                    }

            # 3. Process Valve Index Controllers (with Orbital Inertia Preserved)
            lua_cmds = []
            ctrl_inputs = read_controller_inputs(vr_sys)
            with bridge.lock:
                mappings = list(bridge.config["mappings"])
                flight_mode = bridge.config.get("flight_mode", "body")

            for m in mappings:
                src = m.get("source", "")
                if src not in ctrl_inputs:
                    continue

                raw = ctrl_inputs[src]
                dz = float(m.get("deadzone", 0.0))
                mode = m.get("mode", "continuous")
                min_v = float(m.get("min", 0.0))
                max_v = float(m.get("max", 1.0))
                lua_tpl = m.get("lua", "")
                key = m.get("id", src)

                if abs(raw) > max(0.2, dz):
                    bridge.telemetry["last_signal"] = {"source": src, "value": round(raw, 3)}

                if mode == "on_press":
                    is_down = raw >= 0.5
                    was_down = bridge.prev_buttons.get(key, False)
                    bridge.prev_buttons[key] = is_down
                    if is_down and not was_down:
                        if lua_tpl == "__BRIDGE_RECENTER__":
                            with bridge.lock:
                                bridge.recenter_requested = True
                        elif lua_tpl == "__BRIDGE_TOGGLE_FLIGHT_MODE__":
                            with bridge.lock:
                                bridge.config["flight_mode"] = (
                                    "gaze" if bridge.config["flight_mode"] == "body" else "body"
                                )
                                bridge.telemetry["flight_mode"] = bridge.config["flight_mode"]
                        else:
                            lua_cmds.append(lua_tpl.replace("{VALUE}", f"{max_v:.4f}"))

                elif mode == "step_repeat":
                    if abs(raw) >= dz:
                        last_t = bridge.step_timers.get(key, 0.0)
                        if now - last_t >= 0.30:
                            bridge.step_timers[key] = now
                            val = max_v if raw > 0 else min_v
                            lua_cmds.append(lua_tpl.replace("{VALUE}", f"{val:.4f}"))
                    else:
                        bridge.step_timers[key] = 0.0

                else:  # continuous
                    if abs(raw) < dz:
                        continue
                    sign = 1.0 if raw >= 0 else -1.0
                    norm = (abs(raw) - dz) / max(1e-5, (1.0 - dz))
                    if "Stick" in src or "Trackpad" in src:
                        signed_norm = sign * norm
                        scaled = ((signed_norm + 1.0) * 0.5) * (max_v - min_v) + min_v
                    else:
                        scaled = min_v + norm * (max_v - min_v)

                    if abs(scaled) > 1e-4:
                        if flight_mode == "gaze" and "addTruckMovement" in lua_tpl:
                            lua_cmds.append(
                                build_gaze_thrust_lua(
                                    scaled,
                                    gaze_dir_body[0],
                                    gaze_dir_body[1],
                                    gaze_dir_body[2],
                                )
                            )
                        else:
                            lua_cmds.append(lua_tpl.replace("{VALUE}", f"{scaled:.5f}"))

            if lua_cmds:
                bridge.enqueue_lua(" ".join(lua_cmds))

    finally:
        if grabber is not None:
            grabber.stop()
        openvr.shutdown()
        glfw.terminate()


# --- Official OpenSpace 0.22 Async Socket API Loop ---
async def openspace_api_loop(bridge: BridgeState, host: str, port: int):
    """Stream 1:1 HMD properties + room-scale translation + controller Lua to OpenSpace 0.22."""
    if OpenSpaceApi is None:
        print("[OpenSpace] Error: 'openspace-api' package not installed. Run: pip install openspace-api")
        return

    while True:
        disconnected_event = asyncio.Event()
        api = OpenSpaceApi(host, port)

        async def on_connect():
            bridge.openspace_connected = True
            bridge.telemetry["openspace_connected"] = True
            print(f"[OpenSpace] Connected via openspace-api to {host}:{port}")
            try:
                api.setProperty("NavigationHandler.OrbitalNavigator.Friction.RotationalFriction", True)
                api.setProperty("RenderEngine.GlobalRotation", [0.0, 0.0, 0.0])
            except Exception:
                pass

        def on_disconnect():
            if bridge.openspace_connected:
                print("[OpenSpace] Disconnected. Reconnecting...")
            bridge.openspace_connected = False
            bridge.telemetry["openspace_connected"] = False
            disconnected_event.set()

        api.onConnect(on_connect)
        api.onDisconnect(on_disconnect)

        last_sent_rot = None
        try:
            await api.connect()
            while not disconnected_event.is_set():
                did_work = False

                # 1. Latest-wins direct 1:1 HMD view rotation (zero inertia, zero queue lag)
                with bridge.lock:
                    rot = bridge.latest_global_rot
                    bridge.latest_global_rot = None
                    dx, dy, dz = bridge.pending_walk_delta
                    if abs(dx) > 0.0005 or abs(dy) > 0.0005 or abs(dz) > 0.0005:
                        bridge.pending_walk_delta = [0.0, 0.0, 0.0]
                    else:
                        dx = dy = dz = 0.0

                if rot is not None and rot != last_sent_rot:
                    api.setProperty("RenderEngine.GlobalRotation", rot)
                    last_sent_rot = rot
                    did_work = True

                # 2. Zero-inertia 6DoF room-scale translation step
                if dx != 0.0 or dy != 0.0 or dz != 0.0:
                    await api.executeLuaScript(
                        build_local_translate_lua(dx, dy, dz),
                        getReturnValue=False,
                        shouldBeSynchronized=True,
                    )
                    did_work = True

                # 3. Inertial hand-controller Lua commands
                try:
                    script = bridge.lua_queue.get_nowait()
                    await api.executeLuaScript(script, getReturnValue=False, shouldBeSynchronized=True)
                    did_work = True
                except queue.Empty:
                    pass

                if not did_work:
                    await asyncio.sleep(0.004)
        except Exception:
            bridge.openspace_connected = False
            bridge.telemetry["openspace_connected"] = False
            try:
                api.disconnect()
            except Exception:
                pass
            await asyncio.sleep(1.5)


# --- Local UI WebSocket & HTTP Server ---
async def ui_ws_handler(websocket, bridge: BridgeState):
    async def sender():
        while True:
            with bridge.lock:
                snapshot = {
                    "type": "telemetry",
                    "telemetry": bridge.telemetry,
                    "config": bridge.config,
                }
            await websocket.send(json.dumps(snapshot))
            await asyncio.sleep(0.1)

    send_task = asyncio.create_task(sender())
    try:
        async for raw_msg in websocket:
            msg = json.loads(raw_msg)
            action = msg.get("action")
            if action == "update_config":
                with bridge.lock:
                    bridge.config.update(msg.get("config", {}))
                    bridge.telemetry["flight_mode"] = bridge.config.get("flight_mode", "body")
                    bridge.telemetry["walk_scale"] = bridge.config.get("walk_scale", 1.0)
            elif action == "save_config":
                bridge.save_config()
            elif action == "recenter":
                with bridge.lock:
                    bridge.recenter_requested = True
            elif action == "test_lua":
                bridge.enqueue_lua(msg.get("lua", ""))
    except Exception:
        pass
    finally:
        send_task.cancel()


def start_http_server(port: int):
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(SCRIPT_DIR))

    class QuietHTTPServer(socketserver.ThreadingTCPServer):
        allow_reuse_address = True

    httpd = QuietHTTPServer(("127.0.0.1", port), handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    print(f"[UI] Control Mapper available at http://localhost:{port}/index_mapper.html")


async def main_async(args):
    bridge = BridgeState()
    start_http_server(args.http_port)

    threading.Thread(
        target=vr_loop,
        args=(bridge, args.window_title, args.no_compositor),
        daemon=True,
    ).start()

    ws_server = await websockets.serve(
        functools.partial(ui_ws_handler, bridge=bridge),
        "127.0.0.1",
        args.ui_ws_port,
    )
    print(f"[UI] Bridge WebSocket listening on ws://127.0.0.1:{args.ui_ws_port}")

    await asyncio.gather(
        openspace_api_loop(bridge, args.os_host, args.os_port),
        ws_server.wait_closed(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valve Index + SteamVR Bridge for OpenSpace 0.22")
    parser.add_argument("--os-host", default="localhost", help="OpenSpace TCP API host")
    parser.add_argument("--os-port", type=int, default=4681, help="OpenSpace TCP API port (default: 4681)")
    parser.add_argument("--http-port", type=int, default=8080, help="Local HTML mapper HTTP port")
    parser.add_argument("--ui-ws-port", type=int, default=8081, help="Local UI telemetry WebSocket port")
    parser.add_argument("--window-title", default="OpenSpaceVR", help="SGCT stereo window name to capture")
    parser.add_argument("--no-compositor", action="store_true", help="Disable SteamVR frame submission (controls only)")
    asyncio.run(main_async(parser.parse_args()))