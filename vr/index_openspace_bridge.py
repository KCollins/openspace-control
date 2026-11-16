import argparse
import asyncio
import ctypes
import functools
import http.server
import json
import math
import os
import queue
import socketserver
import sys
import threading
import time
from pathlib import Path

import glfw
import mss
import numpy as np
import openvr
from OpenGL import GL
import websockets


SCRIPT_DIR = Path(__file__).resolve().parent
CONFIG_FILE = SCRIPT_DIR / "index_bindings.json"

DEFAULT_CONFIG = {
    "head_mode": "hybrid",  # "hybrid" (additive delta) or "seated" (1:1 relative to recenter)
    "hmd_sensitivity": {
        "yaw": 1.0,
        "pitch": 1.0,
        "roll": 0.85,
    },
    "capture_override": None,  # Optional {"left": 100, "top": 100, "width": 2880, "height": 1600}
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
            "lua": "__BRIDGE_TOGGLE_HEAD_MODE__",
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


# --- Quaternion Math Helpers ---
def matrix34_to_quat(m):
    """Convert OpenVR 3x4 HmdMatrix34_t rotation block to (w, x, y, z) quaternion."""
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
    return (w / norm, x / norm, y / norm, z / norm)


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


def quat_to_euler_rad(q):
    """Return (pitch, yaw, roll) in radians for OpenVR (+X right, +Y up, -Z forward)."""
    w, x, y, z = q
    yaw = math.atan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (y * y + x * x))
    sinp = max(-1.0, min(1.0, 2.0 * (w * x - z * y)))
    pitch = math.asin(sinp)
    roll = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (x * x + z * z))
    return pitch, yaw, roll


# --- Cross-Platform Stereo Window Locator ---
def find_window_rect(title="OpenSpaceVR", fallback=None):
    """Locate the client screen bounds of the OpenSpaceVR window on Windows or Linux."""
    if fallback:
        return dict(fallback)

    if sys.platform.startswith("win"):
        try:
            user32 = ctypes.windll.user32

            class RECT(ctypes.Structure):
                _fields_ = [
                    ("left", ctypes.c_long),
                    ("top", ctypes.c_long),
                    ("right", ctypes.c_long),
                    ("bottom", ctypes.c_long),
                ]

            class POINT(ctypes.Structure):
                _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

            hwnd = user32.FindWindowW(None, title)
            if hwnd and not user32.IsIconic(hwnd):
                rect = RECT()
                if user32.GetClientRect(hwnd, ctypes.byref(rect)):
                    pt = POINT(0, 0)
                    user32.ClientToScreen(hwnd, ctypes.byref(pt))
                    w = rect.right - rect.left
                    h = rect.bottom - rect.top
                    if w > 100 and h > 100:
                        return {"left": int(pt.x), "top": int(pt.y), "width": int(w), "height": int(h)}
        except Exception:
            pass
    elif sys.platform.startswith("linux"):
        try:
            import subprocess

            out = subprocess.check_output(["xwininfo", "-name", title], stderr=subprocess.DEVNULL, text=True)
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

    return {"left": 100, "top": 100, "width": 2880, "height": 1600}


# --- Bridge State ---
class BridgeState:
    def __init__(self):
        self.lock = threading.Lock()
        self.config = self.load_config()
        self.openspace_connected = False
        self.steamvr_connected = False
        self.lua_queue = queue.Queue(maxsize=32)

        # HMD tracking state
        self.q_prev = None
        self.q_ref = None
        self.last_seated_euler = (0.0, 0.0, 0.0)
        self.recenter_requested = True

        # Controller state tracking
        self.prev_buttons = {}
        self.step_timers = {}
        self.telemetry = {
            "hmd_euler_deg": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0},
            "head_mode": self.config["head_mode"],
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
    """Return dict of normalized axis and digital button values for Left & Right Index controllers."""
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

        # Axes (OpenVR Legacy / Index mapping)
        inputs[f"{side}_Stick_X"] = float(state.rAxis[0].x)
        inputs[f"{side}_Stick_Y"] = float(state.rAxis[0].y)
        inputs[f"{side}_Trigger"] = float(state.rAxis[1].x)
        inputs[f"{side}_Grip"] = float(state.rAxis[2].x)
        inputs[f"{side}_Trackpad_X"] = float(state.rAxis[3].x)
        inputs[f"{side}_Trackpad_Y"] = float(state.rAxis[3].y)

        # Digital Buttons
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
        print("[VR] Connected to SteamVR runtime (Valve Index ready).")
    except Exception as exc:
        print(f"[VR] Could not initialize SteamVR: {exc}")
        glfw.terminate()
        return

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

    # Flip V coordinates so top-down screen buffer renders right-side up in OpenGL
    bounds = openvr.VRTextureBounds_t(0.0, 1.0, 1.0, 0.0)
    tex_left_desc = openvr.Texture_t()
    tex_left_desc.handle = ctypes.cast(int(left_tex), ctypes.c_void_p)
    tex_left_desc.eType = openvr.TextureType_OpenGL
    tex_left_desc.eColorSpace = openvr.ColorSpace_Gamma

    tex_right_desc = openvr.Texture_t()
    tex_right_desc.handle = ctypes.cast(int(right_tex), ctypes.c_void_p)
    tex_right_desc.eType = openvr.TextureType_OpenGL
    tex_right_desc.eColorSpace = openvr.ColorSpace_Gamma

    last_rect_check = 0.0
    capture_rect = {"left": 100, "top": 100, "width": 2880, "height": 1600}

    with mss.mss() as sct:
        try:
            while not glfw.window_should_close(gl_window):
                now = time.perf_counter()
                compositor.waitGetPoses(poses, None)

                # Periodically refresh OpenSpaceVR window bounds
                if now - last_rect_check > 1.0:
                    with bridge.lock:
                        override = bridge.config.get("capture_override")
                    capture_rect = find_window_rect(window_title, override)
                    bridge.telemetry["capture_rect"] = capture_rect
                    last_rect_check = now

                # 1. Capture side-by-side stereo window and submit to SteamVR Compositor
                if not disable_compositor and capture_rect["width"] >= 200 and capture_rect["height"] >= 100:
                    try:
                        grab = sct.grab(capture_rect)
                        frame = np.frombuffer(grab.bgra, dtype=np.uint8).reshape((grab.height, grab.width, 4))
                        eye_w = grab.width // 2
                        eye_h = grab.height

                        left_eye = np.ascontiguousarray(frame[:, :eye_w, :])
                        right_eye = np.ascontiguousarray(frame[:, eye_w : eye_w * 2, :])

                        if tex_dims[0] != eye_w or tex_dims[1] != eye_h:
                            GL.glBindTexture(GL.GL_TEXTURE_2D, left_tex)
                            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, eye_w, eye_h, 0, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, left_eye)
                            GL.glBindTexture(GL.GL_TEXTURE_2D, right_tex)
                            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, eye_w, eye_h, 0, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, right_eye)
                            tex_dims = [eye_w, eye_h]
                        else:
                            GL.glBindTexture(GL.GL_TEXTURE_2D, left_tex)
                            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, eye_w, eye_h, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, left_eye)
                            GL.glBindTexture(GL.GL_TEXTURE_2D, right_tex)
                            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, eye_w, eye_h, GL.GL_BGRA, GL.GL_UNSIGNED_BYTE, right_eye)

                        compositor.submit(openvr.Eye_Left, tex_left_desc, bounds)
                        compositor.submit(openvr.Eye_Right, tex_right_desc, bounds)
                    except Exception:
                        pass

                lua_cmds = []

                # 2. Process HMD 6DoF Orientation
                hmd_pose = poses[openvr.k_unTrackedDeviceIndex_Hmd]
                if hmd_pose.bPoseIsValid:
                    q_curr = matrix34_to_quat(hmd_pose.mDeviceToAbsoluteTracking)
                    pitch_abs, yaw_abs, roll_abs = quat_to_euler_rad(q_curr)
                    bridge.telemetry["hmd_euler_deg"] = {
                        "yaw": round(math.degrees(yaw_abs), 1),
                        "pitch": round(math.degrees(pitch_abs), 1),
                        "roll": round(math.degrees(roll_abs), 1),
                    }

                    with bridge.lock:
                        head_mode = bridge.config["head_mode"]
                        sens = dict(bridge.config["hmd_sensitivity"])
                        recenter = bridge.recenter_requested
                        bridge.recenter_requested = False

                    if recenter or bridge.q_ref is None:
                        bridge.q_ref = q_curr
                        bridge.q_prev = q_curr
                        bridge.last_seated_euler = (0.0, 0.0, 0.0)

                    # Gain tuned for OpenSpace 0.22 local impulse scale
                    gain = 32.0
                    if head_mode == "hybrid":
                        q_delta = quat_mul(quat_inv(bridge.q_prev), q_curr)
                        dp, dy, dr = quat_to_euler_rad(q_delta)
                        bridge.q_prev = q_curr
                    else:
                        # 1:1 Seated Absolute offset from calibrated reference
                        q_offset = quat_mul(quat_inv(bridge.q_ref), q_curr)
                        sp, sy, sr = quat_to_euler_rad(q_offset)
                        lp, ly, lr = bridge.last_seated_euler
                        dp, dy, dr = (sp - lp), (sy - ly), (sr - lr)
                        bridge.last_seated_euler = (sp, sy, sr)
                        bridge.q_prev = q_curr

                    # Note: OpenSpace addLocalRotation positive horizontal rotates LEFT, so negate dy
                    yaw_cmd = -dy * gain * sens["yaw"]
                    pitch_cmd = dp * gain * sens["pitch"]
                    roll_cmd = -dr * gain * sens["roll"]

                    if abs(yaw_cmd) > 1e-4 or abs(pitch_cmd) > 1e-4:
                        lua_cmds.append(f"openspace.navigation.addLocalRotation({yaw_cmd:.5f}, {pitch_cmd:.5f});")
                    if abs(roll_cmd) > 1e-4:
                        lua_cmds.append(f"openspace.navigation.addLocalRoll({roll_cmd:.5f});")

                # 3. Process Valve Index Controllers
                ctrl_inputs = read_controller_inputs(vr_sys)
                with bridge.lock:
                    mappings = list(bridge.config["mappings"])

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

                    # Detect most active input for live UI monitor
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
                            elif lua_tpl == "__BRIDGE_TOGGLE_HEAD_MODE__":
                                with bridge.lock:
                                    bridge.config["head_mode"] = (
                                        "seated" if bridge.config["head_mode"] == "hybrid" else "hybrid"
                                    )
                                    bridge.recenter_requested = True
                                    bridge.telemetry["head_mode"] = bridge.config["head_mode"]
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
                        # Renormalize past deadzone
                        sign = 1.0 if raw >= 0 else -1.0
                        norm = (abs(raw) - dz) / max(1e-5, (1.0 - dz))
                        if "Stick" in src or "Trackpad" in src:
                            # Signed -1..1 -> min..max
                            signed_norm = sign * norm
                            scaled = ((signed_norm + 1.0) * 0.5) * (max_v - min_v) + min_v
                        else:
                            # Unipolar 0..1 (Trigger/Grip) -> min..max
                            scaled = min_v + norm * (max_v - min_v)

                        if abs(scaled) > 1e-4:
                            lua_cmds.append(lua_tpl.replace("{VALUE}", f"{scaled:.5f}"))

                if lua_cmds:
                    bridge.enqueue_lua(" ".join(lua_cmds))

        finally:
            openvr.shutdown()
            glfw.terminate()


# --- Async OpenSpace WebSocket Sender + Local UI WebSocket Server ---
async def openspace_sender_loop(bridge: BridgeState, host: str, port: int):
    uri = f"ws://{host}:{port}/"
    while True:
        try:
            async with websockets.connect(uri, ping_interval=None) as ws:
                bridge.openspace_connected = True
                bridge.telemetry["openspace_connected"] = True
                print(f"[OpenSpace] Connected to {uri}")
                while True:
                    try:
                        script = bridge.lua_queue.get_nowait()
                        payload = json.dumps({"topic": 0, "type": "luascript", "payload": {"script": script}})
                        await ws.send(payload)
                    except queue.Empty:
                        await asyncio.sleep(0.004)
        except Exception:
            if bridge.openspace_connected:
                print("[OpenSpace] Disconnected. Reconnecting...")
            bridge.openspace_connected = False
            bridge.telemetry["openspace_connected"] = False
            await asyncio.sleep(1.5)


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
                    bridge.telemetry["head_mode"] = bridge.config["head_mode"]
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

    vr_thread = threading.Thread(
        target=vr_loop,
        args=(bridge, args.window_title, args.no_compositor),
        daemon=True,
    )
    vr_thread.start()

    ws_server = await websockets.serve(
        functools.partial(ui_ws_handler, bridge=bridge),
        "127.0.0.1",
        args.ui_ws_port,
    )
    print(f"[UI] Bridge WebSocket listening on ws://127.0.0.1:{args.ui_ws_port}")

    await asyncio.gather(
        openspace_sender_loop(bridge, args.os_host, args.os_port),
        ws_server.wait_closed(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Valve Index + SteamVR Bridge for OpenSpace 0.22")
    parser.add_argument("--os-host", default="localhost", help="OpenSpace WebSocket host")
    parser.add_argument("--os-port", type=int, default=4682, help="OpenSpace WebSocket port")
    parser.add_argument("--http-port", type=int, default=8080, help="Local HTML mapper HTTP port")
    parser.add_argument("--ui-ws-port", type=int, default=8081, help="Local UI telemetry WebSocket port")
    parser.add_argument("--window-title", default="OpenSpaceVR", help="SGCT stereo window name to capture")
    parser.add_argument("--no-compositor", action="store_true", help="Disable SteamVR frame submission (controls only)")
    asyncio.run(main_async(parser.parse_args()))
