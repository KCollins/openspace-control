Install dependencies: `pip install openvr websockets mss numpy glfw PyOpenGL`

## How to Launch & Test

0) Move valveindex_vr.json to your config (or user/config) folder.

1) Start SteamVR with your Valve Index connected (leave SteamVR in normal Direct Mode).

2) Launch OpenSpace 0.22 using valveindex_vr.json as your SGCT configuration.

3) Run the Python bridge: `python index_openspace_bridge.py`

4) Put on the Index & tune live: Open http://localhost:8080/index_mapper.html in your browser to see live head-tracking angles, switch between Hybrid Additive and 1:1 Seated head modes (or press Left B on the controller), recenter forward look (Left A), and adjust flight/time sensitivities on the fly.
