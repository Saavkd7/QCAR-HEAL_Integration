"""Build CoopFront/: the N-agent cooperative OPV2V-layout dataset, generalized
from the original two-agent build (9 paired trajectories, qcar-52775 x
qcar-52776, captured 2026-09-13/14), converted from raw .bag via
bag_to_dataset_rosbags.py.

**Functional for N=2 today (the only physically calibrated pair), written to
be functional for N>2 as soon as a 3rd+ real QCar gets its own calibration.**
Nothing here invents data for an uncalibrated car -- adding agent "3" means
adding a real CAR_REGISTRY entry, a real EXTRINSIC_PARAMS entry, and real
captured bags, exactly the same way "1"/"2" were added originally.

Pipeline, mirroring build_smoketest_dataset.py's conventions:
  - x10 horizontal-only world scale (position/extent x,y; NOT z/height/K).
  - Real per-car fisheye K/D, same-K undistortion (cv2.fisheye.undistortImage).
  - Each agent's own lidar_pose is its real (scaled) Vicon world position --
    NOT forced to the origin, because agents genuinely move here.
  - The SCENE has len(AGENT_IDS) dynamic vehicles plus len(STATIC_TARGETS)
    STATIC targets (currently 2, parked at Node 9 / Node 11 for the ENTIRE
    capture -- user-confirmed 2026-09-14: all 4 were physically present
    throughout). Their Vicon TCP protocol only ever carries "ego"/"target" --
    two slots -- so there is no live tracking of statics in these bags; since
    they never moved, their fixed Vicon reading is injected into every frame
    instead.
  - vehicles[other_id] = each OTHER vehicle's real position, WITH the
    marker-to-physical-center body-frame offset applied (the correction
    that matters for box placement accuracy; ego's own lidar_pose is left
    as the raw Vicon reading, standard convention). LEFT UNFILTERED by
    geometry (every other vehicle, every frame) EXCEPT for real wall
    occlusion (see WALLS/wall_geometry.py below), which is hard-excluded
    from that one agent's own yaml -- everything else is left for HEAL's
    OWN native per-cav box_is_visible() + cross-cav-union logic to decide
    at load time (verified by reading intermediate_heter_fusion_dataset.py:
    generate_object_center is called separately per cav with that cav's own
    vehicles{} + own mask, and the final ego GT is the union of every
    loaded cav's filtered result, deduped by id -- pre-filtering here would
    only ever DROP targets HEAL's own mask would have kept).
  - Wall occlusion: user-decided 2026-09-22 to always go by REAL geometry
    (WALLS, loaded from shared/walls.json via wall_geometry.py -- 5 real
    wall segments measured from media/Walls_Layout+Scenario1.jpeg) rather
    than a fixed hand-confirmed pair fact, even though that geometry is
    known-incomplete (validated 2026-09-22: only reproduces 73.9%/79.1% of
    the previously-confirmed always-blocked pairs (1,3)/(2,4) -- see the
    2026-09-22 "19 ANALISIS" research note for the root cause, two real
    corridors where the measured wall segments don't reach). The historical
    fixed fact this replaced (agent "1" x Node 9, agent "2" x Node 11
    always blocked, regardless of FOV/range, a WALL_BLOCKED constant) has
    been fully retired -- diagnose_coop_breakdown.py and
    compare_extrinsic_nominal_vs_calibrated.py, the only other consumers,
    were updated the same day to use wall_geometry.blocked_by_wall()
    directly instead.
  - scene_gt/<frame>.json: an evaluation oracle per frame -- lists a
    vehicle ONLY when it is visible to at least one agent (empty/absent
    otherwise), using the SAME wall-corrected visibility as above, so a
    detection can be scored as right or wrong against "what should be
    detectable", not against omniscient world truth nobody could see.
  - qcar_provenance.cav_id is written so patch_real_extrinsic.py can select
    the right camera geometry per agent at load time.
  - Split by WHOLE TRAJECTORY (never a frame cut).
  - Trajectories are DISCOVERED, not hardcoded: for each active agent, every
    `converted_*_<NN>` folder under `<onboard-root>/<ip>/dataset/` is a
    candidate; a scenario is built for every trailing index present on
    ALL active agents (different cars' route-name prefixes can differ --
    e.g. Node3Via1To13 vs Node15Via6To8 -- pairing is by the shared
    trailing index only, never by matching the full folder name).

N-agent generalization (new in this revision):
  - AGENT_IDS is no longer hardcoded to ("1","2") -- it comes from --agents
    (default "1,2", so existing behavior/output is byte-identical unless you
    pass something else).
  - CAR_REGISTRY maps a fixed agent id -> (real IP, calibration folder name,
    label, visibility mask filename). The id<->physical-car mapping is
    declared here explicitly, NOT auto-derived by sorting IP strings on
    disk -- physical facts (wall geometry, static target positions) are
    tied to a SPECIFIC car's identity, and a sort-order-based id assignment
    could silently swap which car a fact applies to (e.g. "192.168.1.158"
    sorts before "192.168.1.198" lexicographically, which would flip agent
    "1"/"2" from their established meaning). Adding a 3rd+ car means adding
    its own registry entry, its own EXTRINSIC_PARAMS entry (real measured
    or at least real nominal-manual geometry -- never fabricated), and
    updating shared/walls.json/STATIC_TARGETS if new physical occlusion
    facts apply.
  - pair_frames(frames_a, frames_b) was retired -- every importer
    (Report_creation.py, the diagnose_*.py scripts, main() itself) now
    calls pair_frames_n({EGO_ID: frames_a, PEER_ID: frames_b}) directly, a
    single N-agent implementation instead of two near-duplicate ones.
  - build_scene(fa, fb) keeps its EXACT original 2-argument signature too,
    as a thin wrapper around the new build_scene_n(frames_by_agent) --
    still valid whenever EGO_ID/PEER_ID (the first two configured agents)
    are the pair in question.
  - EGO_ID/PEER_ID and SCENE_VEHICLE_IDS remain as plain module-level
    values (not functions) for backward compatibility with importers that
    read them as attributes.

Capture sessions and per-vehicle centres (2026-10-06):
  - conf.json's `sessions` lists every capture that goes into the dataset,
    each with its raw layout ("converted": ExperimentNo1's
    <ip>/dataset/converted_*_<NN>/; "frames": the October captures'
    <ip>/<run>/frames/), its per-agent folder (the cars' IPs changed between
    sessions; agent "1" is always qcar-52775, "2" always qcar-52776), how runs
    pair across cars (trailing index or identical name), its scenario name
    template, its validation runs, and its targets: the parked QCars
    (static_targets) and/or a dynamic target whose Vicon pose is broadcast in
    every frame (dynamic_target, e.g. the third car, id 3, pose taken from the
    reference agent's frame).
  - The marker-plate -> centre offset is per vehicle TYPE (conf.json's
    marker_offsets): every QCar shares the measured QCar offset; the third
    car has its own, from a 180-degree reversal test (reversal_center_offset).
    The yaml's qcar_provenance.marker_offset_applied_m records, per listed
    vehicle id, the offset actually used.
  - Output keeps the established layout; scenario names come from the
    session template (qcar_coop_<NN> for ExperimentNo1).

Usage:
    python build_coop_train_val_dataset.py      # every default comes from conf.json

Every default (paths, agents, sessions, val split, pairing tolerance, static
targets, marker offsets) lives in conf.json next to this script; edit it
there. Flags are optional one-run overrides for anyone who doesn't want to
touch the file:
    --conf <path>  --calib-root <dir>  --onboard-root <dir>  --out-dir <dir>
    --vis-masks-dir <dir>  --walls-json <file>  --val-indices 07,08
    --agents 1,2  --max-trajectories N  --sessions ExperimentNo1,2026-10-05
"""
import argparse
import json
import math
import os
import re
import shutil
import sys

import cv2
import numpy as np
import yaml

import wall_geometry

HERE = os.path.dirname(os.path.abspath(__file__))
PIPELINE_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))  # scripts/build/<category>/ -> build/ -> scripts/ -> pipeline/
# qcar_testbed_integration root: pipeline/ -> qcar_dataset/ -> data/ -> repo root.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(PIPELINE_ROOT)))
DEFAULT_CONF = os.path.join(HERE, "conf.json")


def _load_conf(path):
    """conf.json next to this script (shared with build_inference.py) is THE
    source of every default below -- nothing is hardcoded in this file. A
    CLI flag is optional and only overrides the matching key for that one
    run; no script ever writes conf.json. Relative paths in it are resolved
    against the qcar_testbed_integration/ root (REPO_ROOT)."""
    if not os.path.exists(path):
        raise SystemExit("conf.json not found: %s" % path)
    with open(path) as f:
        return json.load(f)


def _path_from_conf(key):
    """conf.json path value; relative ones are resolved against REPO_ROOT."""
    v = _conf.get(key)
    return v if not v or os.path.isabs(v) else os.path.join(REPO_ROOT, v)


# --- CLI flags -----------------------------------------------------------
# This module is both a runnable script AND a library imported by the
# diagnose_*.py / Report_creation.py scripts. `add_help` is off and
# `parse_known_args` is used (not `parse_args`) so importing this module
# never steals `-h`/errors out on an importing script's own flags -- see
# qcar_dataset/pipeline/README.md's "Flag convention" section.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--conf", default=DEFAULT_CONF)
_conf_path = _pre.parse_known_args()[0].conf
_conf = _load_conf(_conf_path)

_parser = argparse.ArgumentParser(add_help=False)
_parser.add_argument(
    "--conf", default=DEFAULT_CONF,
    help="Path to the conf.json to read defaults from (default: %s)" % DEFAULT_CONF)
_parser.add_argument(
    "--calib-root", default=_path_from_conf("calib_root"),
    help="Root of calibration_Matries/ (default: conf.json's calib_root)")
_parser.add_argument(
    "--onboard-root", default=_path_from_conf("onboard_root"),
    help="Root holding one <ip>/dataset/ per active agent (default: conf.json's onboard_root)")
_parser.add_argument(
    "--out-dir", default=_path_from_conf("out_dir"),
    help="Where to write CoopFront/ (default: conf.json's out_dir)")
_parser.add_argument(
    "--vis-masks-dir", default=_path_from_conf("vis_masks_dir"),
    help="Folder with each car's <car>_bev_visibility.png (default: conf.json's vis_masks_dir)")
_parser.add_argument(
    "--walls-json", default=_path_from_conf("walls_json"),
    help="Real wall-segment geometry used for occlusion (default: conf.json's walls_json)")
_parser.add_argument(
    "--val-indices", default=_conf.get("val_indices"),
    help="Comma-separated trajectory indices held out for validate; discovered "
         "indices not listed go to train, '' means no validate split "
         "(default: conf.json's val_indices)")
_parser.add_argument(
    "--agents", default=_conf.get("agents"),
    help="Comma-separated agent ids (from CAR_REGISTRY) to include this run. Every id "
         "listed must have both a CAR_REGISTRY entry and an EXTRINSIC_PARAMS entry "
         "(default: conf.json's agents)")
_parser.add_argument(
    "--max-trajectories", type=int, default=_conf.get("max_trajectories"),
    help="Only build the first N discovered trajectory indices (sorted) -- for a fast "
         "smoke-test build from 100%% real data/code. null = every discovered "
         "trajectory (default: conf.json's max_trajectories)")
_parser.add_argument(
    "--sessions", default=None,
    help="Comma-separated conf.json session names to build (default: every session)")
# add_help=False means -h/--help would otherwise be silently ignored (and
# this DESTRUCTIVE script -- main() does shutil.rmtree(DST) then rebuilds --
# would run for real). Handle it manually, only when run as a script (never
# when imported, so an importing script's own --help still wins there).
if __name__ == "__main__" and any(a in ("-h", "--help") for a in sys.argv[1:]):
    _parser.print_help()
    raise SystemExit(0)
_args, _ = _parser.parse_known_args()
_missing = [k for k in ("calib_root", "onboard_root", "out_dir", "vis_masks_dir",
                        "walls_json", "val_indices", "agents")
            if getattr(_args, k) is None]
if _missing:
    raise SystemExit("No value for %s: set it in %s or pass the flag."
                     % (", ".join("--" + k.replace("_", "-") for k in _missing), _conf_path))

CALIB = _args.calib_root
ONBOARD_ROOT = _args.onboard_root
DST = _args.out_dir
VIS_MASKS = _args.vis_masks_dir

SCALE = 10.0
QCAR_L, QCAR_W, QCAR_H = 0.425, 0.192, 0.190
PAIR_TOLERANCE_SEC = _conf["pair_tolerance_sec"]  # project's established wall-time pairing tolerance

def reversal_center_offset(a, b):
    """Body-frame (forward, lateral) metres from the marker plate to the
    midpoint of two Vicon readings a/b = (x, y, yaw_rad) taken with the
    vehicle turned 180 degrees inside the same taped wheel marks. Averaged
    over both readings (they agree to ~3 mm when the test is done right)."""
    cx, cy = (a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0
    fwd = lat = 0.0
    for x, y, yaw in (a, b):
        c, s = math.cos(yaw), math.sin(yaw)
        fwd += (c * (cx - x) + s * (cy - y)) / 2.0
        lat += (-s * (cx - x) + c * (cy - y)) / 2.0
    return fwd, lat


def _resolve_marker_offsets(spec):
    """conf.json's marker_offsets -> {vehicle_type: (fwd_m, lat_m)}. A type
    is measured directly (fwd_m/lat_m) or by a reversal test relative to an
    already-resolved type (see conf.json's _comment for why relative)."""
    out = {}
    pending = {k: v for k, v in spec.items() if not k.startswith("_")}
    while pending:
        progressed = False
        for name, v in list(pending.items()):
            if "fwd_m" in v:
                out[name] = (v["fwd_m"], v["lat_m"])
            elif v.get("relative_to") in out:
                ref = out[v["relative_to"]]
                veh = reversal_center_offset(v["reversal"]["A"], v["reversal"]["B"])
                rr = reversal_center_offset(v["reference_reversal"]["A"], v["reference_reversal"]["B"])
                out[name] = (ref[0] + veh[0] - rr[0], ref[1] + veh[1] - rr[1])
            else:
                continue
            del pending[name]
            progressed = True
        if not progressed:
            raise SystemExit("marker_offsets: unresolvable entries %s" % sorted(pending))
    return out


# Vicon-origin -> exterior-box-centre offset per offset name (conf.json's
# marker_offsets). Each agent uses its own measured entry (vehicles.<id>.
# marker_offset); targets name theirs in their 'vehicle' field.
MARKER_OFFSETS = _resolve_marker_offsets(_conf["marker_offsets"])
OFFSET_FWD_M, OFFSET_LAT_M = MARKER_OFFSETS["qcar"]  # legacy names, read by importers

# --- Physical car registry (conf.json's vehicles) ---------------------------
# One entry per REAL QCar this project has actually calibrated. The agent id
# assignment is declared explicitly in conf.json, NOT auto-derived by sorting
# IPs on disk -- see the module docstring for why that matters (wall geometry
# and other physical facts are tied to a specific physical car's identity).
# To add a 3rd+ car: add a real vehicles entry (real calibration_Matries/<name>/
# must exist) pointing at a real marker_offsets and camera_extrinsics entry --
# never fabricate either.
CAR_REGISTRY = {aid: v for aid, v in _conf["vehicles"].items() if not aid.startswith("_")}
AGENT_MARKER_OFFSET = {aid: v["marker_offset"] for aid, v in CAR_REGISTRY.items()}
for _aid, _name in AGENT_MARKER_OFFSET.items():
    if _name not in MARKER_OFFSETS:
        raise SystemExit("vehicles.%s.marker_offset %r is not in marker_offsets" % (_aid, _name))

# Camera-to-vehicle rotation + body-frame mount translation (conf.json's
# camera_extrinsics). qcar/patches/patch_real_extrinsic.py reads the same
# entries, so this visibility precheck agrees with what the model sees.
_EXTR = _conf["camera_extrinsics"]


def _extrinsic(name):
    e = _EXTR[name]
    return np.array(e["R_cb"], dtype=np.float64), np.array(e["t_cb_m"], dtype=np.float64)


FRONT_MOUNT_XYZ = tuple(_EXTR[_EXTR["cam_cords_mount"]]["t_cb_m"])  # cam_cords (rotation handled by the patch)
R_CB_NOMINAL = _extrinsic("manual_nominal")[0]             # legacy names, read by importers
R_CB_QCAR52775, T_CB_QCAR52775 = _extrinsic("qcar52775_refined")
EXTRINSIC_PARAMS = {aid: _extrinsic(v["camera_extrinsic"]) for aid, v in CAR_REGISTRY.items()}

IMG_W, IMG_H = 640, 480

# --- Resolve which agents are active this run ------------------------------
AGENT_IDS = tuple(a.strip() for a in _args.agents.split(",") if a.strip())
if len(AGENT_IDS) < 2:
    raise ValueError("--agents needs at least 2 ids for cooperative perception, got %r" % (AGENT_IDS,))
for _aid in AGENT_IDS:
    if _aid not in CAR_REGISTRY:
        raise ValueError(
            "agent id %r (from --agents) has no CAR_REGISTRY entry -- add its real "
            "calibration folder first, it cannot be fabricated" % _aid)
    if _aid not in EXTRINSIC_PARAMS:
        raise ValueError(
            "agent id %r (from --agents) has no EXTRINSIC_PARAMS entry -- add its real "
            "measured or manual-nominal camera-to-vehicle geometry first" % _aid)

# Legacy aliases: the first two active agents, exactly what EGO_ID/PEER_ID
# meant before this revision. Kept so build_scene()'s original 2-argument
# signature (and any importer reading these attributes directly) keeps
# working unchanged for the --agents 1,2 default case.
EGO_ID, PEER_ID = AGENT_IDS[0], AGENT_IDS[1]


def _load_cam_params():
    params = {}
    for aid in AGENT_IDS:
        reg = CAR_REGISTRY[aid]
        calib = np.load(os.path.join(
            CALIB, reg["calib_name"], "latest_front",
            "%s_front_intrinsics_verified.npz" % reg["calib_name"]))
        params[aid] = {
            "K": calib["front_K"].astype(np.float64), "D": calib["front_D"].astype(np.float64),
            "car": reg["car_label"],
            "root": os.path.join(ONBOARD_ROOT, reg["ip"], "dataset"),
            "vis_mask": os.path.join(VIS_MASKS, reg["vis_mask_file"]),
        }
    return params


CAM_PARAMS = _load_cam_params()

VAL_PAIRS = set(i.strip() for i in _args.val_indices.split(",") if i.strip())

_TRAJ_INDEX_RE = re.compile(r"^converted_.+_(\d+)$")


def discover_trajectories(cav_id):
    """Scan <root>/ for every converted_*_<NN> folder and return
    {index: folder_name}. Indices are whatever trailing digits are
    present in the real folder name (zero-padded or not) -- never assumed
    to be a fixed "00".."08" range."""
    root = CAM_PARAMS[cav_id]["root"]
    found = {}
    if not os.path.isdir(root):
        return found
    for name in sorted(os.listdir(root)):
        m = _TRAJ_INDEX_RE.match(name)
        if m and os.path.isdir(os.path.join(root, name)):
            found[m.group(1)] = name
    return found


def common_indices():
    """Trajectory indices present on EVERY active agent (AGENT_IDS), sorted.
    This -- not a fixed range() -- is what main() and every importing
    script should iterate over, so a newly captured trajectory (any index,
    any route name) is picked up automatically as soon as all active
    agents' converted_* folders exist, with no code change."""
    per_agent = [set(discover_trajectories(aid)) for aid in AGENT_IDS]
    if not per_agent or any(not s for s in per_agent):
        return []
    common = set.intersection(*per_agent)
    return sorted(common, key=lambda s: (len(s), s))


# Static targets parked at Node 9 / Node 11 for the ENTIRE capture (real
# Vicon reading, provided by the user for the robustness-capture plan and
# confirmed 2026-09-14 to have been physically present during these same 9
# trajectories -- not just planned for a future capture). Read from
# conf.json's static_targets (user-decided 2026-09-23: real positions are
# DATA, not code -- editing conf.json to add/move a static target should
# never require touching this file).
STATIC_TARGETS = _conf["static_targets"]
# All vehicles in the scene this run: every active dynamic agent + every
# static target. A plain tuple (not a function) so importers can keep
# reading it as a module attribute, same as before this revision.
SCENE_VEHICLE_IDS = tuple(AGENT_IDS) + tuple(STATIC_TARGETS.keys())

# Real wall-segment geometry (shared/walls.json, user-confirmed 2026-09-22,
# measured from media/Walls_Layout+Scenario1.jpeg) -- a PER-FRAME occlusion
# check (does the line of sight from the agent's real position to the
# target's real position cross a real wall). This is now the ONLY wall
# check this script's dataset-building logic uses (see the module docstring
# for why, and its known incompleteness). Loaded once; add more walls to
# that JSON file, no code change needed here.
WALLS = wall_geometry.load_walls(_args.walls_json)


def world_box_corners(cx, cy, cz, yaw_deg, l, w, h):
    """8 corners in world/real metres from a FULL-extent l,w,h box centered
    at (cx,cy,cz), yawed by yaw_deg about z."""
    hl, hw, hh = l / 2.0, w / 2.0, h / 2.0
    yaw = math.radians(yaw_deg)
    c, s = math.cos(yaw), math.sin(yaw)
    local = np.array([
        [hl, hw, hh], [hl, -hw, hh], [-hl, -hw, hh], [-hl, hw, hh],
        [hl, hw, -hh], [hl, -hw, -hh], [-hl, -hw, -hh], [-hl, hw, -hh],
    ])
    rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ rot.T + np.array([cx, cy, cz])


def project_world_to_camera(points_world, agent_x, agent_y, agent_z, agent_yaw_rad, cav_id, K):
    """Real-metre world points -> that agent's own camera, using the SAME
    geometry as patch_real_extrinsic.py (inverted: world -> camera)."""
    c, s = math.cos(agent_yaw_rad), math.sin(agent_yaw_rad)
    R_ego = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    t_ego = np.array([agent_x, agent_y, agent_z])
    R_cb, t_cb_body = EXTRINSIC_PARAMS[cav_id]
    p_ego_rel = (points_world - t_ego) @ R_ego  # R_ego^T @ v, row-vector form
    p_cam = (p_ego_rel - t_cb_body) @ R_cb       # R_cb^T @ v, row-vector form
    depth = p_cam[:, 2]
    uvw = p_cam @ K.T
    uv = uvw[:, :2] / np.clip(uvw[:, 2:3], 1e-6, None)
    return uv, depth


def truly_visible_to_agent(target_x, target_y, target_z, target_yaw_deg, agent_pose, cav_id):
    """Precise per-pixel-projection visibility check (matches
    build_pretrain_labels.py's truly_in_frame convention): >=4 corners with
    positive depth AND the projected centroid inside the real image
    bounds, using that agent's own real camera geometry."""
    corners = world_box_corners(target_x, target_y, target_z, target_yaw_deg,
                                 QCAR_L, QCAR_W, QCAR_H)
    K = CAM_PARAMS[cav_id]["K"]
    uv, depth = project_world_to_camera(
        corners, agent_pose["x"], agent_pose["y"], agent_pose["z"],
        agent_pose["yaw_rad"], cav_id, K)
    valid = depth > 0.05
    if valid.sum() < 4:
        return False
    center = uv[valid].mean(axis=0)
    return bool(0 <= center[0] < IMG_W and 0 <= center[1] < IMG_H)


def apply_marker_offset(pose, vehicle="qcar"):
    """Vicon reports the reflective-marker-plate position, not the
    physical center; the body-frame offset of that vehicle TYPE
    (MARKER_OFFSETS) is rotated by the vehicle's own yaw."""
    fwd, lat = MARKER_OFFSETS[vehicle]
    c, s = math.cos(pose["yaw_rad"]), math.sin(pose["yaw_rad"])
    dx = c * fwd - s * lat
    dy = s * fwd + c * lat
    return {"x": pose["x"] + dx, "y": pose["y"] + dy, "z": pose["z"],
            "yaw_rad": pose["yaw_rad"]}


def static_targets_meta():
    """STATIC_TARGETS as the per-scene target description build_scene_n()
    and the writers take: {id: {pose, static, node, vehicle}}."""
    return {vid: {"pose": {k: t[k] for k in ("x", "y", "z", "yaw_rad")}, "static": True,
                  "node": t["node"], "vehicle": t["vehicle"]}
            for vid, t in STATIC_TARGETS.items()}


def build_scene_n(frames_by_agent, targets=None):
    """Ground-truth registry for one paired frame group across N agents:
    real (uncorrected Vicon) pose for every vehicle in the scene, the
    marker-corrected physical-center pose, and per-agent visibility of
    each vehicle to each active agent's real camera.

    frames_by_agent: {agent_id: frame_dict} for every id in AGENT_IDS.
    targets: {id: {pose, static, node, vehicle}} for every non-agent vehicle
    in this frame; default = the session's STATIC_TARGETS. Agents are QCars.
    """
    targets = static_targets_meta() if targets is None else targets
    raw = {aid: {"x": frames_by_agent[aid]["x"], "y": frames_by_agent[aid]["y"],
                 "z": frames_by_agent[aid]["z"], "yaw_rad": frames_by_agent[aid]["yaw_rad"]}
           for aid in AGENT_IDS}
    raw.update({vid: t["pose"] for vid, t in targets.items()})
    vehicle_type = {vid: targets[vid]["vehicle"] if vid in targets else AGENT_MARKER_OFFSET[vid]
                    for vid in raw}

    corrected = {vid: apply_marker_offset(raw[vid], vehicle_type[vid]) for vid in raw}
    agent_pose = {aid: raw[aid] for aid in AGENT_IDS}  # camera geometry keys off the RAW (un-offset) ego pose

    visibility = {}
    for vid in raw:
        tgt = corrected[vid]
        yaw_deg = math.degrees(tgt["yaw_rad"])
        by_agent = {}
        for aid in AGENT_IDS:
            if aid == vid:
                by_agent[aid] = None  # self, not a detection target
                continue
            # Real per-frame wall occlusion: line of sight from the agent's
            # own real (raw) position to the target's real (raw) position,
            # tested against every real wall segment in shared/walls.json.
            # This is the ONLY wall check -- user-decided 2026-09-22: always
            # go by real geometry, never a fixed hand-confirmed pair fact,
            # even though walls.json is known-incomplete (see the
            # 2026-09-22 19 ANALISIS research note).
            if wall_geometry.blocked_by_wall(
                    (agent_pose[aid]["x"], agent_pose[aid]["y"]), (raw[vid]["x"], raw[vid]["y"]), WALLS):
                by_agent[aid] = False
                continue
            by_agent[aid] = truly_visible_to_agent(
                tgt["x"], tgt["y"], tgt["z"], yaw_deg, agent_pose[aid], aid)
        any_visible = bool(any(v for v in by_agent.values() if v is not None))
        visibility[vid] = {"by_agent": by_agent, "any": any_visible}

    return raw, corrected, visibility


def build_scene(fa, fb):
    """Original 2-agent entry point, kept byte-for-byte call-compatible for
    importers (Report_creation.py etc.) -- a thin wrapper around
    build_scene_n(). Only meaningful when EGO_ID/PEER_ID are the pair you
    actually want (true for the --agents 1,2 default)."""
    return build_scene_n({EGO_ID: fa, PEER_ID: fb})


def undistort(src_png, dst_png, K, D):
    im = cv2.imread(src_png)
    if im is None:
        raise IOError("Could not read %s" % src_png)
    out = cv2.fisheye.undistortImage(im, K, D, Knew=K)
    cv2.imwrite(dst_png, out)


def _read_pose(path):
    """A Vicon pose json -> {x, y, z, yaw_rad}, or None if missing, flagged
    invalid, or non-finite."""
    if not os.path.exists(path):
        return None
    with open(path) as f:
        p = json.load(f)
    if p.get("valid", 0.0) != 1.0:
        return None
    pose = {"x": p["x"], "y": p["y"], "z": p["z"], "yaw_rad": p["yaw"]}
    return pose if all(math.isfinite(v) for v in pose.values()) else None


def load_run_frames(run_dir, target_file=None):
    """Usable frames of one agent's run: <run_dir>/<digits>/ with
    timestamp.json, a valid ego_vicon_pose.json and front.png -- plus, when
    the session tracks a dynamic target, a valid <target_file> (the target's
    Vicon pose as broadcast in that same frame). Sorted by (time, frame id)."""
    frames = []
    for fid in sorted(os.listdir(run_dir)):
        fdir = os.path.join(run_dir, fid)
        if not os.path.isdir(fdir) or not fid.isdigit():
            continue
        ts_path = os.path.join(fdir, "timestamp.json")
        if not (os.path.exists(ts_path) and os.path.exists(os.path.join(fdir, "front.png"))):
            continue
        pose = _read_pose(os.path.join(fdir, "ego_vicon_pose.json"))
        target = _read_pose(os.path.join(fdir, target_file)) if target_file else None
        if pose is None or (target_file and target is None):
            continue
        with open(ts_path) as f:
            t = float(json.load(f)["reference_time_sec"])
        if not math.isfinite(t):
            continue
        frames.append(dict(fid=fid, dir=fdir, t=t, target=target, **pose))
    frames.sort(key=lambda r: (r["t"], r["fid"]))
    return frames


def load_frames(cav_id, idx):
    """Frames of trajectory <idx> of agent <cav_id> in the onboard_root
    (ExperimentNo1-layout) session -- the entry point importers use."""
    p = CAM_PARAMS[cav_id]
    folder_name = discover_trajectories(cav_id).get(idx)
    if folder_name is None:
        raise FileNotFoundError(
            "No converted_*_%s folder found for cav %s under %s" % (idx, cav_id, p["root"]))
    return load_run_frames(os.path.join(p["root"], folder_name))


_TRAILING_INDEX_RE = re.compile(r"^(?:converted_)?.+_(\d+)$")


def session_runs(session):
    """Runs of one conf.json session present on EVERY active agent, as
    [(key, {agent_id: run_frames_dir})]. layout "converted":
    <root>/<agent_dir>/dataset/converted_<route>_<NN>/<frame>/;
    layout "frames": <root>/<agent_dir>/<run>/frames/<frame>/. Runs are
    paired across agents by trailing index ("index", route names may differ
    per car) or by identical folder name ("name")."""
    root = _session_root(session)
    by_agent = {}
    for aid in AGENT_IDS:
        adir = os.path.join(root, session["agent_dirs"][aid])
        if session["layout"] == "converted":
            adir = os.path.join(adir, "dataset")
        runs = {}
        for name in sorted(os.listdir(adir)) if os.path.isdir(adir) else []:
            run_dir = os.path.join(adir, name)
            if session["layout"] == "frames":
                run_dir = os.path.join(run_dir, "frames")
            if not os.path.isdir(run_dir):
                continue
            if session["pair_runs_by"] == "index":
                m = _TRAILING_INDEX_RE.match(name)
                if not m:
                    continue
                key = m.group(1)
            else:
                key = name
            if key in runs:
                raise ValueError("session %s: two runs share key %r under %s" % (session["name"], key, adir))
            runs[key] = run_dir
        by_agent[aid] = runs
    common = set.intersection(*(set(r) for r in by_agent.values()))
    for aid, runs in by_agent.items():
        for key in sorted(set(runs) - common):
            print("session %s: run %r of agent %s has no counterpart on every agent -- skipped"
                  % (session["name"], key, aid))
    return [(key, {aid: by_agent[aid][key] for aid in AGENT_IDS})
            for key in sorted(common, key=lambda s: (len(s), s))]


def _session_root(session):
    root = session.get("root") or ONBOARD_ROOT
    return root if os.path.isabs(root) or re.match(r"^[A-Za-z]:[\\/]", root) else os.path.join(REPO_ROOT, root)


def pair_frames_n(frames_by_agent):
    """Time-synchronize frames across all active agents: the first agent in
    AGENT_IDS is the time reference; for each of its frames, every OTHER
    active agent must have a frame within PAIR_TOLERANCE_SEC, or the whole
    group is dropped (never fabricated). Reduces to a simple pairwise
    nearest-neighbor match when AGENT_IDS has exactly 2 entries -- the
    former 2-argument pair_frames(frames_a, frames_b) was retired in favor
    of this single N-agent implementation; every importer now calls
    pair_frames_n({EGO_ID: frames_a, PEER_ID: frames_b}) instead.

    frames_by_agent: {agent_id: [frame_dict, ...]} for every id in AGENT_IDS.
    Returns: list of {agent_id: frame_dict} groups.
    """
    ref_id = AGENT_IDS[0]
    groups = []
    for f_ref in frames_by_agent[ref_id]:
        group = {ref_id: f_ref}
        ok = True
        for aid in AGENT_IDS[1:]:
            candidates = frames_by_agent[aid]
            if not candidates:
                ok = False
                break
            match = min(candidates, key=lambda r: abs(r["t"] - f_ref["t"]))
            if abs(match["t"] - f_ref["t"]) > PAIR_TOLERANCE_SEC:
                ok = False
                break
            group[aid] = match
        if ok:
            groups.append(group)
    return groups


def build_agent_yaml(me, me_id, raw, corrected, visibility, out_frame_dir, out_frame_num, targets=None):
    targets = static_targets_meta() if targets is None else targets
    p_me = CAM_PARAMS[me_id]
    me_yaw_deg = math.degrees(me["yaw_rad"])
    pose = [me["x"] * SCALE, me["y"] * SCALE, 0.0, 0.0, me_yaw_deg, 0.0]

    c, s = math.cos(math.radians(me_yaw_deg)), math.sin(math.radians(me_yaw_deg))
    mx, my = FRONT_MOUNT_XYZ[0], FRONT_MOUNT_XYZ[1]
    cam_cords = [me["x"] * SCALE + (c * mx - s * my) * SCALE,
                 me["y"] * SCALE + (s * mx + c * my) * SCALE,
                 FRONT_MOUNT_XYZ[2], 0.0, me_yaw_deg, 0.0]

    # Unfiltered by design: list every OTHER vehicle, exactly like stock
    # OPV2V's own yaml (the "god's eye" scene truth), and let HEAL's own
    # native per-cav box_is_visible() + generate_visible_object_center()
    # decide visibility at load time -- confirmed by reading
    # intermediate_heter_fusion_dataset.py: generate_object_center is
    # called separately per cav (that cav's own vehicles{} + own mask),
    # and the final training GT is the UNION of every loaded cav's
    # filtered result, deduped by object id (lines 176-285, 469-503).
    # Pre-filtering here would only ever DROP targets HEAL's own mask
    # would have kept.
    vehicles = {}
    for vid in visibility:
        if vid == me_id:
            continue  # self-box; HEAL's own object_id union then covers cross-visibility
        # Real per-frame wall occlusion, same wall_geometry check as
        # build_scene_n's visibility loop (user-decided 2026-09-22: always
        # go by real geometry, not the fixed WALL_BLOCKED fact) -- never
        # list a wall-blocked vehicle for THIS agent's own yaml, or HEAL's
        # generic cone-only box_is_visible() (which doesn't know about
        # walls either) could still train it as a hallucinated detection.
        if wall_geometry.blocked_by_wall(
                (raw[me_id]["x"], raw[me_id]["y"]), (raw[vid]["x"], raw[vid]["y"]), WALLS):
            continue
        tgt = corrected[vid]
        tgt_yaw_deg = math.degrees(tgt["yaw_rad"])
        entry = {
            "location": [tgt["x"] * SCALE, tgt["y"] * SCALE, 0.0],
            "center": [0.0, 0.0, QCAR_H / 2.0],
            "extent": [QCAR_L * SCALE / 2.0, QCAR_W * SCALE / 2.0, QCAR_H / 2.0],
            "angle": [0.0, tgt_yaw_deg, 0.0],
            "static": bool(targets.get(vid, {}).get("static", False)),
        }
        # visible_to_agent<id> per ACTIVE agent -- for the --agents 1,2
        # default this produces exactly visible_to_agent1/visible_to_agent2,
        # byte-identical to before this revision.
        for aid in AGENT_IDS:
            entry["visible_to_agent%s" % aid] = visibility[vid]["by_agent"].get(aid)
        vehicles[int(vid)] = entry

    params = {
        "lidar_pose": pose, "lidar_pose_clean": pose,
        "true_ego_pos": pose, "ego_speed": 0.0,
        "vehicles": vehicles,
        "qcar_provenance": {
            "car": p_me["car"], "cav_id": me_id,
            "pose_source": "Vicon, REAL",
            "marker_offset_applied_m": {
                str(vid): list(MARKER_OFFSETS[targets[vid]["vehicle"] if vid in targets
                                    else AGENT_MARKER_OFFSET[vid]])
                for vid in visibility},
            "world_scale": "x10 horizontal only; z/height/K untouched",
            "visibility_rule": "unfiltered scene truth (god's eye); HEAL's own "
                                "per-cav box_is_visible() + cross-cav union decides "
                                "visibility at load time, not this file",
        },
        "camera0": {
            "cords": cam_cords, "extrinsic": np.eye(4).tolist(),
            "intrinsic": p_me["K"].tolist(),
        },
    }
    with open(os.path.join(out_frame_dir, "%s.yaml" % out_frame_num), "w") as f:
        yaml.safe_dump(params, f, default_flow_style=None, sort_keys=False)


def write_scene_gt(raw, corrected, visibility, out_dir, out_frame_num, targets=None):
    """Evaluation oracle: lists a vehicle ONLY when visible to >=1 active
    agent (wall-corrected), empty/absent otherwise -- so a detection is
    scored against "what should be detectable", never against omniscient
    world truth nobody's camera could have produced."""
    targets = static_targets_meta() if targets is None else targets
    os.makedirs(out_dir, exist_ok=True)
    doc = {"vehicles": {}}
    for vid in visibility:
        if not visibility[vid]["any"]:
            continue  # nobody sees it this frame -- correctly absent, not just flagged False
        entry = {
            "type": "static" if targets.get(vid, {}).get("static") else "dynamic",
            "node": targets[vid]["node"] if vid in targets else None,
            "vicon_raw": raw[vid],
            "physical_center": corrected[vid],
        }
        for aid in AGENT_IDS:
            entry["visible_to_agent%s" % aid] = visibility[vid]["by_agent"].get(aid)
        doc["vehicles"][vid] = entry
    json.dump(doc, open(os.path.join(out_dir, "%s.json" % out_frame_num), "w"), indent=2)


def _scene_targets(session, frame_group):
    """Non-agent vehicles of one paired frame: the session's static targets
    and/or its dynamic target, whose pose is the one broadcast in the
    reference (first) agent's frame -- the time reference of the pairing."""
    targets = static_targets_meta() if session.get("static_targets") else {}
    dyn = session.get("dynamic_target")
    if dyn:
        if dyn["id"] in targets or dyn["id"] in AGENT_IDS:
            raise ValueError("session %s: dynamic target id %s collides" % (session["name"], dyn["id"]))
        targets[dyn["id"]] = {"pose": frame_group[AGENT_IDS[0]]["target"], "static": False,
                              "node": dyn.get("node"), "vehicle": dyn["vehicle"]}
    return targets


def main():
    sessions = [s for s in _conf["sessions"] if not _args.sessions
                or s["name"] in _args.sessions.split(",")]
    if not sessions:
        raise SystemExit("No session selected (conf.json sessions / --sessions)")
    for s in sessions:
        if set(s["agent_dirs"]) != set(AGENT_IDS):
            raise SystemExit("session %s: agent_dirs %s != active agents %s"
                             % (s["name"], sorted(s["agent_dirs"]), AGENT_IDS))
    if os.path.isdir(DST):
        shutil.rmtree(DST)
    print("Active agents: %s" % (AGENT_IDS,))
    print("Marker offsets (fwd, lat) m: %s" % {k: [round(v, 5) for v in o] for k, o in MARKER_OFFSETS.items()})

    manifest = {"agents": list(AGENT_IDS), "sessions": [s["name"] for s in sessions],
                "marker_offsets_m": {k: list(v) for k, v in MARKER_OFFSETS.items()},
                "pair_tolerance_sec": PAIR_TOLERANCE_SEC, "pairs": {}, "splits": {}}
    for session in sessions:
        runs = session_runs(session)
        if _args.max_trajectories:
            runs = runs[:_args.max_trajectories]
        val = set(session["val"]) if session.get("val") is not None else VAL_PAIRS
        unknown_val = val - {key for key, _ in runs}
        if unknown_val:
            print("Warning: session %s val names %s, not among its runs" % (session["name"], sorted(unknown_val)))
        print("\nSession %s: %d run(s)" % (session["name"], len(runs)))
        dyn = session.get("dynamic_target")
        for key, run_dirs in runs:
            _build_run(session, key, run_dirs, "validate" if key in val else "train",
                       dyn["file"] if dyn else None, manifest)

    with open(os.path.join(DST, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n" + json.dumps(manifest["splits"], indent=2))


def _build_run(session, key, run_dirs, split, target_file, manifest):
    frames_by_agent = {aid: load_run_frames(run_dirs[aid], target_file) for aid in AGENT_IDS}
    if any(not f for f in frames_by_agent.values()):
        print("skip run %s: missing frames (%s)"
              % (key, {aid: len(f) for aid, f in frames_by_agent.items()}))
        return
    groups = pair_frames_n(frames_by_agent)
    scen_name = session["scenario"].format(index=key, name=re.sub(r"[^A-Za-z0-9]+", "_", key).strip("_"))
    if scen_name in manifest["pairs"]:
        raise ValueError("duplicate scenario name %s" % scen_name)
    n_written = 0
    for k, frame_group in enumerate(groups):
        out_num = "%06d" % k
        targets = _scene_targets(session, frame_group)
        raw, corrected, visibility = build_scene_n(frame_group, targets)
        for me_id in AGENT_IDS:
            me_frame = frame_group[me_id]
            out_dir = os.path.join(DST, split, scen_name, me_id)
            os.makedirs(out_dir, exist_ok=True)
            src_png = os.path.join(me_frame["dir"], "front.png")
            dst_png = os.path.join(out_dir, "%s_camera0.png" % out_num)
            p = CAM_PARAMS[me_id]
            undistort(src_png, dst_png, p["K"], p["D"])
            dst_vis = os.path.join(out_dir, "%s_bev_visibility.png" % out_num)
            try:
                os.link(p["vis_mask"], dst_vis)
            except OSError:
                shutil.copy2(p["vis_mask"], dst_vis)
            build_agent_yaml(me_frame, me_id, raw, corrected, visibility, out_dir, out_num, targets)
        # scene_gt lives OUTSIDE train/validate entirely -- HEAL's own
        # reinitialize() lists every subdirectory of a scenario folder
        # and assumes each one is an integer cav id (crashed on
        # 'scene_gt' when it was nested inside the scenario dir).
        scene_dir = os.path.join(DST, "scene_gt", split, scen_name)
        write_scene_gt(raw, corrected, visibility, scene_dir, out_num, targets)
        n_written += 1
    manifest["pairs"][scen_name] = {
        "session": session["name"], "run": key, "split": split,
        "frames": {aid: len(frames_by_agent[aid]) for aid in AGENT_IDS},
        "paired": n_written,
    }
    manifest["splits"][split] = manifest["splits"].get(split, 0) + n_written
    min_frames = min(len(f) for f in frames_by_agent.values())
    print("%s (%s): %d/%d frames paired (tolerance %.0fms)" % (
        scen_name, split, n_written, min_frames, PAIR_TOLERANCE_SEC * 1000))


if __name__ == "__main__":
    main()
