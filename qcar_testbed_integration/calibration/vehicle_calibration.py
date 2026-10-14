"""Reader for the physical-facts files, shared by every script that needs them.

  calibration/vehicles.json      per-car identity, camera extrinsics, footprint names
  calibration/physical_vehicle_footprints.json
                                 copy of the lab's measurement record: Vicon-origin ->
                                 box-centre offsets and box length/width, checked
                                 against vehicles.json's SHA256
  pipeline/shared/scene.json     walls, parked cars

Only reads; never writes. A missing key or a changed footprints file is an error,
never a silent fallback.
"""
import hashlib
import json
import os

import numpy as np


def _load(path):
    with open(path) as f:
        return json.load(f)


def entries(d):
    return {k: v for k, v in d.items() if not k.startswith("_")}


def load_vehicles(vehicles_json):
    """vehicles.json -> (doc, {agent_id: entry}) with camera extrinsics resolved."""
    doc = _load(vehicles_json)
    doc["_dir"] = os.path.dirname(os.path.abspath(vehicles_json))
    extr = doc["camera_extrinsics"]
    agents = {}
    for aid, v in entries(doc["vehicles"]).items():
        e = extr[v["camera_extrinsic"]]
        agents[aid] = dict(v, R_cb=np.array(e["R_cb"], dtype=np.float64),
                           t_cb=np.array(e["t_cb_m"], dtype=np.float64))
    return doc, agents


def camera_extrinsic(doc, name):
    e = doc["camera_extrinsics"][name]
    return np.array(e["R_cb"], dtype=np.float64), np.array(e["t_cb_m"], dtype=np.float64)


def cam_cords_mount(doc):
    """t_cb_m of the entry that places camera0's cords in every agent yaml."""
    extr = doc["camera_extrinsics"]
    return tuple(extr[extr["cam_cords_mount"]]["t_cb_m"])


def load_footprints(doc):
    """vehicles.json's footprints.file (next to it), refused unless its SHA256 is the
    one vehicles.json acknowledges. Returns {footprint_name: (fwd_m, lat_m, length_m, width_m)}."""
    footprints_json = os.path.join(doc["_dir"], doc["footprints"]["file"])
    with open(footprints_json, "rb") as f:
        digest = hashlib.sha256(f.read()).hexdigest()
    expected = doc["footprints"]["sha256"]
    if digest != expected:
        raise SystemExit(
            "%s has SHA256 %s, vehicles.json acknowledges %s. The footprints were "
            "re-measured or edited: review the change, then update footprints.sha256 "
            "in vehicles.json." % (footprints_json, digest, expected))
    out = {}
    for name, v in _load(footprints_json)["vehicles"].items():
        m = v["marker_to_center"]
        if abs(m.get("yaw_rad", 0.0)) > 1e-9:
            raise SystemExit("footprint %r has a marker->vehicle yaw; not supported" % name)
        out[name] = (m["x_m"], m["y_m"], v["length_m"], v["width_m"])
    return out


def load_scene(scene_json):
    """scene.json -> (walls list, static_targets dict, {parked_name: (fwd, lat, l, w)})."""
    doc = _load(scene_json)
    parked = {k: (v["marker_to_center_fwd_m"], v["marker_to_center_lat_m"], v["length_m"], v["width_m"])
              for k, v in entries(doc["parked_vehicles"]).items()}
    return doc["walls"], entries(doc["static_targets"]), parked


def common_box_size(doc, *size_tables):
    """(length, width, height) shared by every vehicle in the given
    {name: (fwd, lat, length, width)} tables; height from vehicles.json. The box
    writers use one size for all cars, so different sizes are an error."""
    sizes = {(v[2], v[3]) for table in size_tables for v in table.values()}
    if len(sizes) != 1:
        raise SystemExit("vehicles have different box sizes %s; per-vehicle sizes "
                         "are not wired into the box writers yet" % sorted(sizes))
    (length, width), = sizes
    return length, width, doc["footprints"]["box_height_m"]
