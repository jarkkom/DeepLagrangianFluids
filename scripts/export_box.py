#!/usr/bin/env python3

import os
import sys
import argparse
import numpy as np
import json
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'datasets'))
from create_physics_scenes import obj_surface_to_particles, obj_volume_to_particles

def main():
    parser = argparse.ArgumentParser(
        description=
        "Runs a fluid network on the given scene and saves the particle positions as npz sequence"
    )
    parser.add_argument("--scene",
                        type=str,
                        required=True,
                        help="A json file which describes the scene.")
    parser.add_argument("--output",
                        type=str,
                        required=True,
                        help="A json file where scene box points and normals are exported")

    args = parser.parse_args()
    print(args)

    with open(args.scene, 'r') as f:
        scene = json.load(f)

    walls = []
    for x in scene['walls']:
        points, normals = obj_surface_to_particles(x['path'])
        if 'invert_normals' in x and x['invert_normals']:
            normals = -normals
        points += np.asarray([x['translation']], dtype=np.float32)
        walls.append((points, normals))
    box = np.concatenate([x[0] for x in walls], axis=0)
    box_normals = np.concatenate([x[1] for x in walls], axis=0)

    jsondata = {}
    jsondata['points'] = box.tolist()
    jsondata['normals'] = box.tolist()

    with open(args.output, 'w+') as f:
        json.dump(jsondata, f)

if __name__ == '__main__':
    sys.exit(main())
