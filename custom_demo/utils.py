import numpy as np
import re
from scipy.spatial.transform import Rotation as R


def forward_kinematic(offsets, rotations, parents, root_position):
        positions = []
        orientations = []
        for idx, parent in enumerate(parents):
            if parent == -1:
                positions.append(root_position)
                orientations.append(rotations[idx])
            else:
                cur_orientation = R.from_quat(orientations[parent]) * R.from_quat(rotations[idx])
                cur_position = R.from_quat(orientations[parent]).apply(offsets[idx]) + positions[parent]
                orientations.append(cur_orientation.as_quat())
                positions.append(cur_position)
        positions = np.array(positions)
        orientations = np.array(orientations)
        return positions, orientations


def load_bvh(filename):
    
    fh = open(filename, mode='r')
    active = -1
    end_site = False
    
    names = []
    offsets = np.array([]).reshape((0, 3))
    parents = np.array([], dtype=int)
    
    num_frame = -1
    frame_time = -1
    frame_list = []
    
    for line in fh:
        
        if "HIERARCHY" in line:
            continue
    
        if "MOTION" in line:
            continue
        
        rmatch = re.match(r"ROOT (\w+:?\w+)", line)
        if rmatch:
            names.append(rmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue
        
        if "{" in line:
            continue
        
        if "}" in line:
            if end_site:
                end_site = False
            else:
                active = parents[active]
            continue
        
        offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
        if offmatch:
            if not end_site:
                offsets[active] = np.array([list(map(float, offmatch.groups()))])
            continue
        
        jmatch = re.match("\s*JOINT\s+(\w+:?\w+)", line)
        if jmatch:
            names.append(jmatch.group(1))
            offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
            parents = np.append(parents, active)
            active = (len(parents) - 1)
            continue
        
        if "End Site" in line:
            end_site = True
            continue
        
        if 'CHANNELS' in line:
            continue
        
        if "Frames" in line:
            num_frame = int(line.strip().split(' ')[-1])
            continue
            
        if "Frame Time" in line:
            frame_time = float(line.strip().split(' ')[-1])
            continue
        
        line = map(float, line.strip().split(" "))
        frame_list.append(list(line))
            
    fh.close()
    
    data = np.array(frame_list, dtype=np.float32).reshape(num_frame, len(names) + 1, 3)
    
    # F x 3
    video_root_positions = data[:, 0, :]
    
    # F x N x 3
    video_joint_rotations_euler = data[:, 1:, :]
    
    video_joint_positions = []
    video_joint_rotations = []
    video_joint_orientations = []
    
    for idx in range(num_frame):
        
        frame_joint_rotations = R.from_euler(seq='ZYX', angles=video_joint_rotations_euler[idx], degrees=True).as_quat()
        frame_joint_positions, frame_joint_orientations = forward_kinematic(offsets, frame_joint_rotations, parents, video_root_positions[idx])
        video_joint_positions.append(frame_joint_positions)
        video_joint_rotations.append(frame_joint_rotations)
        video_joint_orientations.append(frame_joint_orientations)
        
    video_joint_positions = np.array(video_joint_positions)
    video_joint_rotations = np.array(video_joint_rotations)
    video_joint_orientations = np.array(video_joint_orientations)
   
    return names, parents, offsets, video_joint_positions, video_joint_rotations, video_joint_orientations