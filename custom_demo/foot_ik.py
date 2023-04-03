from argparse import ArgumentParser
import numpy as np
import copy
import re
import pickle
from scipy.spatial.transform import Rotation as R

import mmcv



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
    
    data = np.array(frame_list).reshape(num_frame, len(names) + 1, 3)
    
    # F x 3
    video_root_positions = data[:, 0, :]
    
    # F x N x 3
    video_joint_rotations_euler = data[:, 1:, :]
    
    # N x 3, mm => m
    offsets = offsets / 1000.0
    
    video_joint_positions = []
    video_joint_rotations = []
    video_joint_orientations = []
    
    for idx in range(num_frame):
        
        frame_joint_rotations = R.from_euler(seq='ZYX', angles=video_joint_rotations_euler[idx], degrees=True).as_quat()
        frame_joint_positions, frame_joint_orientations = fk(offsets, frame_joint_rotations, video_root_positions[idx])
        video_joint_positions.append(frame_joint_positions)
        video_joint_rotations.append(frame_joint_rotations)
        video_joint_orientations.append(frame_joint_orientations)
        
    video_joint_positions = np.array(video_joint_positions)
    video_joint_rotations = np.array(video_joint_rotations)
    video_joint_orientations = np.array(video_joint_orientations)
   
    return names, parents, offsets, video_joint_positions, video_joint_rotations, video_joint_orientations


def fk(offsets, rotations, root_position):
    positions = []
    orientations = []
    for i in range(len(offsets)):
        if i == 0:
            positions.append(root_position)
            orientations.append(rotations[i])
        else:
            cur_orientation = R.from_quat(orientations[i - 1]) * R.from_quat(rotations[i])
            cur_position = R.from_quat(orientations[i - 1]).apply(offsets[i]) + positions[i - 1]
            orientations.append(cur_orientation.as_quat())
            positions.append(cur_position)
    positions = np.array(positions)
    orientations = np.array(orientations)
    return positions, orientations


def cal_theta_of_two_vec(vec1, vec2):
    normalize_vec1 = vec1 / np.linalg.norm(vec1)
    normalize_vec2 = vec2 / np.linalg.norm(vec2)
    theta = np.arccos(np.clip(np.dot(normalize_vec1, normalize_vec2), -1, 1))
    return theta


def cal_theta_by_length(lc, la, lb):
    # lc、la、lb的长度可能构成不了三角形
    cos_theta = np.clip((lc ** 2 - la ** 2 - lb ** 2) / (-2 * la * lb), -1, 1)
    theta = np.arccos(cos_theta)
    return theta


def two_bone_ik(joint_offsets, joint_positions, joint_orientations, target_pos):
    
    assert len(joint_offsets) == len(joint_positions) == len(joint_orientations) == 3
    new_joint_positions = copy.deepcopy(joint_positions)
    new_joint_orientations = copy.deepcopy(joint_orientations)
    
    a, b, c = joint_positions
    t = target_pos
    
    len_ab = np.linalg.norm(b - a)
    len_bc = np.linalg.norm(c - b)
    len_at = np.linalg.norm(t - a)
    
    if (len_at > len_ab + len_bc) or (len_at < abs(len_ab - len_bc)):
        print(f'len_ab:{len_ab}, leb_bc:{len_bc}, len_at:{len_at}')
        raise ValueError("Couldn't close to target")
    
    # step 1: rotate b such that |ac| = |at|.
    theta_ba_bc_0 = cal_theta_of_two_vec(a - b, c - b)
    theta_ba_bc_1 = cal_theta_by_length(len_at, len_ab, len_bc)

    axis0 = np.cross(c - b, a - b) / np.linalg.norm(np.cross(c - b, a - b))
    rot1 = (-theta_ba_bc_1 + theta_ba_bc_0) * axis0
    rot1 = R.from_rotvec(rot1)
    
    for i in range(1, 3):
        new_joint_orientations[i] = rot1 * new_joint_orientations[i]
        new_joint_positions[i] = new_joint_orientations[i-1].apply(joint_offsets[i]) + new_joint_positions[i-1]
        
    # step 2: rotate a such c lands in t.
    c = new_joint_positions[-1]
    theta_at_ac_0 = cal_theta_of_two_vec(t - a, c - a)
    axis1 = np.cross(t - a, c - a) / np.linalg.norm(np.cross(t - a, c - a))
    rot2 = -theta_at_ac_0 * axis1
    rot2 = R.from_rotvec(rot2)

    for i in range(0, 3):
        new_joint_orientations[i] = rot2 * new_joint_orientations[i]
        if i != 0:
            new_joint_positions[i] = new_joint_orientations[i-1].apply(joint_offsets[i]) + new_joint_positions[i-1]
        
    new_joint_rotations = []
    for i in range(3):
        if i == 0:
            new_joint_rotations.append(new_joint_orientations[i])
        else:
            new_joint_rotations.append(R.inv(new_joint_orientations[i-1]) * new_joint_orientations[i])
            
    return new_joint_positions, new_joint_rotations, new_joint_orientations
    
    
def foot_ik(offsets, frame_joint_positions, frame_joint_orientations, indices):
    
    # 1. get joint_offsets, joint_positions, joint_orientations, target_pos
    joint_offsets = offsets[indices]
    joint_positions = frame_joint_positions[indices]
    joint_orientations = frame_joint_orientations[indices]
    target_pos = cal_target_pos(joint_positions[-1])
    
    # 2. two bone ik
    new_joint_positions, new_joint_rotations, new_joint_orientations = two_bone_ik(joint_offsets, joint_positions, joint_orientations, target_pos)
    print(joint_positions, new_joint_positions)
    exit()
    
    # 3. foot rotation
    pass


def cal_target_pos(end_effector, ground, height=77/1000.0):
    target_pos = np.array(end_effector)
    target_pos[1] -= height
    # print(end_effector, target_pos)
    # exit()
    return target_pos


def get_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--bvh-file', required=True)
    parser.add_argument('--footcontact-pkl-file', required=True)
    parser.add_argument('--track-id', default=0, type=int)
    
    return parser


def main(args):
    # use bvh file and foot-contact file
    names, parents, offsets, video_joint_positions, video_joint_rotations, video_joint_orientations = load_bvh(args.bvh_file)
    
    index_Left_hip = names.index('Left_hip')
    index_Left_knee = names.index('Left_knee')
    index_Left_foot = names.index('Left_foot')
    
    index_Right_hip = names.index('Right_hip')
    index_Right_knee = names.index('Right_knee')
    index_Right_foot = names.index('Right_foot')
    
    left_indices = [index_Left_hip, index_Left_knee, index_Left_foot]
    right_indices = [index_Right_hip, index_Right_knee, index_Right_foot]
    
    with open(args.footcontact_pkl_file, 'rb') as fin:
        footcontact = pickle.load(fin)
    
    # footcontact中包含了多个被追踪人的触地状况
    footcontact = footcontact[args.track_id]
    assert len(video_joint_positions) == len(footcontact)
    
    for frame_id, single_frame_contact in enumerate(mmcv.track_iter_progress(footcontact)):
        # res: [left_heel, left_toes, right_heel, right_toes]
        res = single_frame_contact[2]
        frame_joint_positions = video_joint_positions[frame_id]
        frame_joint_orientations = video_joint_orientations[frame_id]
        
        if res[0] and res[1]:
            # 左脚触地
            foot_ik(offsets, frame_joint_positions, frame_joint_orientations, left_indices)
        
        if res[2] and res[3]:
            # 右脚触地
            foot_ik(offsets, frame_joint_positions, frame_joint_orientations, right_indices)
    


if __name__ == '__main__':
    
    # offsets = np.random.rand(3, 3) * 100
    # rotations = R.from_quat(np.random.rand(3, 4))
    # root_position = np.random.rand(3) * 10
    # target_position = np.random.rand(3) * 100
    
    # positions, orientations = fk(offsets, rotations, root_position)

    # new_positions, new_rotations, _ = two_bone_ik(offsets, positions, orientations, target_position)
    # print(target_position, new_positions[-1])
    
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    
    # names, parents, offsets, video_joint_positions, video_joint_rotations, video_joint_orientations = load_bvh('../video2bvh/output.bvh')
    
    # with open("tmp.pkl", "wb") as fout:
    #     pickle.dump({
    #         "names": names,
    #         "parents": parents,
    #         "offsets": offsets,
    #         "video_joint_positions": video_joint_positions,
    #         "video_joint_rotations": video_joint_rotations,
    #         "video_joint_orientations": video_joint_orientations
    #     }, fout)