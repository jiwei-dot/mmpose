from argparse import ArgumentParser
import numpy as np
import copy
import os
import pickle
from scipy.spatial.transform import Rotation as R

import mmcv

from utils import load_bvh



# def load_bvh(filename):
    
#     fh = open(filename, mode='r')
#     active = -1
#     end_site = False
    
#     names = []
#     offsets = np.array([]).reshape((0, 3))
#     parents = np.array([], dtype=int)
    
#     num_frame = -1
#     frame_time = -1
#     frame_list = []
    
#     for line in fh:
        
#         if "HIERARCHY" in line:
#             continue
    
#         if "MOTION" in line:
#             continue
        
#         rmatch = re.match(r"ROOT (\w+:?\w+)", line)
#         if rmatch:
#             names.append(rmatch.group(1))
#             offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
#             parents = np.append(parents, active)
#             active = (len(parents) - 1)
#             continue
        
#         if "{" in line:
#             continue
        
#         if "}" in line:
#             if end_site:
#                 end_site = False
#             else:
#                 active = parents[active]
#             continue
        
#         offmatch = re.match(r"\s*OFFSET\s+([\-\d\.e]+)\s+([\-\d\.e]+)\s+([\-\d\.e]+)", line)
#         if offmatch:
#             if not end_site:
#                 offsets[active] = np.array([list(map(float, offmatch.groups()))])
#             continue
        
#         jmatch = re.match("\s*JOINT\s+(\w+:?\w+)", line)
#         if jmatch:
#             names.append(jmatch.group(1))
#             offsets = np.append(offsets, np.array([[0, 0, 0]]), axis=0)
#             parents = np.append(parents, active)
#             active = (len(parents) - 1)
#             continue
        
#         if "End Site" in line:
#             end_site = True
#             continue
        
#         if 'CHANNELS' in line:
#             continue
        
#         if "Frames" in line:
#             num_frame = int(line.strip().split(' ')[-1])
#             continue
            
#         if "Frame Time" in line:
#             frame_time = float(line.strip().split(' ')[-1])
#             continue
        
#         line = map(float, line.strip().split(" "))
#         frame_list.append(list(line))
            
#     fh.close()
    
#     data = np.array(frame_list, dtype=np.float32).reshape(num_frame, len(names) + 1, 3)
    
#     # F x 3
#     video_root_positions = data[:, 0, :]
    
#     # F x N x 3
#     video_joint_rotations_euler = data[:, 1:, :]
    
#     # N x 3, mm => m
#     offsets = offsets / 1000.0
    
#     video_joint_positions = []
#     video_joint_rotations = []
#     video_joint_orientations = []
    
#     for idx in range(num_frame):
        
#         frame_joint_rotations = R.from_euler(seq='ZYX', angles=video_joint_rotations_euler[idx], degrees=True).as_quat()
#         frame_joint_positions, frame_joint_orientations = fk(offsets, frame_joint_rotations, parents, video_root_positions[idx])
#         video_joint_positions.append(frame_joint_positions)
#         video_joint_rotations.append(frame_joint_rotations)
#         video_joint_orientations.append(frame_joint_orientations)
        
#     video_joint_positions = np.array(video_joint_positions)
#     video_joint_rotations = np.array(video_joint_rotations)
#     video_joint_orientations = np.array(video_joint_orientations)
   
#     return names, parents, offsets, video_joint_positions, video_joint_rotations, video_joint_orientations


def save_bvh(args, root_positions_list, rotations_list, seq='ZYX'):
    
    new_filename = 'new_' + os.path.basename(args.bvh_file)
    output_path = os.path.join(args.out_root, new_filename)
    
    with open(output_path, 'w') as fout:
        with open(args.bvh_file, 'r') as fin:
            for line in fin:
                s_line = line.strip()
                if not s_line.startswith('MOTION'):
                    fout.write(line)
                else:
                    break
        fout.write('MOTION\n')
        fout.write(f'Frames: {len(rotations_list)}\n')
        fout.write('Frame Time: 0.03333333333333333\n')
        for idx in range(len(rotations_list)):
            fout.write(f'{root_positions_list[idx][0]} {root_positions_list[idx][1]} {root_positions_list[idx][2]} ')
            rotation = rotations_list[idx]
            for rot in rotation:
                euler = R.from_quat(rot).as_euler(seq=seq, degrees=True)
                fout.write(f'{euler[0]} {euler[1]} {euler[2]} ')
            fout.write('\n')
            
    print('convert success!')


def fk(offsets, rotations, parents, root_position):
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


def two_bone_ik(joint_offsets, joint_positions, joint_rotations, joint_orientations, target_pos):
    
    assert len(joint_offsets) == len(joint_positions) == len(joint_orientations) == 3
    
    new_joint_positions = copy.deepcopy(joint_positions)
    new_joint_rotations = copy.deepcopy(joint_rotations)
    new_joint_orientations = copy.deepcopy(joint_orientations)
    
    a, b, c = joint_positions
    t = target_pos
    
    len_ab = np.linalg.norm(b - a)
    len_bc = np.linalg.norm(c - b)
    len_at = np.linalg.norm(t - a)
    
    # step 0: judge 
    if (len_ab + len_bc) < len_at or (len_ab + len_at) < len_bc or (len_bc + len_at) < len_ab: 
        print(f'len_ab:{len_ab}, leb_bc:{len_bc}, len_at:{len_at}')
        # raise ValueError("Couldn't close to target")
        print("Couldn't close to target")
        return joint_positions, joint_rotations, joint_orientations
    
    # step 1: rotate b such that |ac| = |at|.
    theta_ba_bc_0 = cal_theta_of_two_vec(a - b, c - b)
    theta_ba_bc_1 = cal_theta_by_length(len_at, len_ab, len_bc)

    axis0 = np.cross(c - b, a - b) / np.linalg.norm(np.cross(c - b, a - b))
    rot1 = (-theta_ba_bc_1 + theta_ba_bc_0) * axis0
    rot1 = R.from_rotvec(rot1)
    
    for i in range(1, 3):
        new_joint_orientations[i] = (rot1 * R.from_quat(new_joint_orientations[i])).as_quat()
        new_joint_positions[i] = R.from_quat(new_joint_orientations[i-1]).apply(joint_offsets[i]) + new_joint_positions[i-1]
        
    # step 2: rotate a such c lands in t.
    c = new_joint_positions[-1]
    theta_at_ac_0 = cal_theta_of_two_vec(t - a, c - a)
    axis1 = np.cross(t - a, c - a) / np.linalg.norm(np.cross(t - a, c - a))
    rot2 = -theta_at_ac_0 * axis1
    rot2 = R.from_rotvec(rot2)

    for i in range(0, 3):
        new_joint_orientations[i] = (rot2 * R.from_quat(new_joint_orientations[i])).as_quat()
        if i != 0:
            new_joint_positions[i] = R.from_quat(new_joint_orientations[i-1]).apply(joint_offsets[i]) + new_joint_positions[i-1]
        
    new_joint_rotations = []
    for i in range(3):
        if i == 0:
            new_joint_rotations.append(new_joint_orientations[i])
        else:
            new_joint_rotations.append((R.inv(R.from_quat(new_joint_orientations[i-1])) * R.from_quat(new_joint_orientations[i])).as_quat())
            
    return new_joint_positions, new_joint_rotations, new_joint_orientations
    
    
def foot_ik(frame_id, offsets, frame_joint_positions, frame_joint_rotations, frame_joint_orientations, indices, foot_type):
    
    foot_height = cal_foot_height(frame_joint_positions, foot_type)
    
    # 1. get joint_offsets, joint_positions, joint_orientations, target_pos
    joint_offsets = offsets[indices]
    joint_positions = frame_joint_positions[indices]
    joint_rotations = frame_joint_rotations[indices]
    joint_orientations = frame_joint_orientations[indices]
    target_pos = cal_target_pos(joint_positions[-1], foot_height)
    
    # 2. two bone ik
    new_joint_positions, new_joint_rotations, new_joint_orientations = \
        two_bone_ik(joint_offsets, joint_positions, joint_rotations, joint_orientations, target_pos)

    # 3. foot rotation, rotate foot such that big(small) toe and heel are contact ground 
    # todo
    return new_joint_positions, new_joint_rotations, new_joint_orientations


def cal_target_pos(end_effector, height):
    target_pos = np.array(end_effector)
    target_pos[-1] = height
    return target_pos


def cal_foot_height(frame_joint_positions, foot_type):
    assert foot_type in ('left', 'right')
    if foot_type == 'left':
        point_indices = 3
        plane_indices = [4, 5, 6]
    else:
        point_indices = 9
        plane_indices = [10, 11, 12]
    
    frame_joint_positions = frame_joint_positions[..., :3]
    vec1 = frame_joint_positions[plane_indices[2]] - frame_joint_positions[plane_indices[0]]
    vec2 = frame_joint_positions[plane_indices[2]] - frame_joint_positions[plane_indices[1]]
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    vec1 = vec1 / (norm_vec1 + 1e-5)
    vec2 = vec2 / (norm_vec2 + 1e-5)
    vec3 = np.cross(vec1, vec2)
    # plane: Ax + By + Cz + D = 0
    # (A, B, C) = vec3
    D = -np.dot(frame_joint_positions[plane_indices[0]], vec3)
    # print(vec3, D)
    # print(np.dot(vec3, frame_joint_positions[plane_indices[1]]) + D)
    # print(np.dot(vec3, frame_joint_positions[plane_indices[2]]) + D)
    dist = np.abs(np.dot(frame_joint_positions[point_indices], vec3) + D) / (np.linalg.norm(vec3) + 1e-5)
    return dist


def get_parser():
    parser = ArgumentParser()
    
    parser.add_argument('--bvh-file', required=True)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--footcontact-pkl-file', required=True)
    parser.add_argument('--track-id', default=0, type=int)
    
    return parser


def main(args):

    # video_joint_rotations:    相对旋转
    # video_joint_orientations: 绝对旋转
    names, parents, offsets, video_joint_positions, \
        video_joint_rotations, video_joint_orientations = load_bvh(args.bvh_file)
    
    ground_height = np.min(video_joint_positions[:, :, -1])
    video_joint_positions[:, :, -1] -= ground_height
    
    index_Left_hip = names.index('left_hip')
    index_Left_knee = names.index('left_knee')
    index_Left_foot = names.index('left_foot')
    
    index_Right_hip = names.index('right_hip')
    index_Right_knee = names.index('right_knee')
    index_Right_foot = names.index('right_foot')
    
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
        frame_joint_rotations = video_joint_rotations[frame_id]
        frame_joint_orientations = video_joint_orientations[frame_id]
        
        if res[0] and res[1]:
            # print(video_joint_positions[frame_id][[4, 5, 6]])
            # 左脚触地
            new_joint_positions, new_joint_rotations, new_joint_orientations = \
                foot_ik(frame_id, offsets, frame_joint_positions, frame_joint_rotations, frame_joint_orientations, left_indices, 'left')
            frame_joint_positions[left_indices] = new_joint_positions
            frame_joint_rotations[left_indices] = new_joint_rotations
            # video_joint_orientations[left_indices] = new_joint_orientations
            
        if res[2] and res[3]:
            # print(video_joint_positions[frame_id][[10, 11, 12]])
            # 右脚触地
            new_joint_positions, new_joint_rotations, new_joint_orientations = \
                foot_ik(frame_id, offsets, frame_joint_positions, frame_joint_rotations, frame_joint_orientations, right_indices, 'right')
            frame_joint_positions[right_indices] = new_joint_positions
            frame_joint_rotations[right_indices] = new_joint_rotations
            # video_joint_orientations[right_indices] = new_joint_orientations
            
    save_bvh(args, video_joint_positions[:, 0, :], video_joint_rotations)
    

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