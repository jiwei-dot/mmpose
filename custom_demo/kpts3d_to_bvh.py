from argparse import ArgumentParser
import pickle
import copy
import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch import optim
from utils import load_bvh
from scipy.spatial.transform import Rotation as R
from mmcv import Config
from mmpose.datasets import DatasetInfo
    

class ForwardKinematicQuat(nn.Module):

    def __init__(self, num_joints=65, device='cpu'):
        super().__init__()
        self.num_joints = num_joints
        self.device = device
        self.rotations = self._init_parameters()
        weight = nn.Parameter(self.rotations, requires_grad=True)
        self.register_parameter('weight', weight)

    def _init_parameters(self):
        params = torch.zeros((self.num_joints, 4), device=self.device)
        params.data[:, -1] = 1.0
        return params
    
    def reset_parameters(self):
        value = torch.zeros((self.num_joints, 4), device=self.device)
        value[:, -1] = 1.0
        with torch.no_grad():
            self.weight.data = value

    def forward(self, joint_offsets, joint_parents, root_position):
        joint_positions = []
        joint_orientations = []

        for index, parent in enumerate(joint_parents):
            if parent == -1:
                joint_orientations.append(self.weight[index])
                joint_positions.append(root_position)
            else:
                cur_orientation = self.quaternion_mult(joint_orientations[parent], self.weight[index])
                joint_orientations.append(cur_orientation)
                cur_position = self.quaternion_rotate(joint_orientations[parent], joint_offsets[index]) + \
                               joint_positions[parent]
                joint_positions.append(cur_position)

        joint_positions = torch.stack(joint_positions, dim=0)
        
        return joint_positions

    def quaternion_mult(self, quat1, quat2):

        quat1 = quat1 / (torch.norm(quat1) + 1e-5)
        quat2 = quat2 / (torch.norm(quat2) + 1e-5)

        px, py, pz, pw = quat1
        qx, qy, qz, qw = quat2

        rx = pw * qx + px * qw + py * qz - pz * qy
        ry = pw * qy - px * qz + py * qw + pz * qx
        rz = pw * qz + px * qy - py * qx + pz * qw
        rw = pw * qw - px * qx - py * qy - pz * qz

        return torch.stack([rx, ry, rz, rw])

    def quaternion_rotate(self, quat, vec):

        quat = quat / (torch.norm(quat) + 1e-5)

        x, y, z, w = quat

        x2 = x * x
        y2 = y * y
        z2 = z * z
        w2 = w * w

        xy = x * y
        zw = z * w
        xz = x * z
        yw = y * w
        yz = y * z
        xw = x * w

        m00 = x2 - y2 - z2 + w2
        m10 = 2 * (xy + zw)
        m20 = 2 * (xz - yw)

        m01 = 2 * (xy - zw)
        m11 = - x2 + y2 - z2 + w2
        m21 = 2 * (yz + xw)

        m02 = 2 * (xz + yw)
        m12 = 2 * (yz - xw)
        m22 = - x2 - y2 + z2 + w2
        
        m = torch.stack([m00, m01, m02, m10, m11, m12, m20, m21, m22]).reshape(-1, 3)

        return torch.matmul(m, vec)


class Criterion(nn.Module):
    def __init__(self, weight, device, loss_type='mse'):
        super().__init__()
        self.num = np.count_nonzero(weight)
        self.weight = torch.from_numpy(weight).to(device)
        assert loss_type in ('mse', 'l1', 'smoothl1')
        if loss_type == 'mse':
            self.loss = nn.MSELoss(reduction='none')
        elif loss_type == 'l1':
            self.loss = nn.L1Loss(reduction='none')
        else:
            self.loss = nn.SmoothL1Loss(reduction='none')
        
    def forward(self, pred, gt):
        loss = self.loss(pred, gt)
        loss = loss * self.weight
        loss = loss.sum()
        loss = loss / self.num
        return loss
        
        
def fill_list(video_list):
    new_video_list = copy.deepcopy(video_list)
    i = len(new_video_list) - 1
    j = i
    
    while i >= 0:
        if new_video_list[i] is not None:
            i -= 1
        else:
            j = i - 1
            while j >= 0 and new_video_list[j] is None:
                j -= 1
            
            if j >= 0:
                for k in range(j + 1, i + 1):
                    new_video_list[k] = new_video_list[j]
                i = j - 1
            else:
                if i + 1 >= len(video_list):
                    print(f"{video_list} only have None")
                    raise ValueError
                for k in range(0, i + 1):
                    new_video_list[k] = new_video_list[i + 1]
                break
            
    return new_video_list


def get_video_kpts3d(pkl_file, need_track_id=0):
    with open(pkl_file, 'rb') as fin:
        video_list = pickle.load(fin)
        
    data_list = [None] * len(video_list)
    for idx, frame in enumerate(video_list):
        for p in frame:
            track_id = p['track_id']
            kpts3d = p['keypoints_3d']
            if track_id != need_track_id:
                continue
            data_list[idx] = kpts3d
            
    filled_data_list = fill_list(data_list)
    video_kpts3d = np.array(filled_data_list)[..., :3]
    
    # 相机坐标系转化到BVH坐标系
    # 相机坐标系: x向右，y向下，z向前
    # BVH坐标系: x向右，y向前，z向上
    # video_kpts3d[..., [0, 1, 2]] = -video_kpts3d[..., [0, 2, 1]]
    # video_kpts3d[..., 2] = -video_kpts3d[..., 2]
    return video_kpts3d


def adjust_order(video_kpts3d, bvh_idx2name, kpts_name2idx):
    """
        把网络输出的关键点顺序调整为BVH中定义的顺序
    """
    new_order = [kpts_name2idx[n] for i, n in bvh_idx2name.items()]
    video_kpts3d[:, :, :] = video_kpts3d[:, new_order, :]
    return video_kpts3d
        

def get_flip_skeletons():
    flip_skeletons = [
        [(0, 1), (0, 4)],
        [(1, 2), (4, 5)],
        [(2, 3), (5, 6)],
        [(3, 62), (6, 59)],
        [(3, 63), (6, 60)],
        [(3, 64), (6, 61)],
        [(8, 14), (8, 11)],
        [(14, 15), (11, 12)],
        [(15, 16), (12, 13)],
        [(16, 38), (13, 17)],
        [(38, 39), (17, 18)],
        [(39, 40), (18, 19)],
        [(40, 41), (19, 20)],
        [(41, 42), (20, 21)],
        [(38, 43), (17, 22)],
        [(43, 44), (22, 23)],
        [(44, 45), (23, 24)],
        [(45, 46), (24, 25)],
        [(38, 47), (17, 26)],
        [(47, 48), (26, 27)],
        [(48, 49), (27, 28)],
        [(49, 50), (28, 29)],
        [(38, 51), (17, 30)],
        [(51, 52), (30, 31)],
        [(52, 53), (31, 32)],
        [(53, 54), (32, 33)],
        [(38, 55), (17, 34)],
        [(55, 56), (34, 35)],
        [(56, 57), (35, 36)],
        [(57, 58), (36, 37)]
    ]
    return flip_skeletons
 

def cal_bone_len(video_kpts3d, skeletons):
    """
        根据所有帧计算骨骼长度
    """
    skeleton_len = dict()
    
    for skeleton in skeletons:
        point_a = video_kpts3d[:, skeleton[0], :]
        point_b = video_kpts3d[:, skeleton[1], :]
        dist = np.linalg.norm(point_a - point_b, axis=-1).mean()
        skeleton_len[tuple(skeleton)] = dist
         
    flip_skeletons = get_flip_skeletons()    
    for flip_skeleton in flip_skeletons:
        mean_dist = (skeleton_len[tuple(flip_skeleton[0])] + skeleton_len[tuple(flip_skeleton[1])]) / 2
        skeleton_len[tuple(flip_skeleton[0])] = mean_dist
        skeleton_len[tuple(flip_skeleton[1])] = mean_dist
        
    return skeleton_len


def adjust_video_kpts3d(video_kpts3d, skeleton_len):
    # print(np.linalg.norm(video_kpts3d[10][17] - video_kpts3d[10][18]))
    new_video_kpts3d = []
    for frame_kpts3d in video_kpts3d:
        new_frame_kpts3d = np.zeros_like(frame_kpts3d)
        new_frame_kpts3d[0] = frame_kpts3d[0]
        
        skeleton_keys = list(skeleton_len.keys())
        skeleton_keys = sorted(skeleton_keys, key=lambda x: x[0])
        for skeleton in skeleton_keys:
            a, b = skeleton
            direct_ab = (frame_kpts3d[b] - frame_kpts3d[a]) / np.linalg.norm(frame_kpts3d[b] - frame_kpts3d[a])
            new_frame_kpts3d[b] = new_frame_kpts3d[a] + direct_ab * skeleton_len[skeleton]
        new_video_kpts3d.append(new_frame_kpts3d)
    new_video_kpts3d = np.array(new_video_kpts3d)
    # print(np.linalg.norm(new_video_kpts3d[10][17] - new_video_kpts3d[10][18]))
    return new_video_kpts3d
                
        
def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--pkl-path', required=True)
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--template-bvh', default='custom_demo/bvh/direction.bvh')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--track-id', type=int, default=0)
    return parser


def train_on_single_frame(args, frame_index, model, criterion, offsets_np, parents, gt_np):
    
    # if frame_index % 100 == 0:
    model.reset_parameters()

    # model.train()
    # optimizer = optim.Adam(model.parameters(), lr=1)
    optimizer = optim.LBFGS(model.parameters(), lr=1, line_search_fn="strong_wolfe")
    
    pre_loss_e = None
    min_loss_diff = 1e-5
    loss_e = 1e5
    epoch = 0
    max_epoch = 100
    
    offsets = torch.from_numpy(offsets_np).to(dtype=torch.float32, device=args.device)
    gt = torch.from_numpy(gt_np).to(dtype=torch.float32, device=args.device)
    root_position = gt[0]
    
    while loss_e > 1e-3 and epoch < max_epoch:
        
        if isinstance(optimizer, optim.LBFGS):
            def closure():
                pred = model(offsets, parents, root_position)
                loss = criterion(pred, gt)
                optimizer.zero_grad()
                loss.backward()
                return loss
            loss = optimizer.step(closure)
        elif isinstance(optimizer, (optim.SGD, optim.Adam)):
            pred = model(offsets, parents, root_position)
            loss = criterion(pred, gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        else:
            raise NotImplemented
        
        loss_e = loss.item()
        print(frame_index, epoch, loss_e)
        # print(model.weight.grad)
        epoch += 1
        
        if pre_loss_e is not None:
            # 这里暂时有bug
            if loss_e == pre_loss_e and epoch == 1:
                model.reset_parameters()
                optimizer = optim.LBFGS(model.parameters(), history_size=100, max_iter=5, lr=1, line_search_fn="strong_wolfe")
            elif loss_e <= pre_loss_e and pre_loss_e - loss_e < min_loss_diff:
                break
            
        pre_loss_e = loss_e
        
    return model.weight.detach().cpu().numpy()


def save_bvh(args, offsets, root_positions_list, rotations_list, seq='ZYX'):
    
    offsets = offsets * 1000
    video_name = args.pkl_path.split('/')[-1].split('_')[1]
    output = os.path.join(args.out_root, f'{video_name}.bvh')
    offset_idx = 0
    last_offset = None
    
    with open(output, 'w') as fout:
        with open(args.template_bvh, 'r') as fin:
            while True:
                line = fin.readline()
                s_line = line.strip()
                if s_line.startswith('MOTION'):
                    break
                if s_line.startswith('ROOT') or s_line.startswith('JOINT'):
                    fout.write(line)
                    line = fin.readline()
                    fout.write(line)
                    line = fin.readline()
                    num_whitespace = line.find('OFFSET')
                    new_line = ' ' * num_whitespace + 'OFFSET '
                    new_line += ' '.join([str(t) for t in offsets[offset_idx]])
                    last_offset = ' '.join([str(t) for t in offsets[offset_idx]])
                    new_line += '\n'
                    fout.write(new_line)
                    offset_idx += 1
                elif s_line.startswith('End Site'):
                    fout.write(line)
                    line = fin.readline()
                    fout.write(line)
                    line = fin.readline()
                    num_whitespace = line.find('OFFSET')
                    new_line = ' ' * num_whitespace + 'OFFSET '
                    new_line += last_offset
                    new_line += '\n'
                    fout.write(new_line)
                    line = fin.readline()
                    fout.write(line) 
                else:
                    fout.write(line)
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


def main(args):
    
    h36m_wo_face_dataset_info = 'configs/_base_/datasets/h36m_wo_face.py'
    cfg = Config.fromfile(h36m_wo_face_dataset_info)
    h36m_wo_face_dataset_info = DatasetInfo(cfg.dataset_info)
    
    joint_names, joint_parents, joint_offsets, _, _, _ = load_bvh(args.template_bvh) 
    
    # root + 躯干部位 + 左臂 + 左手 + 右臂 + 右手
    # joint_weights = [0.0] + [1.0] * 16 + [1.0] * 3 + [10.0] * 21 + [1.0] * 3 + [10.0] * 21
    joint_weights = [0.0] + [1.0] * 64
    # joint_weights = [0.0] + [1.0] * 16 + [1.0] * 3 + [0.0] * 21 + [1.0] * 3 + [0.0] * 21
    joint_weights_np = np.array(joint_weights, dtype=np.float32)[:, np.newaxis]

    kpts_idx2name = h36m_wo_face_dataset_info.keypoint_id2name
    kpts_name2idx = h36m_wo_face_dataset_info.keypoint_name2id
    bvh_idx2name = {idx: name for idx, name in enumerate(joint_names)}
    bvh_name2idx = {name: idx for idx, name in bvh_idx2name.items()}
    
    video_kpts3d = get_video_kpts3d(args.pkl_path, args.track_id)
    skeleton_len = cal_bone_len(video_kpts3d, h36m_wo_face_dataset_info.skeleton)
    video_kpts3d = adjust_video_kpts3d(video_kpts3d, skeleton_len)

    new_offsets_len = [0.0] * len(joint_offsets)
    for idx, parent in enumerate(joint_parents):
        if parent == -1:
            continue
        else:
            cur_bvh_name = bvh_idx2name[idx]
            par_bvh_name = bvh_idx2name[parent]
            cur_kpts_idx = kpts_name2idx[cur_bvh_name]
            par_kpts_idx = kpts_name2idx[par_bvh_name]
            k = (cur_kpts_idx, par_kpts_idx) if cur_kpts_idx < par_kpts_idx else (par_kpts_idx, cur_kpts_idx)
            new_offsets_len[idx] = skeleton_len[k]
    joint_offsets[1:] = joint_offsets[1:] / np.linalg.norm(joint_offsets[1:], axis=-1, keepdims=True) * np.array(new_offsets_len)[1:, np.newaxis]
    
    video_kpts3d = adjust_order(video_kpts3d, bvh_idx2name, kpts_name2idx)
    
    model = ForwardKinematicQuat(num_joints=len(joint_names), device=args.device)
    model.train()
    model.to(args.device)
    
    
    # criterion = nn.MSELoss(reduction='mean').to(args.device)
    # criterion = nn.L1Loss(reduction='mean').to(args.device)
    # criterion = nn.SmoothL1Loss(reduction='none').to(args.device)
    criterion = Criterion(joint_weights_np, args.device, 'mse')
    
    video_rotations = []
    video_root_positions = []
    
    # 一帧一帧地处理
    for frame_index, frame_kpts3d_np in enumerate(tqdm(video_kpts3d)):
        # 坐标系转换
        frame_kpts3d_np = frame_kpts3d_np[:, [0, 2, 1]]
        frame_kpts3d_np[..., 2] = -frame_kpts3d_np[..., 2]
        frame_rotations = train_on_single_frame(args, frame_index, model, criterion, joint_offsets, 
                                                joint_parents, frame_kpts3d_np)
        video_root_positions.append(frame_kpts3d_np[0])
        video_rotations.append(frame_rotations)
        
        
    save_bvh(args, joint_offsets, video_root_positions, video_rotations, seq='ZYX')


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    