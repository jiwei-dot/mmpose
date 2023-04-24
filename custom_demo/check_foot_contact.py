from argparse import ArgumentParser
import pickle
from collections import defaultdict
import numpy as np
import copy
import cv2
import os

import torch
from torch import nn
TORCH_VER = torch.__version__

import mmcv


def init_model(args):
    
    class OpenPoseModel(nn.Module):
        def __init__(self, window_size, joints, pred_size, feat_size):
            super(OpenPoseModel, self).__init__()
            self.window_size = window_size
            self.contact_size = pred_size
            self.feat_size = feat_size
            #
            # Losses
            #
            self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
            self.sigmoid = nn.Sigmoid()

            #
            # Create the model
            #
            self.model = nn.Sequential(
                            nn.Linear(window_size*joints*self.feat_size, 1024),
                            nn.BatchNorm1d(1024),
                            nn.ReLU(),
                            nn.Linear(1024, 512),
                            nn.BatchNorm1d(512),
                            nn.ReLU(),
                            nn.Linear(512, 128),
                            nn.BatchNorm1d(128),
                            nn.ReLU(),
                            nn.Dropout(p=0.3),
                            nn.Linear(128, 32),
                            nn.BatchNorm1d(32),
                            nn.ReLU(),
                            nn.Linear(32, 4*pred_size)
                        )
            # initialize weights
            self.model.apply(self.init_weights)

        def init_weights(self, m):
            if type(m) == nn.Linear:
                torch.nn.init.xavier_uniform_(m.weight)
                m.bias.data.fill_(0.01)

        def forward(self, x):
            # data_batch is B x N x J x 3(or 2 if no confidence)
            B, N, J, F = x.size()
            # flatten to single data vector
            x = x.view(B, N*J*F)
            # run model
            x = self.model(x)
            return x.view(B, self.contact_size, 4)

        def loss(self, outputs, labels):
            ''' Returns the loss value for the given network output '''
            B, N, _ = outputs.size()
            outputs = outputs.view(B, N*4)

            B, N, _ = labels.size()
            labels = labels.view(B, N*4)

            loss_flat = self.bce_loss(outputs, labels)
            loss = loss_flat.view(B, N, 4)

            return loss

        def prediction(self, outputs, thresh=0.5):
            probs = self.sigmoid(outputs)
            pred = probs > thresh
            return pred, probs

        def accuracy(self, outputs, labels, thresh=0.5, tgt_frame=None):
            ''' Calculates confusion matrix counts for TARGET (middle) FRAME ONLY'''
            # threshold to classify
            pred, _ = self.prediction(outputs, thresh)

            if tgt_frame is None:
                tgt_frame = self.contact_size // 2

            # only want to evaluate accuracy of middle frame
            pred = pred[:, tgt_frame, :]
            if TORCH_VER == '1.0.0' or TORCH_VER == '1.1.0':
                pred = pred.byte()
            else:
                # 1.2.0
                pred = pred.to(torch.bool)
            labels = labels[:, tgt_frame, :]
            if TORCH_VER == '1.0.0' or TORCH_VER == '1.1.0':
                labels = labels.byte()
            else:
                labels = labels.to(torch.bool)

            # counts for confusion matrix
            # true positive (pred contact, labeled contact)
            true_pos = pred & labels
            true_pos_cnt = torch.sum(true_pos).to('cpu').item()
            # false positive (pred contact, not lebeled contact)
            false_pos = pred & ~(labels)
            false_pos_cnt = torch.sum(false_pos).to('cpu').item()
            # false negative (pred no contact, labeled contact)
            false_neg = ~(pred) & labels
            false_neg_cnt = torch.sum(false_neg).to('cpu').item()
            # true negative (pred no contact, no labeled contact)
            true_neg = (~pred) & (~labels)
            true_neg_cnt = torch.sum(true_neg).to('cpu').item()

            return true_pos_cnt, false_pos_cnt, false_neg_cnt, true_neg_cnt
    
    model = OpenPoseModel(9, 13, 5, 3)
    cpt = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(cpt, strict=True)
    model.to(args.device)
    model.eval()
    return model
    

def get_normalize_info(video_kpts2d_list):
    person_dist_list = defaultdict(list)
    for single_frame_dict in video_kpts2d_list:
        for track_id, person_dict in single_frame_dict.items():
            wholebody_keypoints = person_dict['wholebody_keypoints']
            left_toe = wholebody_keypoints[17, :2]
            mid_hip = (wholebody_keypoints[11, :2] + wholebody_keypoints[12, :2]) / 2
            person_dist_list[track_id].append(np.linalg.norm(left_toe - mid_hip))
    
    person_dist = dict()
    for track_id in person_dist_list:
        person_dist[track_id] = np.median(person_dist_list[track_id])
    return person_dist


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
                

def extract_target_trackid(video_kpts2d_list, track_id=0):
    single_person_list = [None] * len(video_kpts2d_list)
    for frame_idx, single_frame_dict in enumerate(video_kpts2d_list):
        if track_id in single_frame_dict:
            single_person_list[frame_idx] = single_frame_dict[track_id]
    # 处理未检测到track_id的帧
    single_person_list = fill_list(single_person_list)
    return single_person_list


def get_max_track_id(video_kpts2d_list):
    max_track_id = -1
    for single_frame_dict in video_kpts2d_list:
        if len(single_frame_dict) != 0:
            max_track_id = max(max_track_id, max(single_frame_dict.keys()))
    if max_track_id == -1:
        raise ValueError
    return max_track_id


def build_batch_data(data_list, med_dist, window_size=9):
    
    # OP_LOWER_JOINTS_MAP = { "MidHip"    : 0,
    #                     "RHip"      : 1,
    #                     "RKnee"     : 2,
    #                     "RAnkle"    : 3,
    #                     "LHip"      : 4,
    #                     "LKnee"     : 5,
    #                     "LAnkle"    : 6,
    #                     "LBigToe"   : 7,
    #                     "LSmallToe" : 8,
    #                     "LHeel"     : 9,
    #                     "RBigToe"   : 10,
    #                     "RSmallToe" : 11,
    #                     "RHeel"     : 12  }
    
    lower_indices = [12, 14, 16, 11, 13, 15, 17, 18, 19, 20, 21, 22]
    batched_data = np.zeros((len(data_list), window_size, len(lower_indices) + 1, 3), dtype=np.float32)
    pad_size = window_size // 2
    data_list = [data_list[0]] * pad_size + data_list + [data_list[-1]] * pad_size
    start_idx = window_size // 2
    end_idx = len(data_list) - start_idx
    
    rel_tgt_idx = window_size // 2
    root_idx = 0
    for i in range(start_idx, end_idx):
        w_start = i - window_size // 2
        w_end = i + window_size // 2
        for j in range(w_start, w_end + 1):
            wholebody_keypoints = data_list[j]['wholebody_keypoints']
            lower_keypoints = wholebody_keypoints[lower_indices]
            mid_hip = (lower_keypoints[0:1] + lower_keypoints[3:4]) / 2
            lower_keypoints = np.concatenate([mid_hip, lower_keypoints])
            lower_keypoints[:, :2] /= med_dist         
            batched_data[i-start_idx][j-w_start] = lower_keypoints
        tgt_root = batched_data[i-start_idx][rel_tgt_idx, root_idx, :2].copy()
        tgt_root = tgt_root.reshape((1, 1, 2))
        batched_data[i-start_idx][:, :, :2] -= tgt_root
        batched_data[i-start_idx][rel_tgt_idx, root_idx, :2] = tgt_root
    return batched_data


def get_parser():
    parser = ArgumentParser()
    parser.add_argument('--pkl-file', required=True, help='video 2d kpts pkl file')
    parser.add_argument('--out-root', required=True)
    parser.add_argument('--checkpoint', default='workspace/checkpoints/contact_detection_weights.pth')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--show-result', action='store_true')
    parser.add_argument('--video-path', default=None)
    return parser


def main(args):
    
    with open(args.pkl_file, 'rb') as fin:
        video_kpts2d_list = pickle.load(fin)
    
    med_dist_dict = get_normalize_info(video_kpts2d_list)
    max_track_id = get_max_track_id(video_kpts2d_list)
    model = init_model(args)
    
    foot_contact_results = dict()
    for track_id in range(max_track_id + 1):
        track_id_video_list = extract_target_trackid(video_kpts2d_list, track_id)
        batched_data = build_batch_data(track_id_video_list, med_dist_dict[track_id])
        batched_data_torch = torch.from_numpy(batched_data).to(args.device)
        with torch.no_grad():
            output = model(batched_data_torch)
            pred, _ = model.prediction(output)
        pred = pred.cpu().numpy()
        foot_contact_results[track_id] = pred

        name = (os.path.basename(args.video_path)).split('.')[0]
        save_filename = f'footcontact_{name}.pkl'
        with open(os.path.join(args.out_root,save_filename), 'wb') as fout:
            pickle.dump(foot_contact_results, fout)
            

    if args.show_result:
        assert args.video_path is not None
        video = mmcv.VideoReader(args.video_path)
        assert len(video) == len(video_kpts2d_list)
        # name = (os.path.basename(args.video_path)).split('.')[0]
        save_filename = f'footcontact_{name}.mp4'
        writer = cv2.VideoWriter(
            os.path.join(args.out_root, save_filename),
            cv2.VideoWriter_fourcc(*'mp4v'),
            video.fps,
            video.resolution)
        left_heel_idx, left_toe_idx, right_heel_idx, right_toe_idx = 19, 17, 22, 20
        for frame_id, cur_frame in enumerate(mmcv.track_iter_progress(video)):
            single_frame_dict = video_kpts2d_list[frame_id]
            for track_id, person_dict in single_frame_dict.items():
                wholebody_keypoints = person_dict['wholebody_keypoints']
                left_heel = wholebody_keypoints[left_heel_idx, :2]
                left_toe = wholebody_keypoints[left_toe_idx, :2]
                right_heel = wholebody_keypoints[right_heel_idx, :2]
                right_toe = wholebody_keypoints[right_toe_idx, :2]
                
                # res: [left_heel, left_toes, right_heel, right_toes]
                res = foot_contact_results[track_id][frame_id][2]
                
                left_heel_color = (0, 255, 0) if res[0] else (0, 0, 255)
                left_toe_color = (0, 255, 0) if res[1] else (0, 0, 255)
                right_heel_color = (0, 255, 0) if res[2] else (0, 0, 255)
                right_toe_color = (0, 255, 0) if res[3] else (0, 0, 255)
                
                cv2.circle(cur_frame, (int(left_heel[0]), int(left_heel[1])), 4, left_heel_color, -1)
                cv2.circle(cur_frame, (int(left_toe[0]), int(left_toe[1])), 4, left_toe_color, -1)
                cv2.circle(cur_frame, (int(right_heel[0]), int(right_heel[1])), 4, right_heel_color, -1)
                cv2.circle(cur_frame, (int(right_toe[0]), int(right_toe[1])), 4, right_toe_color, -1)
                
                # cv2.imwrite(f'{frame_id}.jpg', cur_frame)
                writer.write(cur_frame)
        writer.release()
            

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)