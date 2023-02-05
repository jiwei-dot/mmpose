# Author: JiWei
# Date: 02/03/2023
# Email: 821957540@qq.com


import cv2
import numpy as np


class AssignHandBBoxForEachPerson:
    """
    为data_list中每一项加上`left_hand_valid`, `left_hand_bbox`     
                         `right_hand_valid`, `right_hand_bbox`
    `left_hand_valid`和`right_hand_valid`为True时代表当前帧检测到手且正确分配
    """
    def __init__(self, kpt_thr = 0.3):
        self.kpt_thr = kpt_thr
    
    def __call__(self, data_list, hand_bboxes):
        # data_list: [dict(bbox, keypoints, area, track_id), ...], data为单张图片的wholebody2d检测结果,
        #       长度对应单张图片检测到的人数
        # hand_bboxes: ndarray [Nx5]
        num_hand_bboxes = len(hand_bboxes)
        if num_hand_bboxes == 0:
            for idx in range(len(data_list)):
                data_list[idx]['left_hand_bbox'] = np.empty((0, 5))
                data_list[idx]['left_hand_valid'] = False
                data_list[idx]['right_hand_bbox'] = np.empty((0, 5))
                data_list[idx]['right_hand_valid'] = False
        else:
            for idx, data in enumerate(data_list):  
                dist_left_arm = np.ones((num_hand_bboxes, )) * float('inf')
                dist_right_arm = np.ones((num_hand_bboxes, )) * float('inf')
                person_keypoints2d = data['keypoints']
                
                # left arm
                if person_keypoints2d[9][-1] > self.kpt_thr and person_keypoints2d[7][-1] > self.kpt_thr:
                    dist_wrist_elbow = np.linalg.norm(person_keypoints2d[9][:2] - person_keypoints2d[7][:2])
                    c_x = (hand_bboxes[:, 0] + hand_bboxes[:, 2]) / 2
                    c_y = (hand_bboxes[:, 1] + hand_bboxes[:, 3]) / 2
                    center = np.stack((c_x, c_y), axis=-1)
                    dist_bbox_ankle = np.linalg.norm(center - person_keypoints2d[9][:2], axis=-1)
                    mask = dist_bbox_ankle < dist_wrist_elbow * 1.5
                    dist_left_arm[mask] = dist_bbox_ankle[mask]

                # right arm
                if person_keypoints2d[10][-1] > self.kpt_thr and person_keypoints2d[8][-1] > self.kpt_thr:
                    dist_wrist_elbow = np.linalg.norm(person_keypoints2d[10][:2] - person_keypoints2d[8][:2])
                    c_x = (hand_bboxes[:, 0] + hand_bboxes[:, 2]) / 2
                    c_y = (hand_bboxes[:, 1] + hand_bboxes[:, 3]) / 2
                    center = np.stack((c_x, c_y), axis=-1)
                    dist_bbox_ankle = np.linalg.norm(center - person_keypoints2d[10][:2], axis=-1)
                    mask = dist_bbox_ankle < dist_wrist_elbow * 1.5
                    dist_right_arm[mask] = dist_bbox_ankle[mask]
                    
                left_id = np.argmin(dist_left_arm)
                right_id = np.argmin(dist_right_arm)
                
                if left_id != right_id:
                    if dist_left_arm[left_id] < float('inf'):
                        data_list[idx]['left_hand_bbox'] = hand_bboxes[left_id]
                        data_list[idx]['left_hand_valid'] = True
                    else:
                        data_list[idx]['left_hand_bbox'] = np.empty((0, 5))
                        data_list[idx]['left_hand_valid'] = False
                        
                    if dist_right_arm[right_id] < float('inf'):
                        data_list[idx]['right_hand_bbox'] = hand_bboxes[right_id]
                        data_list[idx]['right_hand_valid'] = True
                    else:
                        data_list[idx]['right_hand_bbox'] = np.empty((0, 5))
                        data_list[idx]['right_hand_valid'] = False 
                else:
                    assign_hand = None
                    assign_id = None
                    data_list[idx]['left_hand_bbox'] = np.empty((0, 5))
                    data_list[idx]['left_hand_valid'] = False
                    data_list[idx]['right_hand_bbox'] = np.empty((0, 5))
                    data_list[idx]['right_hand_valid'] = False  
                    
                    if dist_left_arm[left_id] < dist_right_arm[right_id]:
                        assign_hand = 'left_hand'
                        assign_id = left_id
                        
                    elif dist_left_arm[left_id] > dist_right_arm[right_id]:
                        assign_hand = 'right_hand'
                        assign_id = right_id
                
                    if assign_hand is not None:
                        data_list[idx][assign_hand+"_bbox"] = hand_bboxes[assign_id]
                        data_list[idx][assign_hand+"_valid"] = True
        return data_list
    
    
class AssignHandKeypoints2dForEachPerson:
    """
    为data_list中每一项加上`left_hand_keypoints`, `right_hand_keypoints`
    """
    def __init__(self):
        pass
    
    def __call__(self, data_list, data_list_last_frame, hand_keypoints2d_list):
        # data_list: [{bbox, keypoints, area, track_id, left_hand_bbox, left_hand_valid, right_hand_bbox, right_hand_valid}, ...]
        # data_list_last_frame: [{}, ...]
        # hand_keypoints2d_list: [{bbox, keypoints}, ...]
        
        tmp_idx = 0
        
        for person in data_list:
            # person: dict(bbox, keypoints, area, track_id, left_hand_bbox, left_hand_valid, right_hand_bbox, right_hand_valid)
            # 下面的代码是在每个person中加入`left_hand_keypoints`和`right_hand_keypoints`
            # left_hand_valid为True时表示当前帧检测到人手
            if person['left_hand_valid']:
                person['left_hand_keypoints'] = hand_keypoints2d_list[tmp_idx]['keypoints']
                tmp_idx += 1
            else:
                tmp_person = None
                for person_last in data_list_last_frame:
                    if person_last['track_id'] == person['track_id']:
                        tmp_person = person_last
                        break
                if tmp_person is None or not tmp_person['left_hand_valid']:
                    # 没检测到hand, 同时上一帧也没有对应的信息或者上一帧没检测到手, 用tcformer结果代替
                    person['left_hand_keypoints'] = person['keypoints'][91: 112]
                    # raise NotImplementedError     
                else:
                    # 利用前一帧修正
                    person['left_hand_keypoints'] = tmp_person['left_hand_keypoints']
                    person['left_hand_bbox'] = tmp_person['left_hand_bbox']
                    # person['left_hand_valid'] = False
                  
            if person['right_hand_valid']:
                person['right_hand_keypoints'] = hand_keypoints2d_list[tmp_idx]['keypoints']
                tmp_idx += 1
            else:
                tmp_person = None
                for person_last in data_list_last_frame:
                    if person_last['track_id'] == person['track_id']:
                        tmp_person = person_last
                        break
                if tmp_person is None or not tmp_person['right_hand_valid']:
                    person['right_hand_keypoints'] = person['keypoints'][112: 133]
                    # raise NotImplementedError
                else:
                    # 利用前一帧修正
                    person['right_hand_keypoints'] = tmp_person['right_hand_keypoints']
                    person['right_hand_bbox'] = tmp_person['right_hand_bbox']
                    # person['right_hand_valid'] = False
            
            person['keypoints'][91: 112] = person['left_hand_keypoints'] 
            person['keypoints'][112: 133] = person['right_hand_keypoints']
            
        return data_list
            

class FilterLargeHandForEachPerson:
    def __init__(self, area_thr = 0.02):
        self.area_thr = area_thr
    
    def __call__(self, data_list):
        # data_list: [dict(bbox, keypoints, left_hand_valid, left_hand_bbox, left_hand_keypoints
        #            right_hand_valid, right_hand_bbox, right_hand_keypoints, area, track_id), ...]
        for data in data_list:
            person_bbox = data['bbox']
            left_hand_bbox = data['left_hand_bbox']
            right_hand_bbox = data['right_hand_bbox']
            person_area = (person_bbox[3] - person_bbox[1]) * (person_bbox[2] - person_bbox[0])
            if data['left_hand_valid'] or len(left_hand_bbox) != 0:
                left_hand_area = (left_hand_bbox[3] - left_hand_bbox[1]) * (left_hand_bbox[2] - left_hand_bbox[0])
            else:
                left_hand_area = (max(data['left_hand_keypoints'][:, 0]) - min(data['left_hand_keypoints'][:, 0])) * \
                                (max(data['left_hand_keypoints'][:, 1]) - min(data['left_hand_keypoints'][:, 1]))
                                    
            if data['right_hand_valid'] or len(right_hand_bbox) != 0:
                right_hand_area = (right_hand_bbox[3] - right_hand_bbox[1]) * (right_hand_bbox[2] - right_hand_bbox[0])
            else:
                right_hand_area = (max(data['right_hand_keypoints'][:, 0]) - min(data['right_hand_keypoints'][:, 0])) * \
                                (max(data['right_hand_keypoints'][:, 1]) - min(data['right_hand_keypoints'][:, 1]))
            print(left_hand_area / person_area, right_hand_area / person_area)
            
            # 检测的手不合理, 用另外一只手代替
            if left_hand_area / person_area > self.area_thr:
                print('left hand too large')
                if (data['right_hand_valid'] or len(right_hand_bbox) != 0) and (right_hand_area / person_area <= self.area_thr):
                    
                    # 1. 先将右手平移到左手
                    data['left_hand_keypoints'][:, :2] = data['right_hand_keypoints'][:, :2] + data['keypoints'][9][:2] - data['keypoints'][10][:2]
                    
                    # 2. 再旋转，使得手臂和中指的夹角双手保存一致
                    c = data['keypoints'][9, :2]
                    
                    # 计算右臂夹角
                    vec1 = data['keypoints'][10, :2] - data['keypoints'][8, :2]
                    vec2 = data['right_hand_keypoints'][0, :2] - data['keypoints'][10, :2]
                    cos_ = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    sin_ = np.cross(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    arctan2_ = np.arctan2(sin_, cos_)
                    theta1 = arctan2_ / np.pi
                    
                    # 计算左臂夹角
                    vec1 = data['keypoints'][9, :2] - data['keypoints'][7, :2]
                    vec2 = data['left_hand_keypoints'][0, :2] - data['keypoints'][9, :2]
                    cos_ = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    sin_ = np.cross(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    arctan2_ = np.arctan2(sin_, cos_)
                    theta2 = arctan2_ / np.pi
                    
                    r = (theta2 + theta1) * 180
                    s = 1
                    M = cv2.getRotationMatrix2D(c, r, s)
                    data['left_hand_keypoints'][:, :2] = np.column_stack([data['left_hand_keypoints'][:, :2], np.ones(len(data['left_hand_keypoints'][:, :2]))]) @ M.T
                    
                    # 3. 根据左手镜像翻转
                    line_pt = data['keypoints'][9, :2]
                    line_vec = data['left_hand_keypoints'][0, :2] - line_pt
                    for i in range(1, len(data['left_hand_keypoints'])):
                        pt_vec = data['left_hand_keypoints'][i, :2] - line_pt
                        proj_vec = np.dot(pt_vec, line_vec) / np.dot(line_vec, line_vec) * line_vec
                        pt_sym = line_pt + 2 * proj_vec - pt_vec
                        data['left_hand_keypoints'][i, :2] = pt_sym
                    
                    # lazy way 
                    data['left_hand_bbox'] = np.empty((0, 5))
                    data['left_hand_valid'] = False
                    
                else:
                    print("Couldn't correct hand kps, will fix in future")
                    data['left_hand_bbox'] = np.empty((0, 5))
                    data['left_hand_valid'] = False
                    # raise NotImplementedError

            if right_hand_area / person_area > self.area_thr:
                print('right hand too large')
                if (data['left_hand_valid'] or len(left_hand_bbox) != 0) and (left_hand_area / person_area <= self.area_thr):
                    
                    # 1. 先将左手平移到右手
                    data['right_hand_keypoints'][:, :2] = data['left_hand_keypoints'][:, :2] + data['keypoints'][10][:2] - data['keypoints'][9][:2]
                    
                    # 2. 再旋转，使得手臂和中指的夹角双手保存一致
                    c = data['keypoints'][10, :2]
                    
                    # 计算左臂夹角
                    vec1 = data['keypoints'][9, :2] - data['keypoints'][7, :2]
                    vec2 = data['left_hand_keypoints'][0, :2] - data['keypoints'][9, :2]
                    cos_ = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    sin_ = np.cross(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    arctan2_ = np.arctan2(sin_, cos_)
                    theta1 = arctan2_ / np.pi
                    
                    # 计算右臂夹角
                    vec1 = data['keypoints'][10, :2] - data['keypoints'][8, :2]
                    vec2 = data['right_hand_keypoints'][0, :2] - data['keypoints'][10, :2]
                    cos_ = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    sin_ = np.cross(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    arctan2_ = np.arctan2(sin_, cos_)
                    theta2 = arctan2_ / np.pi
                    
                    r = (theta2 + theta1) * 180
                    s = 1
                    M = cv2.getRotationMatrix2D(c, r, s)
                    data['right_hand_keypoints'][:, :2] = np.column_stack([data['right_hand_keypoints'][:, :2], np.ones(len(data['right_hand_keypoints'][:, :2]))]) @ M.T
                    
                    # 3. 根据右手进行镜像翻转
                    line_pt = data['keypoints'][10, :2]
                    line_vec = data['right_hand_keypoints'][0, :2] - line_pt
                    for i in range(1, len(data['right_hand_keypoints'])):
                        pt_vec = data['right_hand_keypoints'][i, :2] - line_pt
                        proj_vec = np.dot(pt_vec, line_vec) / np.dot(line_vec, line_vec) * line_vec
                        pt_sym = line_pt + 2 * proj_vec - pt_vec
                        data['right_hand_keypoints'][i, :2] = pt_sym
                        
                    # lazy way
                    data['right_hand_bbox'] = np.empty((0, 5)) 
                    data['right_hand_valid'] = False
                else:
                    print("Couldn't correct hand kps, will fix in future")
                    data['right_hand_bbox'] = np.empty((0, 5))
                    data['right_hand_valid'] = False
                    # raise NotImplementedError
            
            data['keypoints'][91: 112] = data['left_hand_keypoints'] 
            data['keypoints'][112: 133] = data['right_hand_keypoints']
            
        return data_list