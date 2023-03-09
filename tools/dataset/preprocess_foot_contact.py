import os
import os.path as osp
import json
import numpy as np


data_root = '/data/jiwei/Datasets/FootContact/images/'
save_root = '/data/jiwei/Datasets/FootContact/annotations/'


characters = os.listdir(data_root)
openpose_model = {
    8: 'MidHip',
    9: 'RHip',
    10: 'RKnee',
    11: 'RAnkle',
    12: 'LHip',
    13: 'LKnee',
    14: 'LAnkle',
    19: 'LBigToe',
    20: 'LSmallToe',
    21: 'LHeel',
    22: 'RBigToe',
    23: 'RSmallToe',
    24: 'RHeel'
}
lower_body_indices = list(openpose_model.keys())


def json2kpst2d(json_path):

    with open(json_path, 'r') as fin:
        data = json.load(fin)
        
    # [25, 3], we don't need all the joints, only the lower body joints
    valid = True
    try:
        kpts2d = np.array(data['people'][0]['pose_keypoints_2d']).reshape(-1, 3)
        kpts2d = kpts2d[lower_body_indices]
    except:
        valid = False
        kpts2d = np.zeros((len(lower_body_indices), 3))
        
    return valid, kpts2d
    

# [N， ]
_imgnames = []
# [N, J, 3]
_kpts2d = []
# [N, 4]
_contacts = []

for character in characters:
    path_character = osp.join(data_root, character)
    actions = os.listdir(path_character)
    for action in actions:
        path_action = osp.join(data_root, character, action)
        foot_contacts_npy = osp.join(path_action, 'foot_contacts.npy')
        keypoints_view0 = osp.join(path_action, 'keypoints_view0')
        keypoints_view1 = osp.join(path_action, 'keypoints_view1')
        
        assert osp.exists(foot_contacts_npy)
        assert osp.exists(keypoints_view0)
        assert osp.exists(keypoints_view1)
        
        foot_contacts = np.load(foot_contacts_npy)
        assert len(foot_contacts) == len(os.listdir(keypoints_view0)) == len(os.listdir(keypoints_view1))
        
        # process view0
        view0 = []
        view0_valid = []
        view0_imgname = []
        for json_file in sorted(os.listdir(keypoints_view0)):
            imgname = osp.join(character, action, 'view0', json_file.replace('_keypoints.json', '.png'))
            assert osp.exists(osp.join(data_root, imgname))
            valid, kpts2d = json2kpst2d(osp.join(keypoints_view0, json_file))
            view0.append(kpts2d)
            view0_valid.append(valid)
            view0_imgname.append(imgname)
            
        view0 = np.array(view0)
        view0_valid = np.array(view0_valid)
        view0_imgname = np.array(view0_imgname)
        
        _imgnames.append(view0_imgname[view0_valid])
        _kpts2d.append(view0[view0_valid])
        _contacts.append(foot_contacts[view0_valid])
            
            
        # process view1
        view1 = []
        view1_valid = []
        view1_imgname = []
        for json_file in sorted(os.listdir(keypoints_view1)):
            imgname = osp.join(character, action, 'view1', json_file.replace('_keypoints.json', '.png'))
            assert osp.exists(osp.join(data_root, imgname))
            valid, kpts2d = json2kpst2d(osp.join(keypoints_view1, json_file))
            view1.append(kpts2d)
            view1_valid.append(valid)
            view1_imgname.append(imgname)
            
        view1 = np.array(view1)
        view1_valid = np.array(view1_valid)
        view1_imgname = np.array(view1_imgname)
        
        _imgnames.append(view1_imgname[view1_valid])
        _kpts2d.append(view1[view1_valid])
        _contacts.append(foot_contacts[view1_valid])
        
        
# print(np.concatenate(_kpts2d).shape)
# print(np.concatenate(_contacts).shape)
# print(np.concatenate(_imgnames).shape)
# print(np.concatenate(_imgnames))


_imgnames = np.concatenate(_imgnames)
_kpts2d = np.concatenate(_kpts2d)
_contacts = np.concatenate(_contacts)


train_val_split = int(len(_imgnames) * 0.8)


np.savez(
    f'{save_root}/foot_contact_train.npz',
    imgnames=_imgnames[:train_val_split],
    kpts2d=_kpts2d[:train_val_split],
    contacts=_contacts[:train_val_split]
)


np.savez(
    f'{save_root}/foot_contact_test.npz',
    imgnames=_imgnames[train_val_split:],
    kpts2d=_kpts2d[train_val_split:],
    contacts=_contacts[train_val_split:]
)


