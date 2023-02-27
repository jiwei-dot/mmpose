import json
import cv2
import os
import numpy as np


def kpts2bbox(kpts, scale_factor=1.2):
    x0 = np.min(kpts[:, 0])
    y0 = np.min(kpts[:, 1])
    x1 = np.max(kpts[:, 0])
    y1 = np.max(kpts[:, 1])
    w = x1 - x0
    h = y1 - y0
    new_w = w * scale_factor
    new_h = h * scale_factor
    new_x0 = max(0, x0 - new_w / 2)
    new_y0 = max(0, y0 - new_h / 2)
    return np.array([new_x0, new_y0, new_w, new_h])


def bbox2area(bbox, format='xywh'):
    assert format in ('xyxy', 'xywh')
    if format == 'xywh':
        return bbox[2] * bbox[3]
    else:
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])


coco_path = "data/coco/annotations/coco_wholebody_val_v1.0.json"
h3wb_path = "data/h36m/annotation_wholebody3d/h3wb_test.npz"


with open(coco_path, 'r') as fin:
    coco_data = json.load(fin)
    
    
coco_images = coco_data['images']
coco_annotations = coco_data['annotations']
coco_categories = coco_data['categories']
# print(coco_categories)


max_coco_image_id = -1
for image in coco_images:
    del image['license']
    del image['coco_url']
    del image['date_captured']
    del image['flickr_url']
    image['file_name'] = 'coco/' + image['file_name']
    # print(image['file_name'])
    # exit()
    if image['id'] > max_coco_image_id:
        max_coco_image_id = image['id']
    # print(image)
    # exit()
print("max_coco_image_id", max_coco_image_id)
        

new_coco_annotations = []
coco_anno_id = 0
for anno in coco_annotations:
    keypoints = np.array(anno['keypoints']).reshape(-1, 3)
    if anno['lefthand_valid']:
        lefthand_kpts2d = np.array(anno['lefthand_kpts']).reshape(-1, 3)
        kpts2d = lefthand_kpts2d
        kpts3d = None
        bbox = anno['lefthand_box']
        new_coco_annotations.append({
            'area': bbox2area(bbox),
            'iscrowd': 0,
            'image_id': anno['image_id'],
            'bbox': bbox,
            'category_id': 1,
            'id': coco_anno_id,
            'keypoints_2d': kpts2d.flatten().tolist(),
            'keypoints_3d': None
        })
        coco_anno_id += 1
    
    if anno['righthand_valid']:
        righthand_kpts2d = np.array(anno['righthand_kpts']).reshape(-1, 3)
        kpts2d = righthand_kpts2d
        kpts3d = None
        bbox = anno['righthand_box']
        new_coco_annotations.append({
            'area': bbox2area(bbox),
            'iscrowd': 0,
            'image_id': anno['image_id'],
            'bbox': bbox,
            'category_id': 1,
            'id': coco_anno_id,
            'keypoints_2d': kpts2d.flatten().tolist(),
            'keypoints_3d': None
        })
        coco_anno_id += 1
        
        
h3wb_image_id = max_coco_image_id + 1
h3wb_anno_id = coco_anno_id


h3wb_data = np.load(h3wb_path)
h3wb_dataroot = "data/h36m/images/"
h3wb_images = []
h3wb_annotations = []
h3wb_imgname = h3wb_data['imgname']
h3wb_kpts2d = h3wb_data['part']
h3wb_kpts3d = h3wb_data['S']
lefthand_indices = list(range(91, 112))
righthand_indices = list(range(112, 133))



for idx in range(len(h3wb_imgname)):
    file_name = h3wb_imgname[idx]
    tmp_img = cv2.imread(os.path.join(h3wb_dataroot, file_name))
    h, w, _ = tmp_img.shape
    del tmp_img
    h3wb_images.append({
        'file_name': 'h3wb/'+file_name,
        'height': h,
        'width': w,
        'id': h3wb_image_id,
    })
    
    # lefthand 
    lefthand_kpts2d = h3wb_kpts2d[idx][lefthand_indices]
    lefthand_kpts3d = h3wb_kpts3d[idx][lefthand_indices]
    lefthand_bbox = kpts2bbox(lefthand_kpts2d)
    h3wb_annotations.append({
        'area': bbox2area(lefthand_bbox),
        'iscrowd': 0,
        'image_id': h3wb_image_id,
        'bbox': lefthand_bbox.tolist(),
        'category_id': 1,
        'id': h3wb_anno_id,
        'keypoints_2d': lefthand_kpts2d.flatten().tolist(),
        'keypoints_3d': lefthand_kpts3d.flatten().tolist()
    })
    h3wb_anno_id += 1
    
    # righthand
    righthand_kpts2d = h3wb_kpts2d[idx][righthand_indices]
    righthand_kpts3d = h3wb_kpts3d[idx][righthand_indices]
    righthand_bbox = kpts2bbox(righthand_kpts2d)
    h3wb_annotations.append({
        'area': bbox2area(righthand_bbox),
        'iscrowd': 0,
        'image_id': h3wb_image_id,
        'bbox': righthand_bbox.tolist(),
        'category_id': 1,
        'id': h3wb_anno_id,
        'keypoints_2d': righthand_kpts2d.flatten().tolist(),
        'keypoints_3d': righthand_kpts3d.flatten().tolist()
    })
    h3wb_anno_id += 1
    
    h3wb_image_id += 1
    


coco_images.extend(h3wb_images)
new_coco_annotations.extend(h3wb_annotations)
categories = [{
                "supercategory": "hand", 
                "id": 1, 
                "name": "hand", 
                "keypoints": ["wrist", "thumb1", "thumb2", "thumb3", "thumb4", "forefinger1", "forefinger2", "forefinger3", "forefinger4", "middle_finger1", "middle_finger2", "middle_finger3", "middle_finger4", "ring_finger1", "ring_finger2", "ring_finger3", "ring_finger4", "pinky_finger1", "pinky_finger2", "pinky_finger3", "pinky_finger4"], 
                "skeleton": [[1, 2], 
                             [2, 3], 
                             [3, 4], 
                             [4, 5], 
                             [1, 6], 
                             [6, 7], 
                             [7, 8], 
                             [8, 9], 
                             [1, 10], 
                             [10, 11], 
                             [11, 12], 
                             [12, 13], 
                             [1, 14], 
                             [14, 15], 
                             [15, 16], 
                             [16, 17], 
                             [1, 18], 
                             [18, 19], 
                             [19, 20], 
                             [20, 21]]}]
        
        
with open("merge_cocohand_and_h3wbhand_val_v1.0.json", 'w') as fout:
    json.dump({
        "images": coco_images,
        "annotations": new_coco_annotations,
        "categories": categories
    }, fout)
    
        
    
    
    

