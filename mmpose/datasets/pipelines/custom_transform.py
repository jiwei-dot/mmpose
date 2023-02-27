import numpy as np

from mmpose.datasets.builder import PIPELINES


@PIPELINES.register_module()
class CustomGenerateJoints:
    """
    Generate joints_3d, joints_3d_visible, joints_4d, joints_4d_visible.
    """
    def __init__(self):
        pass
    
    def __call__(self, results):
        
        target_2d = results['target_2d']                    # N x 2
        target_2d_visible = results['target_2d_visible']    # N x 1
        target_3d = results['target']                       # N x 3
        target_3d_visible = results['target_visible']       # N x 1
        
        num_joints = len(target_2d)
        
        joints_3d = np.zeros((num_joints, 3), dtype=np.float32)
        joints_3d_visible = np.zeros((num_joints, 3), dtype=np.float32)
        joints_3d[:, :2] = target_2d[:, :2]
        joints_3d_visible[:, :2] = np.minimum(1, target_2d_visible)
        
        joints_4d = np.zeros((num_joints, 4), dtype=np.float32)
        joints_4d_visible = np.zeros((num_joints, 4), dtype=np.float32)
        joints_4d[:, :3] = target_3d[:, :3]
        joints_4d_visible[:, :3] = np.minimum(1, target_3d_visible)
        
        results['joints_3d'] = joints_3d
        results['joints_3d_visible'] = joints_3d_visible
        results['joints_4d'] = joints_4d
        results['joints_4d_visible'] = joints_4d_visible
        
        return results


@PIPELINES.register_module()
class CustomIntegralGenerateTarget:
    
    def __init__(self):
        pass
    
    def __call__(self, results):
        
        W, H = results['ann_info']['image_size']
        
        joints_3d = results['joints_3d']
        xy_pixel_space = joints_3d[:, :2]                           # 17 x 2
        xy_pixel_space /= np.array([W, H])
        
        joints_4d = results['joints_4d']
        z_camera_space = joints_4d[:, 2:3]                          # 17 x 1
        z_camera_space[1:, 0] -= z_camera_space[0:1, 0]
        z_camera_space[0, 0] /= 10.0                                # 2.0472015287279803 ~ 8.015182676948639
        z_camera_space[1:, 0] = (z_camera_space[1:, 0] + 1.0) / 2   # -1.0052666933792245 ~ 0.9807961176710647
        
        target_3d = np.column_stack([xy_pixel_space, z_camera_space])
        
        results['target_3d'] = target_3d                            # 17 x 3
        results['target_3d_weight'] = results['target_weight']      # 17 x 1
        
        return results
   
   

@PIPELINES.register_module()
class CustomGenerateDepth:
    
    def __init__(self, root_index=0):
        self.root_index = root_index    
    
    def __call__(self, results):
        
        joints_4d = results['joints_4d']                    # N x 4
        joints_4d_visible = results['joints_4d_visible']    # N x 4
        
        joints_4d_depth = joints_4d[:, 2]                 # N
        abs_depth = joints_4d_depth[self.root_index]
        joints_4d_rel_depth = joints_4d_depth - abs_depth
        
        joints_4d_depth_visible = joints_4d_visible[:, 2]   # N
        cfg = results['ann_info']
        joint_weights = cfg['joint_weights']
        use_different_joint_weights = cfg['use_different_joint_weights']
        
        depth_weight = joints_4d_depth_visible
        if use_different_joint_weights:
            depth_weight = np.multiply(depth_weight, joint_weights)   

        results['abs_depth'] = abs_depth
        results['target_z'] = joints_4d_rel_depth[1:]
        results['target_z_weight'] = depth_weight[1:]   
        
        # print('abs_depth', abs_depth)  
        # print('joints_4d_rel_depth', joints_4d_rel_depth)
        # print('depth_weight', depth_weight)
        # exit()
        
        return results
    
    
@PIPELINES.register_module()
class CustomPrintResults:
    def __init__(self):
        pass
    
    def __call__(self, results):
        import pprint
        pprint.pprint(results)
        print(results.keys())
        print(results['target'].shape, results['target_weight'].shape)
        exit()
        
   
@PIPELINES.register_module()     
class CustomShowImgAndKps:
    def __init__(self):
        pass
    
    def __call__(self, results):
        joints_3d = results['joints_3d']
        joints_4d = results['joints_4d']
        camera_param = results['camera_param']
        from mmpose.core.camera import SimpleCamera
        camera = SimpleCamera(camera_param)
        kps2d = camera.camera_to_pixel(joints_4d[:, :3])
        import cv2
        image = results['img'].copy()
        for x, y in joints_3d[:, :2]:
            cv2.circle(image, (int(x), int(y)), 2, (0, 255, 0), -1)
        for x, y in kps2d:
            cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)
        cv2.imwrite('tmp.jpg', image)
        exit()
        return results
        
     
   
@PIPELINES.register_module()    
class CustomGenerateXYZInCameraSpace:
    def __init__(self):
        pass
    
    def __call__(self, results):
        joints_4d = results['joints_4d']
        xyz_camera_space = joints_4d[:, :3]
        results['xyz_camera_space'] = xyz_camera_space
        results['xyz_weight'] = results['target_weight']
        return results
    
