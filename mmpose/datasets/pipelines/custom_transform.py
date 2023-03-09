import numpy as np

from ..builder import PIPELINES


@PIPELINES.register_module()
class CustomNormalizeCoordinates:
    def __init__(self, pixel_length):
        self.pixel_length = pixel_length
        
        
    def __call__(self, results):
        input_2d = results['input_2d']
        input_2d[:, :, :2] /= self.pixel_length
        results['input_2d'] = input_2d
        return results
    
    
@PIPELINES.register_module()
class CustomRelativeToCenterRoot:
    def __init__(self, root_index):
        self.root_index = root_index
        
    def __call__(self, results):
        # [T, J, 3]
        input_2d = np.array(results['input_2d'], dtype=np.float32)
        windows_size = len(input_2d)
        windows_center = windows_size // 2
        
        root_xy = input_2d[windows_center, self.root_index, :2]
        input_2d[:, :, :2]  -= root_xy[None, None, :2]
        input_2d[windows_center, self.root_index, :2] = root_xy
        
        results['input_2d'] = input_2d
        return results
        
