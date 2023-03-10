import numpy as np

from mmpose.datasets.builder import PIPELINES


@PIPELINES.register_module()
class CustomGetLocalAndGlobalOffsets:
    
    def __init__(self, parents):
        self.parents = parents
        
    def __call__(self, results):
        # [22, 3]
        target = results['target']
        num_keypoints = len(target)
        assert self.parents[0]== -1
        assert len(self.parents) == num_keypoints

        global_offsets = target - target[0: 1]
        global_offsets = np.delete(global_offsets, 0, axis=0)
        
        local_offsets = np.zeros_like(target)
        for i in range(num_keypoints):
            if self.parents[i] == -1:
                continue
            local_offsets[i] = target[i] - target[self.parents[i]]
        local_offsets = np.delete(local_offsets, 0, axis=0)
        
        # [42, 3]
        total_offsets = np.concatenate([global_offsets, local_offsets], axis=0)
        
        # results['global_offsets'] = global_offsets
        # results['local_offsets'] = local_offsets
        results['target'] = total_offsets
        results['parents'] = self.parents
        results['root_position'] = target[0:1]
        
        return results
