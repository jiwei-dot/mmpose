from imagecorruptions import corrupt
import numpy as np
from ..builder import PIPELINES


@PIPELINES.register_module()
class Corrupt:
    """Corruption augmentation.
    Corruption transforms implemented based on
    `imagecorruptions <https://github.com/bethgelab/imagecorruptions>`_.
    Args:
        corruption (str): Corruption name.
        severity (int, optional): The severity of corruption. Default: 1.
    """

    def __init__(self, corruption, severity=1, prob: float = 0.3):
        self.corruption = corruption
        self.severity = severity
        self.prob = prob

    def __call__(self, results):
        """Call function to corrupt image.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Result dict with images corrupted.
        """
        if  np.random.rand() < self.prob:
            if 'img_fields' in results:
                assert results['img_fields'] == ['img'], \
                    'Only single img_fields is allowed'
            results['img'] = corrupt(
                results['img'].astype(np.uint8),
                corruption_name=self.corruption,
                severity=self.severity)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(corruption={self.corruption}, '
        repr_str += f'severity={self.severity})'
        return repr_str