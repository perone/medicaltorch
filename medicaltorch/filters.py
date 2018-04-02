import numpy as np


class SliceFilter(object):
    def __init__(self, filter_empty_mask=True,
                 filter_empty_input=True):
        self.filter_empty_mask = filter_empty_mask
        self.filter_empty_input = filter_empty_input

    def __call__(self, sample):
        input_data, gt_data = sample['input'], sample['gt']
        
        if self.filter_empty_mask:
            if not np.any(gt_data):
                return False
        
        if self.filter_empty_input:
            if not np.any(input_data):
                return False
        
        return True