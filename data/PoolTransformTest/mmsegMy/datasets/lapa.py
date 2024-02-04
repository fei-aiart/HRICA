from .builder import DATASETS
from .custom import CustomDataset
import torch


@DATASETS.register_module()
class LaPaDataset(CustomDataset):
    """LaPa dataset.

    In segmentation map annotation for LaPa, Train-IDs of the 10k version
    are from 1 to 11, where 0 is the ignore index, and Train-ID of COCO Stuff
    164k is from 0 to 170, where 255 is the ignore index. So, they are all 171
    semantic categories. ``reduce_zero_label`` is set to True and False for the
    10k and 164k versions, respectively. The ``img_suffix`` is fixed to '.jpg',
    and ``seg_map_suffix`` is fixed to '.png'.
    """
    CLASSES = ('skin', 'left eyebrow', 'right eyebrow', 'left eye',
               'right eye', 'nose', 'upper lip', 'inner mouth', 'lower lip', 'hair', 'background')

    PALETTE = [[0, 153, 255], [102, 255, 153], [0, 204, 153],
               [255, 255, 102], [255, 255, 204], [255, 153, 0], [255, 102, 255],
               [102, 0, 51], [255, 204, 255], [255, 0, 102], [0, 0, 0]]

    def __init__(self, **kwargs):
        super(LaPaDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', **kwargs)

    def prepare_train_img(self, idx):
        """Get training data and annotations after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training data and annotation after pipeline with new keys
                introduced by pipeline.
        """

        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)
        self.pre_pipeline(results)
        res = self.pipeline(results)
        a = res['gt_semantic_seg'].data
        res['gt_semantic_seg'].data[a == 255] = 10
        # print(res['gt_semantic_seg'].data[a == 9])
        return res