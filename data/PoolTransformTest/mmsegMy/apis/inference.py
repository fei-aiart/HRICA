# Copyright (c) OpenMMLab. All rights reserved.
import matplotlib.pyplot as plt
import mmcv
import torch
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
import kornia
import sys
import os
import numpy as np



sys.path.append("/mnt/home/zhujingjie/projects/dailingna/dln_project/pix2pixHDSpade2/data/PoolTransformTest/")

from mmsegMy.datasets.pipelines import Compose

from mmsegMy.models import build_segmentor


def init_segmentor(config, checkpoint=None, device='cuda:0'):
    """Initialize a segmentor from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str, optional) CPU/CUDA device option. Default 'cuda:0'.
            Use 'cpu' for loading model on CPU.
    Returns:
        nn.Module: The constructed segmentor.
    """
    
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        'but got {}'.format(type(config)))
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_segmentor(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint, map_location='cpu')

        torch.save(checkpoint, 'iter_40000_low.pth.tar', _use_new_zipfile_serialization=False)

        # checkpoint = load_checkpoint(model, checkpoint, map_location={'cuda:1':'cuda:0'})
        model.CLASSES = checkpoint['meta']['CLASSES']
        model.PALETTE = checkpoint['meta']['PALETTE']
    model.cfg = config  # save the config in the model for convenience
    # model.to(device)
    model.cuda()
    model.eval()
    return model


class LoadImage:
    """A simple pipeline to load image."""
    def __init__(self,istrain=False):
        self.istrain = istrain


    def __call__(self, results ):
        """Call function to load images into results.

        Args:
            results (dict): A result dict contains the file name
                of the image to be read.

        Returns:
            dict: ``results`` will be returned containing loaded image.
        """
        # print(results['img'])
        if self.istrain==False:
            if isinstance(results['img'], str):
                # print('1')
                results['filename'] = results['img']
                results['ori_filename'] = results['img']
            else:
                results['filename'] = None
                results['ori_filename'] = None
            img = mmcv.imread(results['img'])
            results['img'] = img
            results['img_shape'] = img.shape
            results['ori_shape'] = img.shape

        else:
            results['img']=results
        
        return results


def inference_segmentor(model, img, istrain=False):
    """Inference image(s) with the segmentor.

    Args:
        model (nn.Module): The loaded segmentor.
        imgs (str/ndarray or list[str/ndarray]): Either image files or loaded  images.

    Returns:
        (list[Tensor]): The segmentation result.
    """
    cfg = model.cfg
    if istrain==False:
        # print('进到此处')
        device = next(model.parameters()).device  # model device
        # print('******',device)
        # build the data pipeline
        test_pipeline = [ LoadImage() ] + cfg.data.test.pipeline[1:]
        # print(cfg.data.test.pipeline[1:])
        
        test_pipeline = Compose(test_pipeline)
        # prepare data
        data = dict(img=img)
        data = test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device])[0]
            # print('++++++++',data)
        else:
            data['img_metas'] = [i.data[0] for i in data['img_metas']]
        # print('------------------------------')
        # print(torch.unique(data['img'][0]))
    # else:
    #     print('这里')
    #     print(cfg.data.test.pipeline[1:])
    #     pipeline = [{'type': 'MultiScaleFlipAug', 
    #                  'img_scale': (256, 256), 
    #                  'flip': False, 
    #                  'transforms': 
    #                             [
    #                              # {'type': 'Resize', 'keep_ratio': True}, 
    #                              # {'type': 'ResizeToMultiple', 'size_divisor': 32},
    #                              # {'type': 'RandomFlip'}, 
    #                              {'type': 'Normalize', 'mean': [123.675, 116.28, 103.53], 'std': [58.395, 57.12, 57.375], 'to_rgb': True}, 
    #                              # {'type': 'ImageToTensor', 'keys': ['img']}, 
    #                              {'type': 'Collect', 'keys': ['img']} 

    #                              ]
    #                 }]
    #     test_pipeline = [ LoadImage() ] + pipeline

    #     test_pipeline = Compose(test_pipeline)
    #     # prepare data
    #     data = dict(img=img)
    #     data = test_pipeline(data)

    #     # prinit(data.shape)

    #     data = collate([data], samples_per_gpu=1)
    #     if next(model.parameters()).is_cuda:
    #         # scatter to specified GPU
    #         data = scatter(data, [device])[0]
    #     else:
    #         data['img_metas'] = [i.data[0] for i in data['img_metas']]

    #     # data = {}
    #     # # img = kornia.enhance.Normalize(mean=torch.Tensor([123.675, 116.28, 103.53]), std=torch.Tensor([58.395, 57.12, 57.375]))(img)
    #     # data['img'] = [img]
    #     # # print(torch.unique(data['img'][0]))
    #     # data['img_metas'] =  [[{'filename': None, 'ori_filename': None, 'ori_shape': (256, 256, 3), 'img_shape': (256, 256, 3), 'pad_shape': (256, 256, 3), 'scale_factor': np.array([1., 1., 1., 1.], dtype=np.float32), 'flip': False, 'flip_direction': 'horizontal', 'img_norm_cfg': {'mean': np.array([123.675, 116.28 , 103.53 ], dtype=np.float32), 'std': np.array([58.395, 57.12 , 57.375], dtype=np.float32), 'to_rgb': True}}]]
     


    # cfg = model.cfg
    # device = next(model.parameters()).device  # model device
    # # build the data pipeline
    # test_pipeline = [LoadImage(istrain=istrain )] + cfg.data.test.pipeline[1:]
    # test_pipeline = Compose(test_pipeline)
    # # prepare data
    # data = dict(img=img)
    # data = test_pipeline(data)
    # data = collate([data], samples_per_gpu=1)
    # if next(model.parameters()).is_cuda:
    #     # scatter to specified GPU
    #     data = scatter(data, [device])[0]
    # else:
    #     data['img_metas'] = [i.data[0] for i in data['img_metas']] 

    # forward the model
    with torch.no_grad():
        # print('输入poolformer的图像尺寸：',data['img'][0].shape)# torch.Size([1, 3, 512, 512])
        result = model(return_loss=False, rescale=True, **data)
        # print('输入poolformer的图像尺寸：',result[0].shape)#numpy 
    return result



def show_result_pyplot(model,
                       img,
                       result,
                       palette=None,
                       fig_size=(15, 10),
                       opacity=0.5,
                       title='',
                       block=True):
    """Visualize the segmentation results on the image.

    Args:
        model (nn.Module): The loaded segmentor.
        img (str or np.ndarray): Image filename or loaded image.
        result (list): The segmentation result.
        palette (list[list[int]]] | None): The palette of segmentation
            map. If None is given, random palette will be generated.
            Default: None
        fig_size (tuple): Figure size of the pyplot figure.
        opacity(float): Opacity of painted segmentation map.
            Default 0.5.
            Must be in (0, 1] range.
        title (str): The title of pyplot figure.
            Default is ''.
        block (bool): Whether to block the pyplot figure.
            Default is True.
    """
    if hasattr(model, 'module'):
        model = model.module
    img = model.show_result(
        img, result, palette=palette, show=False, opacity=opacity)
    plt.figure(figsize=fig_size)
    plt.imshow(mmcv.bgr2rgb(img))
    plt.title(title)
    plt.tight_layout()
    plt.show(block=block)
