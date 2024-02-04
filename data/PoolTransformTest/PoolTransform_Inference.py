import torch
import torchvision.transforms.transforms as T
import sys
sys.path.append("/mnt/home/zhujingjie/projects/dailingna/dln_project/pix2pixHDUpdateFinal/data/PoolTransformTest")


from mmsegMy.apis import init_segmentor
from mmsegMy.apis import inference_segmentor, show_result_pyplot
from mmsegMy.core.evaluation import get_palette
# from mmseg.datasets import LaPaDataset



from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import mmcv
import numpy as np
from PIL import Image
import os
from .dataloader import datasetLoader
from collections import Counter
import cv2
import torch.nn.functional as F
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'


# 0:bg 1.face,2,left_eyebrow,3.right_eyebrow,4.left_eye,5.right_eye,6.nose,7.upper_lip,8.inner_mouth,9.lower_lip,10.hair

# def get_dataloder():
#     img_dir = '/data/Datasets/LaPa/test/images'
#     img_path = []
#     ann_dir = '/data/Datasets/LaPa/test/labels'
#     ann_path = []
#     for filename in sorted(os.listdir(img_dir)):
#         img_path.append(os.path.join(img_dir, filename))
#     for filename in sorted(os.listdir(ann_dir)):
#         ann_path.append(os.path.join(ann_dir, filename))

#     img_norm_cfg = dict(
#         mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])# , to_rgb=True
#     transforms = T.Compose([T.Resize(512),
#                             T.ToTensor(),
#                             T.Normalize(**img_norm_cfg),
#                             ])
#                 # dict(type='Normalize', **img_norm_cfg),
#                 # dict(type='ImageToTensor', keys=['img']),
#                 # dict(type='Collect', keys=['img'])])

#     lapa_dataset = LaPa_dataset(img_path, ann_path, transforms)

#     test_dataloader = torch.utils.data.DataLoader(lapa_dataset,
#                                                  batch_size=1,
#                                                  shuffle=False,
#                                                  num_workers=8)
#     return test_dataloader


def deal_label(ann):
    ann[ann==0] += 11
    ann -= 1
    return ann

def vis(ann,save_dir,channel=11):
    for i in range(channel):
        save = np.zeros(ann.shape).astype(np.uint8)
        temp = np.where(ann==i)
        save[temp[0],temp[1]]=255

        save_path = os.path.join(save_dir,str(i)+'.png')
        cv2.imwrite(save_path,save)


def handleParsing2Mat(parsing):
        
    mat = np.zeros((parsing.shape[0],parsing.shape[1], 16)).astype(np.uint8)
    for index in np.unique(parsing):
        location = np.where(parsing==index)
        mat[:,:,index][location[0],location[1]]=1
    return mat


def face_parsing(im, out, image_name, save_dir, channel_num=16):
    
    im = np.array(im)
    
    vis_im = im.copy().astype(np.uint8)
    part_colors = [[0, 0, 0], [255, 85, 0], [255, 170, 0], [255, 170, 85], [255, 0, 170],
                   [0, 255, 0], [85, 255, 0], [170, 255, 0], [0, 255, 85], [0, 255, 170],
                   [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255], [0, 170, 255],
                   [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255], 
                   [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255],[200, 85, 200]
                   ]
    anno_color = np.zeros((512, 512, 3))
    for i in range(channel_num):

        channel = np.zeros(out.shape).astype(np.uint8)
        temp = np.where(out==i)
        channel[temp[0],temp[1]]=1
         
        temp = np.expand_dims(channel, axis=2).repeat(3, axis=2)
        new_part_colors = temp * part_colors[i]
        anno_color += new_part_colors

    anno_color = np.clip(anno_color, 0, 255).astype(np.uint8)
    vis_im = cv2.addWeighted(cv2.cvtColor(vis_im, cv2.COLOR_RGB2BGR), 0.4, anno_color, 0.6, 0)
     
  
    cv2.imwrite(os.path.join(save_dir, image_name), vis_im, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

class CalculatedMetrics(object):
    def __init__(self,):
        pass

    def mIoU(self, y_true, y_pred, num_class=2):
        mask = (y_true >= 0) & (y_true < num_class)
        temp = np.bincount(num_class * y_true[mask] + y_pred[mask])
        if len(temp) == 1: return 1.
        if len(temp) < num_class**2:
            temp_bin = np.zeros(num_class**2).astype(np.int64)
            for i in range(len(temp)):
                temp_bin[i] = temp[i]
            temp = temp_bin
        matrix = temp.reshape(num_class, num_class)
        intersection = np.diag(matrix)
        union = np.sum(matrix, axis=1) + np.sum(matrix, axis=0) - np.diag(matrix)
        IoU = intersection / union
        m_IoU = np.nanmean(IoU)
        return m_IoU

    def _as_floats(self,im1, im2):
        """Promote im1, im2 to nearest appropriate floating point precision."""
        float_type = np.result_type(im1.dtype, im2.dtype, np.float32)
        im1 = np.asarray(im1, dtype=float_type)
        im2 = np.asarray(im2, dtype=float_type)
        return im1, im2

    def compare_mse(self,im1, im2):
        """Compute the mean-squared error between two images.

        Parameters
        ----------
        im1, im2 : ndarray
            Image.  Any dimensionality.

        Returns
        -------
        mse : float
            The mean-squared error (MSE) metric.

        """
        # _assert_compatible(im1, im2)
        im1, im2 = self._as_floats(im1, im2)
        return np.mean(np.square(im1 - im2), dtype=np.float64)

    def mIoU_matrix_value(self, labels, predicts,n_classes):
        predicts = predicts.astype(np.int64)
        temp_array = np.zeros(n_classes)
        for i in range(n_classes):
            label = labels[i]
            predict = predicts[i]
            temp_array[i] = self.mIoU(label, predict)
            # cv2.imwrite('label_{}.jpg'.format(i),label*255)
            # cv2.imwrite('predict_{}.jpg'.format(i),predict*255)

        return np.mean(temp_array), temp_array
        

    def mse_matrix_value(self,y_true, y_pre,n_classes):
        temp_array = np.zeros(n_classes)
        for i in range(n_classes):
            label = y_true[i]
            predict = y_pre[i]
            temp_array[i] = self.compare_mse(label, predict)
        return np.mean(temp_array), temp_array




def cal_metric(config_file, checkpoint_file):
    calculated_metrics = CalculatedMetrics()
    channel = 16


    # img_dir = '/data/Datasets/CelebAMaskHQ_hry/test/random_erasing_vec'
    # img_path = []
    # ann_dir = '/data/Datasets/CelebAMaskHQ_hry/test/parsing'
    # ann_path = []


    # for filename in sorted(os.listdir(img_dir)):
    #     img_path.append(os.path.join(img_dir, filename))
    # for filename in sorted(os.listdir(ann_dir)):
    #     ann_path.append(os.path.join(ann_dir, filename))




    # vis_dir = '/mnt/home/zhujingjie/projects/dailingna/dln_project/face_parsing_lyh/VIS_result/random'
    # if not os.path.exists(vis_dir):
    #     os.mkdir(vis_dir)



    class_TP, class_FP, class_FN = Counter(), Counter(), Counter()
    Fscore = []
    model = init_segmentor(config_file, checkpoint_file, device='cuda:1')

    num = len(img_path)
    mIoU_value = 0
    mse_value = 0
    mIoU_ele_Value = [0]*channel
    mse_ele_value = [0]*channel

    for idx, img_file in enumerate(img_path):

        # 读取groundtrue
        ann = plt.imread(ann_path[idx]) # 因为plt的读取，所以需要将数据*255
        ann = np.array(ann * 255).astype('uint8')
        # 读取线条（有擦除）
        input = cv2.imread(img_file)
        input = cv2.resize(input, (ann.shape[1],ann.shape[0]))
        # cv2.imwrite('temp1.jpg',input)


        result = inference_segmentor(model, 'temp1.jpg')[0]
      
        face_parsing(input, result, ann_path[idx].split('/')[-1], vis_dir)
    

class PoolTransform(object):
    def __init__(self, channel=16, config_file='/mnt/home/zhujingjie/projects/dailingna/dln_project/pix2pixHDUpdateFinal/data/PoolTransformTest/local_configs/poolformer/PoolFormer/fpn_poolformer_s36_celebamaskhq_40k.py',
        checkpoint_file='/mnt/home/zhujingjie/projects/dailingna/dln_project/pix2pixHDUpdateFinal/checkpoints/fpn_poolformer_s36_celebamaskhq_40k_simplify_random_erasing_v3/iter_40000.pth'):# V3是不论输入的图是否是完整的，都会预测出完整的parsing
        self.channel = channel

        self.config_file=config_file
        self.checkpoint_file=checkpoint_file


        self.model = init_segmentor(self.config_file, self.checkpoint_file, device='cuda:1')

        self.part_colors = [[0, 0, 0], [255, 85, 0], [255, 170, 0], [255, 170, 85], [255, 0, 170],
                            [0, 255, 0], [85, 255, 0], [170, 255, 0], [0, 255, 85], [0, 255, 170],
                            [0, 0, 255], [85, 0, 255], [170, 0, 255], [0, 85, 255], [0, 170, 255],
                            [255, 255, 0], [255, 255, 85], [255, 255, 170], [255, 0, 255], [255, 85, 255], 
                            [255, 170, 255], [0, 255, 255], [85, 255, 255], [170, 255, 255],[200, 85, 200]
                            ]

        self.calculated_metrics = CalculatedMetrics()

        self.mIoU_value = 0
        self.mse_value = 0
        self.mIoU_ele_Value = [0]*self.channel
        self.mse_ele_value = [0]*self.channel

        self.class_TP, self.class_FP, self.class_FN = Counter(), Counter(), Counter()

    
    def CalculatedMetrics(self, gt, pred):
        mse_total, mse_ele = self.calculated_metrics.mse_matrix_value(gt, pred,self.channel)
        mIoU_total, mIoU_ele = self.calculated_metrics.mIoU_matrix_value(gt, pred,self.channel)
        return mse_total, mse_ele, mIoU_total, mIoU_ele

    def OneToMulti(self, pred, gt=None):
        # 把一通道换成channel通道
        pred_out_shot = np.zeros((self.channel, pred.shape[0], pred.shape[1])).astype(np.uint8)

        if len(np.array(gt ==None).shape) != 0:
            gt_out_shot = np.zeros((self.channel, gt.shape[0], gt.shape[1])).astype(np.uint8)
                    
        
        for i in range(self.channel):
            if len(np.array(gt ==None).shape) != 0:
                index = np.where(gt == i)
                gt_out_shot[i,:,:][index[0],index[1]]=1

            index = np.where(pred == i)
            pred_out_shot[i,:,:][index[0],index[1]]=1


        if len(np.array(gt == None).shape) != 0:
            return pred_out_shot, None
        return pred_out_shot, gt_out_shot


    def visual(self, result, ISaddWeighted=True, input=None,):
        anno_color = np.zeros((result.shape[0], result.shape[1], 3))
        for i in range(self.channel):

            channel = np.zeros(result.shape).astype(np.uint8)
            temp = np.where(result==i)
            channel[temp[0],temp[1]]=1
             
            temp = np.expand_dims(channel, axis=2).repeat(3, axis=2)
            new_part_colors = temp * self.part_colors[i]
            anno_color += new_part_colors

        anno_color = np.clip(anno_color, 0, 255).astype(np.uint8)

        if  ISaddWeighted == True:# 表明需要进行融合可视化
            input = np.array(input)
            vis_input = input.copy().astype(np.uint8)
            vis_result = cv2.addWeighted(cv2.cvtColor(vis_input, cv2.COLOR_RGB2BGR), 0.4, anno_color, 0.6, 0)
        else:
            vis_result = anno_color
        

        return vis_result
    
  

    def generate(self, input, isVisual=True, gt=None,isCalculatedMetrics=False,istrain=False):# 生成parsng。gtParsing表示groundtrue
        if istrain == True:# 表示带代码嵌入到训练代码中，输入的数据是tensor类型
            pred = inference_segmentor(self.model, input, istrain=False) # 输出的是list len(list)==bs
            for i in range(len(pred)):
                pred[i] = handleParsing2Mat(pred[i]).transpose((2,0,1))

            pred = torch.tensor(np.array(pred))
            return None, pred

        elif istrain==False:
            pred = inference_segmentor(self.model, input, istrain)[0]
        # print(pred)

        vis_pred = self.visual(pred, ISaddWeighted=True, input=input)

        if isCalculatedMetrics==True:
            pred_out_shot, gt_out_shot = self.OneToMulti(pred, gt)

            mse_total, mse_ele, mIoU_total, mIoU_ele = self.CalculatedMetrics(gt_out_shot,pred_out_shot)
            self.mse_value += mse_total
            for i in range(len(mse_ele)):
                self.mse_ele_value[i] += mse_ele[i]

            self.mIoU_value += mIoU_total
            for i in range(len(mIoU_ele)):
                self.mIoU_ele_Value[i] += mIoU_ele[i]

            # 计算FScore
            self.class_TP += Counter(gt[gt == pred])
            self.class_FP += Counter(gt[gt != pred])
            self.class_FN += Counter(pred[gt != pred])


            return vis_pred,pred, self.class_TP, self.class_FP, self.class_FN, self.mse_ele_value, self.mIoU_ele_Value




        return vis_pred,  pred

    def useGenerate(self,input,isMixture=True):
         
        # print('-------------')
        x = self.model.extract_feat(input)
        # print(x[0])
        
        out = self.model._decode_head_forward_test_my(x)
        seg_logit = F.softmax(out, dim=1)
        
         
        if isMixture==True:
             
            seg_logit = seg_logit.argmax(dim=1).unsqueeze(0) #torch.Size([1,1, 256, 256])
           
         

        return seg_logit


    




       


if __name__ == '__main__':
    
    config_file = '/mnt/home/zhujingjie/projects/dailingna/dln_project/face_parsing_lyh/PoolTransformTest/local_configs/poolformer/PoolFormer/fpn_poolformer_s36_celebamaskhq_40k.py'
    # config_file = '/mnt/home/zhujingjie/projects/dailingna/dln_project/face_parsing_lyh/PoolTransformTest/fpn_poolformer_s36_celebamaskhq_40k.py'
    checkpoint_file = '/mnt/home/zhujingjie/projects/dailingna/dln_project/face_parsing_lyh/work_dirs/fpn_poolformer_s36_celebamaskhq_40k_simplify_random_erasing_v2/iter_40000.pth'

    channel=16
    pooltransform = PoolTransform(channel=channel,config_file=config_file, checkpoint_file=checkpoint_file)
    
    
    '''
    用于外部测试集进行测试
    '''
    input_dir = '/data/Datasets/CelebAMaskHQ_hry/temp/random_vec'
    input_paths = []
    

    vis_dir = '/mnt/home/zhujingjie/projects/dailingna/dln_project/face_parsing_lyh/PoolTransformTest/vis111'
    if not os.path.exists(vis_dir):
        os.mkdir(vis_dir)




    for filename in sorted(os.listdir(input_dir)):
        input_paths.append(os.path.join(input_dir, filename))
  

    for idx, input_path in enumerate(input_paths):
        input = cv2.imread(input_path)
        result_vis, result = pooltransform.generate( input, isVisual=True, gt=None, isCalculatedMetrics=False)
        save_path = os.path.join(vis_dir,input_path.split('/')[-1])
        cv2.imwrite(save_path,result*20)
        break


   
    
 










