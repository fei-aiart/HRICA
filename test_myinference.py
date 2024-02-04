import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
import time
from PIL import Image
import random
import numpy as np
import torchvision.transforms as transforms
import torch.utils.data as data
import cv2

 
def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.resize_or_crop == 'resize_and_crop':
        new_h = new_w = opt.loadSize            
    elif opt.resize_or_crop == 'scale_width_and_crop':
        new_w = opt.loadSize
        new_h = opt.loadSize * h // w

    x = random.randint(0, np.maximum(0, new_w - opt.fineSize))
    y = random.randint(0, np.maximum(0, new_h - opt.fineSize))
    
    flip = random.random() > 0.5
    return {'crop_pos': (x, y), 'flip': flip}

def get_transform(opt, params, method=Image.BICUBIC, normalize=True):
    transform_list = []
    if 'resize' in opt.resize_or_crop:
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, method))   
    elif 'scale_width' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __scale_width(img, opt.loadSize, method)))
        
    if 'crop' in opt.resize_or_crop:
        transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.fineSize)))

    if opt.resize_or_crop == 'none':
        base = float(2 ** opt.n_downsample_global)
        if opt.netG == 'local':
            base *= (2 ** opt.n_local_enhancers)
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base, method)))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))

    transform_list += [transforms.ToTensor()]

    if normalize:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class Geometry_Aware_Drawing_GEneraTor(object):
    def __init__(self,):
        self.opt = TestOptions().parse(save=False)
        self.opt.nThreads = 0   # test code only supports nThreads = 1
        self.opt.batchSize = 1  # test code only supports batchSize = 1
        self.opt.serial_batches = True  # no shuffle
        self.opt.no_flip = True  # no flip
        self.opt.netG = 'stack'
        # self.opt.which_epoch = 'latest'
        self.opt.checkpoint = './checkpoints/pix2pixHDBuQuanSpade2.2.3.2/latest_net_G.pth'

        self.model = create_model(self.opt)
 

    def GADGET_generate(self, sketch_PIL, sketch_parsing_PIL):

         
        params = get_params(self.opt, sketch_PIL.size)
        transform = get_transform(self.opt, params)
        sketch_tensor = transform(sketch_PIL.convert('RGB'))
         
        transform_parsing = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) 
        sketch_parsing_tensor  = transform_parsing(sketch_parsing_PIL)*255



        sketch_tensor = sketch_tensor.unsqueeze(0)
        sketch_parsing_tensor = sketch_parsing_tensor.unsqueeze(0)
        # print(sketch_tensor.shape)
        # print(sketch_parsing_tensor.shape)


        generated = self.model.inference(sketch_tensor, image= None, label_parsing=sketch_parsing_tensor)
 
        return util.tensor2im(generated.data[0])
       

def main():

    GADGET = Geometry_Aware_Drawing_GEneraTor()
    img_path = '/data/Datasets/CelebAMaskHQ_hry/test/random_erasing_vec2_select/2.jpg'
    parsing_path = '/data/Datasets/CelebAMaskHQ_hry/test/parsing/2.png'
    

    img_PIL = Image.open(img_path)
    parsing_PIL = Image.open(parsing_path)
    import time
    temp = time.time()

    result = GADGET.GADGET_generate(img_PIL,parsing_PIL)
    print('单张消耗的时间=',time.time()-temp)

   
    cv2.imwrite('111.jpg',result)






if __name__ == '__main__':
    main()




