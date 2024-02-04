import os.path
from data.base_dataset import BaseDataset, get_params, get_transform, normalize
from data.image_folder import make_dataset
from PIL import Image
from .randomErasing import RandomErasing
import cv2
import numpy as np
from .PoolTransformTest import PoolTransform

class AlignedDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        self.A_paths, self.B_paths = [], []
        self.dir_parsing = ''
        
        self.randomErasing = RandomErasing() 
        if not self.opt.no_instance:
            self.PoolTransform = PoolTransform()

        
        if opt.phase == 'train':
            self.dir_A1 = './datasets/CelebaLine/train/SimplifySketch'
            self.dir_B1 = './datasets/CelebaLine/train/SimplifySketch'
            self.dir_parsing = './datasets/CelebaLine/train/parsing'
            self.random_erasing=True
            if self.opt.geom_loss == True:
                self.dir_depth = './datasets/CelebaLine/train/depth'
        
        elif opt.phase == 'test':
            self.dir_A1 = './datasets/CelebaLine/test/SimplifySketch_erased'
            self.dir_B1 = './datasets/CelebaLine/test/SimplifySketch_erased'
            self.dir_parsing = './datasets/CelebaLine/test/parsing'
            self.random_erasing = False
            if self.opt.geom_loss == True:
                self.dir_depth = './datasets/CelebaLine/test/depth_BoostLeRas'  # png

        self.A_paths, self.B_paths = [],[]
        self.parsing_paths = []
        self.depth_paths = []
        self.A_paths_name, self.B_paths_name = os.listdir(self.dir_A1),os.listdir(self.dir_B1),
        self.A_paths_name.sort()
        self.B_paths_name.sort()
        for A_name,B_name in zip(self.A_paths_name,self.B_paths_name):
            self.A_paths.append(os.path.join(self.dir_A1, A_name),)
            self.B_paths.append(os.path.join(self.dir_B1, B_name),)
            
            if self.dir_parsing !='':
                self.parsing_paths.append(os.path.join(self.dir_parsing, A_name.split('.')[0]+'.png'))
            if self.opt.geom_loss == True:
                self.depth_paths.append(os.path.join(self.dir_depth, A_name.split('.')[0]+'.png'))
        self.dataset_size = len(self.A_paths)
      
    def __getitem__(self, index):        
        ### input A  
        A_path = self.A_paths[index]              
        A = Image.open(A_path)
        A = A.resize((512, 512)) 

        if self.random_erasing==True:
            parsing = cv2.imread(self.parsing_paths[index])
            A = cv2.cvtColor(np.asarray(A), cv2.COLOR_RGB2BGR)
            A_temp = self.randomErasing.RandomErasing2(A,parsing)
            A = Image.fromarray(cv2.cvtColor(A_temp, cv2.COLOR_BGR2RGB))

        params = get_params(self.opt, A.size)
        transform_A = get_transform(self.opt, params)
        A_tensor = transform_A(A.convert('RGB'))
        
        inst_tensor = feat_tensor = 0

        ### input B  
        B_path = self.B_paths[index]  
        if self.opt.output_nc == 3: 
            B = Image.open(B_path).convert('RGB')
        else:
            B = Image.open(B_path).convert('L')
        transform_B = get_transform(self.opt, params)      
        B_tensor = transform_B(B)

        input_dict = {'label': A_tensor,  'image': B_tensor, 'feat': feat_tensor, 'path': A_path}

        input_dict['label_parsing']=0
        if not self.opt.no_instance:
            if self.random_erasing==True:
                label_parsing = self.generateParsing(A_temp)
            else:
                label_parsing = self.generateParsing(cv2.imread(A_path))
            
            transform_parsing = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) 
            input_dict['label_parsing'] = transform_parsing(label_parsing)*255
             
        input_dict['depth'] = 0
        if self.opt.geom_loss == True and self.opt.phase == 'train':
        # if self.opt.geom_loss == True  :
            depth = Image.open(self.depth_paths[index])
             
            transform_depth = get_transform(self.opt, params) 
            depth_tensor = transform_depth(depth.convert('RGB'))

            input_dict['depth'] = depth_tensor

        input_dict['parsing'] = 0
        if self.opt.poolformer_loss == True and self.opt.phase == 'train':
        # if self.opt.poolformer_loss == True  :
            parsing = Image.open(self.parsing_paths[index])
            
            transform_parsing = get_transform(self.opt, params, method=Image.NEAREST, normalize=False) 
            parsing_tensor = transform_parsing(parsing)*255

            input_dict['parsing'] = parsing_tensor

        return input_dict

    def generateParsing(self,img_cv): # return PIL
        # img_cv = cv2.resize(img_cv,(self.opt.fineSize,self.opt.fineSize))
        _, inst_cv = self.PoolTransform.generate(img_cv, isVisual=False)
        cv2.imwrite('inst_cv_parsing2_2_3_2.png', inst_cv)
        inst = Image.open('inst_cv_parsing2_2_3_2.png')
        return inst

    def __len__(self):
        return len(self.A_paths) // self.opt.batchSize * self.opt.batchSize

    def name(self):
        return 'AlignedDataset'