#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @File  : randimErasing.py
# @Author: Dailingna
# @time: 2022/05/30 下午2：55
# @Desc  :


import cv2
import numpy as np
import os
import random



def number_of_certain_probability(  sequence, probability):
    '''
    sequence = [0, 1]
    probability = [0.2,0.7]# 表示有0.2的概率选中0；0.7的概率选中1
    '''
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(sequence, probability):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item
 
def Skeletonization(img, Blur=True):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)

    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    img = cv2.bitwise_not(img)
    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()
 
        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True

    # ret, img2 = cv2.threshold(cv2.GaussianBlur(skel, (3, 3), 0), 1, 255, 0) # original
    ret, img2 = cv2.threshold(cv2.GaussianBlur(skel, (3, 3), 0), 1, 255, 0)
 
    img2 = cv2.dilate(img2, element, iterations=2)  # 2
    return cv2.bitwise_not(img2)
  

class randomErasing0(object):
    def __init__(self, p=0.5, sl=0.02, sh=0.4, r1=0.3, r2=3):
        self.p = p
        self.sl = sl
        self.sh = sh
        self.r1 = r1
        self.r2 = r2

    def random_erasing(self, img): # img:cv2# 使用随机大小的空白正方形 去随机擦除img的随机位置
        
        # if np.random.rand() > self.p: # 表示有self.p的概率要进行遮挡处理
        #     return img
        # img = np.array(img)
        
        while True:
            img_h, img_w, img_c = img.shape
            # print('img_w=',img_w,'img_h',img_h)

            img_area = img_h * img_w
            mask_area = np.random.uniform(self.sl, self.sh) * img_area
            mask_aspect_ratio = np.random.uniform(self.r1, self.r2)
            mask_w = int(np.sqrt(mask_area / mask_aspect_ratio))
            mask_h = int(np.sqrt(mask_area * mask_aspect_ratio))
            # print('mask_w=',mask_w,'mask_h',mask_h)


            # 此处可以选择将mask初始化成任意形状
            choose = random.choice([0,1,2,3])
            if choose == 0: # 初始化一个方形
                mask = np.ones((mask_h, mask_w, img_c)) *255
            elif choose == 1: # 初始化一个圆形
                mask = cv2.imread('./data/images/zero.png')
                mask = cv2.resize(mask,(mask_w,mask_h))
            elif choose==2: # 初始化一个三角形
                mask = cv2.imread('./data/images/three.png')
                mask = cv2.resize(mask,(mask_w,mask_h))           
            elif choose ==3: # 初始化一个五角星
                mask = cv2.imread('./data/images/five.png')
                mask = cv2.resize(mask,(mask_w,mask_h))
             
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            right = left + mask_w
            bottom = top + mask_h
        
            if right <= img_w and bottom <= img_h:
                break
        
        # img[top:bottom, left:right, :] = mask
        index = np.where(mask == 255)# mask==255的位置表示要被擦除的位置
        img[top:bottom, left:right, :][index[0],index[1]] = mask[index[0],index[1]]
        
        
        return img



    def random_deleteBaseParsing(self,img,parsing):

        # 根据parsing随机删掉人脸相应部位（主要针对眉毛、眼睛、耳朵、鼻子、嘴巴、（五官）,头发内部） 
        indexs = [[2],[3],[5],[7],[8,9,10],[14]] # 此时的indexs对应的是celebehqmask的16通道的parsing
        # 设定随机去多种五官
        # index = np.random.randint(0, len(indexs))

        index_list = random.sample(range(0,len(indexs)),np.random.randint(0, len(indexs)))
        # index_list = [0,1,5]
        
        mask = np.zeros((parsing.shape[0],parsing.shape[1]))
        temp = np.zeros((parsing.shape[0],parsing.shape[1]))
        flag=0# 表示是否去掉头发内部线条
        for index in index_list:
            for i in indexs[index]:
                if index  in [0,1,2]: # 当眉毛、眼睛、耳朵的时候选择其中一只去删除
                    a = self.choosehalf(parsing,i)
                elif i == 14:  # 头发的时候需要往里腐蚀，去掉头发内部的线条
                    flag = 1
                    continue
                else:
                    a = np.where(parsing == i)
                 
                mask[a[0],a[1]]=1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)) # 可以修改卷积核大小来增加腐蚀效果，越大腐蚀越强
        mask = cv2.dilate(mask,kernel)

        b = np.where(mask==1)
      
        img[b[0],b[1]]=255


        if flag==1:
            a = np.where(parsing == 14)
            temp[a[0],a[1]]=1
            kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7)) # 可以修改卷积核大小来增加腐蚀效果，越大腐蚀越强

            temp = cv2.erode(temp,kernel2)
            temp = cv2.erode(temp,kernel2)

            b = np.where(temp==1)
            img[b[0],b[1]]=255

        return img

    def choosehalf(self,parsing,i):
        part = np.zeros((parsing.shape[0],parsing.shape[1]))
        index = np.where(parsing==i)
        if len(index[0])!=0:
            part[index[0],index[1]]=1
            
            min_w, max_w = min(index[1]) , max(index[1]) 
            
            half_w = (max_w-min_w)//2+min_w
           

            a = random.choice([0,1]) # 0表示选择左边，1表示选择右边
         
            if  a ==0:
                part[: ,half_w:max_w]=0
            else:
                part[:,min_w:half_w]=0
           
            return np.where(part==1)

        else: return np.where(parsing==i)

    def random_blur_noise(self,image):
        
        img_h, img_w, img_c = image.shape
        for _ in range(100):
            mask_w,mask_h = 25,25
            mask = cv2.imread('./data/images/zero.png')
            mask = cv2.resize(mask,(mask_w,mask_h))
            left = np.random.randint(0, img_w)
            top = np.random.randint(0, img_h)
            right = left + mask_w
            bottom = top + mask_h
        
            # if right <= img_w and bottom <= img_h:
            #     break
            if right >= img_w or bottom >= img_h:
                break

            index = np.where(mask == 255)

            image[top:bottom, left:right, :][index[0],index[1]] = mask[index[0],index[1]]


        return image



    def random_deleteline(self,img):
         
        img_onehot = Skeletonization(img[:,:,0])
        
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(255-img_onehot) # 在黑色的背景里找白色线条
        deletlabels = random.sample(np.unique(labels).tolist(), random.randint(int(num_labels*0.4), int(num_labels*0.5))) 

        for  deletlabel in deletlabels:
            index = np.where(labels==deletlabel)
            img[index[0],index[1]]=255

        return img
            
    def randomerasing0(self,image,parsing=None):
        chooses = [0, 2, 3, 5] # 0表示不进行任何处理，1表示选择simplify, 2 表示选择对其进行去掉五官的操作，3 表示选择对其进行去掉各类形状块，4 表示选择对其随机加白色躁点
         
        # index_list = random.sample(range(0,len(chooses)),np.random.randint(0, len(chooses)))
        index_list = random.sample(chooses,np.random.randint(1, 3))# 最多只能用两种擦除方法

        if len(index_list)==0:
            result = image
        first = True
        for a in index_list:
            if first == False:
                image = result
            first = False

            if a == 3:
                result= self.random_erasing(image)
            elif a ==2:
                result = self.random_deleteBaseParsing(image,parsing)
            
            # elif a ==1:
            #     simplify_path = os.path.join('/data/Datasets/CelebAMaskHQ_hry/test/baimiao_nobg_simplify',name)
            #     result = cv2.imread(simplify_path)
            
            elif a ==0:
                result = image
            
            # elif a ==4:
            #     result = self.random_blur_noise(image)

            elif a ==5:
                result= self.random_deleteline(image)

        return result

# atts2 = [0:'background', 1:'skin', 2:'brow', 3:'eye', 4:'eye_g', 5:'ear', 6:'ear_r',
#         7:'nose', 8:'mouth', 9:'u_lip', 10:'l_lip', 11:'neck', 12:'neck_l', 13:'cloth', 14:'hair', 15:'hat']
class randomAdd(object):
    def __init__(self,):
        pass

    def choosehalf(self,parsing,i):
        part = np.zeros((parsing.shape[0],parsing.shape[1]))
        index = np.where(parsing==i)
        if len(index[0])!=0:
            part[index[0],index[1]]=1
            
            min_w, max_w = min(index[1]) , max(index[1]) 
            
            half_w = (max_w-min_w)//2+min_w
        
            a = random.choice([0,1]) # 0表示选择左边，1表示选择右边
         
            if  a ==0:
                part[: ,half_w:max_w]=0
            else:
                part[:,min_w:half_w]=0
           
            return np.where(part==1)

        else: 
            return np.where(parsing==i)

    def AddFacialFeatures(self,img_cv, parsing, img_cv_handle):# img_cv:表示完整的线条，img_cv_handle : 表示经过parsing处理后得到的线条
        # 根据parsing随机增加人脸相应部位（主要针对眉毛、眼睛、耳朵、鼻子、嘴巴、（五官）） 
        indexs = [[2],[3],[5],[7],[8,9,10]] # 此时的indexs对应的是celebehqmask的16通道的parsing
         

        index_list = random.sample(range(0,len(indexs)),np.random.randint(0, len(indexs)))
        # index_list = [1,3]
        
        mask = np.zeros((parsing.shape[0],parsing.shape[1]))
          
        for index in index_list:
            for i in indexs[index]:
                if index  in [0,1,2]:# 当眉毛、眼睛、耳朵的时候选择其中一只去删除
                    a = self.choosehalf(parsing,i) 
                else:
                    a = np.where(parsing == i)
                a = np.where(parsing == i)
                mask[a[0],a[1]]=1

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)) # 可以修改卷积核大小来增加腐蚀效果，越大腐蚀越强
        mask = cv2.dilate(mask,kernel)

        b = np.where(mask==0)
        
        FacialFeatures = img_cv.copy()
        FacialFeatures[b[0],b[1]]=255
        img_cv_handle =  (img_cv_handle/255)*(FacialFeatures/255)*255

        # img_cv_handle = FacialFeatures 
        # img_cv_handle = 255-((255-img_cv_handle)+(255-FacialFeatures))*255
        return img_cv_handle

    def AddHairOutline(self,img_cv, parsing, img_cv_handle):
        '''
        img_cv:512*512*3
        parsing:512*512
        '''
        # 获取头发parsing 
        hairParsing = np.ones(parsing.shape)
        index = np.where(parsing == 0)
        hairParsing[index[0],index[1]]=0


        hairOutline = img_cv.copy()

        # 得到faceparsing 膨胀以及腐蚀后的结果（腐蚀1值变0；膨胀0值变1）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)) # 可以修改卷积核大小来增加腐蚀效果，越大腐蚀越强
        hairParsing_dliate = cv2.dilate(hairParsing,kernel)# 膨胀，1的范围扩大
        hairParsing_erode = cv2.erode(hairParsing,kernel)# 腐蚀 1的范围缩小

        index = np.where(hairParsing_dliate==0)
        hairOutline[index[0],index[1]]=255

        index = np.where(hairParsing_erode==1)
        hairOutline[index[0],index[1]]=255

        

        # img_cv_handle = 255-((255-img_cv_handle)+(255-hairOutline))*255
        img_cv_handle =  (img_cv_handle/255)*(hairOutline/255)*255 



        return img_cv_handle

    def AddFaceOutline(self,img_cv, parsing, img_cv_handle):
        '''
        img_cv:512*512*3
        parsing:512*512
        '''
        # 获取脸部parsing('skin', 'brow', 'eye', 'eye_g', 'ear', 'ear_r', 'nose', 'mouth', 'u_lip', 'l_lip',)
         
        faceParsing = np.zeros(parsing.shape)
        index = np.where((parsing== 1) | (parsing==2) | (parsing==3) |
                         (parsing==4) | (parsing==7) | (parsing==8) | (parsing==9) | (parsing==10))
        faceParsing[index[0],index[1]]=1


        faceOutline = img_cv.copy()

        # 得到faceparsing 膨胀以及腐蚀后的结果（腐蚀1值变0；膨胀0值变1）
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(9,9)) # 可以修改卷积核大小来增加腐蚀效果，越大腐蚀越强
        faceParsing_dliate = cv2.dilate(faceParsing,kernel)# 膨胀，1的范围扩大
        faceParsing_erode = cv2.erode(faceParsing,kernel)# 腐蚀 1的范围缩小

        index = np.where(faceParsing_dliate==0)
        faceOutline[index[0],index[1]]=255
        index = np.where(faceParsing_erode==1)
        faceOutline[index[0],index[1]]=255
       

        img_cv_handle = (img_cv_handle/255)*(faceOutline/255)*255

        return img_cv_handle


    # def randomadd(self, img_cv, parsing):
    #     # IsAddFaceOuline = random.randint(0,1) # 返回0 <= N <= 1之间的一个整数
    #     # IsHairOutline = random.randint(0,1) # 返回0 <= N <= 1之间的一个整数
    #     # IsFacialFeatures = random.randint(0,1) # 返回0 <= N <= 1之间的一个整数 # 这个设定的概率大一点

    #     IsAddFaceOuline =  number_of_certain_probability([0,1], [0.2,0.8])
    #     IsHairOutline =  number_of_certain_probability([0,1], [0.2,0.8])
    #     IsFacialFeatures =  number_of_certain_probability([0,1], [0.2,0.8])
        
    #     img_cv_handle= np.ones(img_cv.shape)*255   

    #     if IsAddFaceOuline == 1:
    #         img_cv_handle = self.AddFaceOutline(img_cv, parsing,img_cv_handle=img_cv_handle) 
    #         if IsHairOutline == 1:
    #             img_cv_handle = self.AddHairOutline(img_cv, parsing,img_cv_handle=img_cv_handle)     
    #         if IsFacialFeatures ==1:
    #             img_cv_handle = self.AddFacialFeatures(img_cv, parsing,img_cv_handle=img_cv_handle)
    #     else:
    #         img_cv_handle = self.AddFacialFeatures(img_cv, parsing,img_cv_handle=img_cv_handle)

    #     return img_cv_handle

     

    def randomadd2(self, img_cv, parsing):
        choose =  number_of_certain_probability([1,2,3,4,5], [0.2, 0.2, 0.2, 0.2, 0.2])
        img_cv_handle= np.ones(img_cv.shape)*255  
        if choose == 1:
            img_cv_handle = self.AddFaceOutline(img_cv, parsing,img_cv_handle=img_cv_handle) 
        elif choose == 2:
            img_cv_handle = self.AddFaceOutline(img_cv, parsing,img_cv_handle=img_cv_handle) 
            img_cv_handle = self.AddFacialFeatures(img_cv, parsing,img_cv_handle=img_cv_handle)
        elif choose == 3:
            img_cv_handle = self.AddFaceOutline(img_cv, parsing,img_cv_handle=img_cv_handle) 
            img_cv_handle = self.AddHairOutline(img_cv, parsing,img_cv_handle=img_cv_handle)   
            img_cv_handle = self.AddFacialFeatures(img_cv, parsing,img_cv_handle=img_cv_handle)
        elif choose == 4:
            img_cv_handle = self.AddFacialFeatures(img_cv, parsing,img_cv_handle=img_cv_handle)
        elif choose == 5:
            img_cv_handle = self.AddFaceOutline(img_cv, parsing,img_cv_handle=img_cv_handle) 
            img_cv_handle = self.AddHairOutline(img_cv, parsing,img_cv_handle=img_cv_handle) 

        return img_cv_handle
         

   
         



class RandomErasing(randomErasing0,randomAdd):
    def __inif__(self,):
        pass
 
    def RandomErasing2(self,image,parsing=None):
        '''
        image:(512, 512, 3)
        parsing:(512, 512)
        '''
        # choose = random.randint(0,1) # 选择0 表示使用randomErasing0类，选择1 表示使用randomAdd类
        value_list = [0, 1]
        probability = [0.3,0.7]# 表示有0.2的概率选中0；0.7的概率选中1

        choose =  number_of_certain_probability(value_list, probability)
       
        if choose ==0:
            result =  self.randomerasing0(image, parsing=parsing)

        elif choose ==1:
            result =  self.randomadd2(image, parsing=parsing)
      
        return result.astype(np.uint8)

   




def main():
    image_dir = './datasets/CelebaLine/test/SimplifySketch'
    parsing_dir = './datasets/CelebaLine/test/parsing'
    save_dir = './datasets/CelebaLine/test/SimplifySketch_erased'
    random_erasing = RandomErasing()

    for name in os.listdir(image_dir):
        if name not in ['4.jpg','44.jpg']:
            continue
        img_path = os.path.join(image_dir,name)
        parsing_path = os.path.join(parsing_dir,name.replace('.jpg','.png'))
        save_path = os.path.join(save_dir, name)


        image = cv2.imread(img_path)
        parsing = cv2.imread(parsing_path)




        result = random_erasing.RandomErasing2(image,parsing)

        cv2.imwrite(save_path,result)

    pass


if __name__ == '__main__':
    main()
