import numpy as np
import torch
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
from data.PoolTransformTest import PoolTransform
import torchvision.utils as vutils


def par_tensor2pix(label, par_dim, one_hot=True, norm=True):
    '''
        Label Content:
        0:         1:face       2:left eyebrow  3:right eyebrow 4:           5:
        6: eye     7:left ear   8: right ear    9:              10:noses     11:
        12:up lip  13:down lip  14:neak         15:             16:clothes   17:hair
        18: 19: 20: 21: 22: 23: 24:
        0[255,255,255]  1[255, 85, 0] 2[255, 170, 0]  3[255, 0, 85]  4[255, 0, 170]  5[0, 255, 0]
        6  7  8 10 12 13 14 16 17
        '''
    label = label[:, :par_dim]
    if one_hot:
        label = soft2num(label).squeeze(1)
    else:
        label = label.permute([0, 2, 3, 1])
        label = label.argmax(dim=-1)
        if len(label.size()) == 2:
            label = label.unsqueeze(0)
    # rgb_list = torch.FloatTensor(
    #     [[169, 209, 142], [181, 215, 243], [128, 64, 128], [128, 64, 128], [0, 0, 0], [0, 0, 0], [153, 153, 153],
    #      [0, 24, 179], [255, 128, 255], [0, 24, 179], [76, 110, 155]
    #         , [140, 181, 241], [172, 58, 43], [42, 49, 32], [162, 0, 163], [228, 165, 0], [66, 214, 109],
    #      [148, 195, 252], [151, 34, 176]]).to(label.device)
    rgb_list = torch.FloatTensor([[255, 0, 0], [255, 85, 0], [255, 170, 0],
                                  [255, 0, 85], [255, 0, 170],
                                  [0, 255, 0], [85, 255, 0], [170, 255, 0],
                                  [0, 255, 85], [0, 255, 170],
                                  [0, 0, 255], [85, 0, 255], [170, 0, 255],
                                  [0, 85, 255], [0, 170, 255],
                                  [255, 255, 0], [255, 255, 85], [255, 255, 170],
                                  [255, 0, 255], [255, 85, 255], [255, 170, 255],
                                  [0, 255, 255], [85, 255, 255], [170, 255, 255]]).to(label.device)

    b, h, w = label.size()

    img = torch.zeros(b, h, w, 3, device=label.device)
    img[label == 0] = rgb_list[0]
    img[label == 1] = rgb_list[1]  # 面部
    img[label == 2] = rgb_list[2]
    img[label == 3] = rgb_list[3]
    img[label == 4] = rgb_list[4]  # 左眼
    img[label == 5] = rgb_list[5]  # 右眼
    img[label == 6] = rgb_list[6]  # 眼镜维度
    img[label == 7] = rgb_list[7]
    img[label == 8] = rgb_list[8]
    img[label == 9] = rgb_list[9]
    img[label == 10] = rgb_list[10]
    img[label == 11] = rgb_list[11]
    img[label == 12] = rgb_list[12]
    img[label == 13] = rgb_list[13]
    img[label == 14] = rgb_list[14]
    img[label == 15] = rgb_list[15]
    img[label == 16] = rgb_list[16]
    img[label == 17] = rgb_list[17]
    img[label == 18] = rgb_list[18]
    if norm:
        img = img / 255
    img = img.permute([0, 3, 1, 2])
    return img

    
class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'
    
    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss, loss_G_geom, loss_G_par, loss_G_globalL1):
        flags = (True, use_gan_feat_loss, use_vgg_loss, loss_G_geom, loss_G_par, loss_G_globalL1, True, True)
        def loss_filter(g_gan, g_gan_feat, g_vgg, g_geom, g_par, g_globall1, d_real, d_fake):
            return [l for (l,f) in zip((g_gan,g_gan_feat,g_vgg,g_geom,g_par,g_globall1,d_real,d_fake),flags) if f]
        return loss_filter
    
    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        if opt.resize_or_crop != 'none' or not opt.isTrain: # when training at full res this causes OOM
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat
        self.gen_features = self.use_features and not self.opt.load_features
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc

        ##### define networks        
        # Generator network
        netG_input_nc = input_nc        
        if not opt.no_instance:
            netG_input_nc += 1
        if self.use_features:
            netG_input_nc += opt.feat_num                  
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG, 
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers, 
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)        

        # Discriminator network
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc
            if not opt.no_instance:
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid, 
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)

        ### Encoder network
        if self.gen_features:          
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder', 
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)  
        if self.opt.verbose:
                print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain
            # print('-----', opt.which_epoch, pretrained_path,'-----')# ----- latest  -----
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)            
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)              

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss, not opt.no_vgg_loss, opt.geom_loss,opt.poolformer_loss,opt.global_l1loss)
            
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)   
            self.criterionFeat = torch.nn.L1Loss()
            if not opt.no_vgg_loss:             
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)
                
            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN','G_GAN_Feat','G_VGG','G_Geom','G_Poolformer','G_globall1','D_real', 'D_fake')

            # initialize optimizers
            # optimizer G
            if opt.niter_fix_global > 0:                
                import sys
                if sys.version_info >= (3,0):
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())
                params = []
                for key, value in params_dict.items():       
                    if key.startswith('model' + str(opt.n_local_enhancers)):                    
                        params += [value]
                        finetune_list.add(key.split('.')[0])  
                print('------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))                         
            else:
                params = list(self.netG.parameters())
            if self.gen_features:              
                params += list(self.netE.parameters())         
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))                            

            # optimizer D                        
            params = list(self.netD.parameters())    
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))

            # depth
            if self.opt.geom_loss == True:# no_geom_loss, True表示不使用该损失
                self.netGeom = networks.define_depth(input_nc=3, output_nc=3, ngf=64, n_downsampling=4, n_blocks=9,  norm='instance', gpu_ids=self.gpu_ids)
                self.netGeom.load_state_dict(torch.load(opt.feats2Geom_path2))
                self.criterionGeom = torch.nn.BCEWithLogitsLoss(reduce=True)
            if self.opt.poolformer_loss == True:
                self.PoolTransform = PoolTransform(checkpoint_file=self.opt.poolformer_checkpoint)
                self.criterionPar = torch.nn.MSELoss()

    def OneToMulti(self, pred, channel=16):
        # 把一通道换成channel通道
        # print(pred.shape)#torch.Size([2, 1, 512, 512])

        pred_out_shot = torch.zeros((pred.shape[0], channel, pred.shape[-2], pred.shape[-1]))
        for i in range(channel):
            index = torch.where(pred[:,0,:,:] == i)
            pred_out_shot[:,i,:,:][index[0],index[1],index[2]]=1
        pred_out_shot = Variable(pred_out_shot.cuda())
        return pred_out_shot 

    def encode_input(self, label_map, real_image=None, label_parsing=None, depth=None, parsing=None, infer=False):             
        if self.opt.label_nc == 0:
            input_label = label_map.data.cuda()
        
        # get edges from instance map
        if not self.opt.no_instance:
            label_parsing = label_parsing.data.cuda()
            input_label = torch.cat((input_label, label_parsing), dim=1)         
        input_label = Variable(input_label, volatile=infer)

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda())
        if depth is not None:
            depth = Variable(depth.data.cuda())
        if parsing is not None:
            parsing = Variable(parsing.data.cuda())

        return input_label, real_image, depth, parsing

    def discriminate(self, input_label, test_image, use_pool=False):
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)
        if use_pool:            
            fake_query = self.fake_pool.query(input_concat)
            return self.netD.forward(fake_query)
        else:
            return self.netD.forward(input_concat)




##### python train.py --no_flip  --resize_or_crop resize_and_crop    --name pix2pixHDBuQuanSpade2.2.3.2 --geom_loss --global_l1loss --poolformer_loss --gpu_ids 1  --loadSize 286 --fineSize 256 --netG stack

    def forward(self, label,  image,  label_parsing=None, depth=None, parsing=None, infer=False):
         
        # Encode Inputs
        input_label, real_image, depth, parsing  = self.encode_input(label, image, label_parsing, depth, parsing)  

        # Fake Generation
        input_concat = input_label
 
        if self.opt.netG == 'global':
            fake_image = self.netG.forward(input_concat)
        elif self.opt.netG == 'stack' :
            fake_image0, fake_image = self.netG.forward(input_concat)

        # Fake Detection and Loss
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)        

        # Real Detection and Loss        
        pred_real = self.discriminate(input_label, real_image)
        loss_D_real = self.criterionGAN(pred_real, True)

        # GAN loss (Fake Passability Loss)        
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)               
        
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)
            D_weights = 1.0 / self.opt.num_D
            for i in range(self.opt.num_D):
                for j in range(len(pred_fake[i])-1):
                    loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach()) * self.opt.lambda_feat
                   
        # VGG feature matching loss
        loss_G_VGG = 0
        if not self.opt.no_vgg_loss:
            loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat


        ########## 以下是自己添加的损失
        # 此处可以加上depth的损失函数
        #######################
        loss_G_geom = 0
        pred_geom=depth
        if self.opt.geom_loss == True:
            geom_input = fake_image.clone() 
            if geom_input.size()[1] == 1:
                geom_input = geom_input.repeat(1, 3, 1, 1)
            pred_geom = self.netGeom(geom_input)# [-1~1]
            # pred_geom = (pred_geom+1)/2.0 ###[-1, 1] ---> [0, 1] # 需要保证gt的depth也在这个数值范围内，由于使用了BCEWithLogitsLoss损失，该步骤可省略
            loss_G_geom = self.criterionGeom(pred_geom,  depth)*3  # pred_geom是fake_image预测出来的depth,y_depth是gt的depth

            if self.opt.netG == 'stack' :
                geom_input = fake_image0.clone() 
                if geom_input.size()[1] == 1:
                    geom_input = geom_input.repeat(1, 3, 1, 1)
                pred_geom = self.netGeom(geom_input) # [-1~1]
                loss_G_geom = loss_G_geom + self.criterionGeom(pred_geom,  depth)*3
        

        loss_G_L1Par = 0
        fake_transform_par=0
        if self.opt.poolformer_loss==True:
            # fake_image_poolformer = real_image.clone()# test
            fake_image_poolformer = fake_image.clone()        
            fake_image_poolformer = torch.nn.functional.interpolate(fake_image_poolformer, scale_factor=2, mode='bicubic', align_corners=False)
            fake_image_poolformer = fake_image_poolformer/2.0+0.5 # -1~1 ->0~1#  
            fake_image_poolformer = self.ReNormal(fake_image_poolformer)
            fake_transform_par = self.PoolTransform.useGenerate(fake_image_poolformer, isMixture=False)
            loss_G_L1Par = self.criterionPar(fake_transform_par, self.OneToMulti(parsing ) )*50
            if self.opt.netG == 'stack':
                fake_image_poolformer = fake_image0.clone()        
                fake_image_poolformer = torch.nn.functional.interpolate(fake_image_poolformer, scale_factor=2, mode='bicubic', align_corners=False)
                fake_image_poolformer = fake_image_poolformer/2.0+0.5 # -1~1 ->0~1#  
                fake_image_poolformer = self.ReNormal(fake_image_poolformer)
                fake_transform_par = self.PoolTransform.useGenerate(fake_image_poolformer, isMixture=False)
                loss_G_L1Par += self.criterionPar(fake_transform_par, self.OneToMulti(parsing ) )*50

        loss_G_globalL1=0
        if self.opt.global_l1loss == True:
            diff = (input_label[:,:3,:,:]==real_image).float().cuda()# 0表示不相同（也就是被擦除的位置），1表示相同
          
            weight = 10
            loss_G_globalL1 = self.criterionFeat(real_image * diff, fake_image * diff) * weight + \
                              self.criterionFeat(real_image * (1 - diff), fake_image * (1 - diff)) * weight * 4
            # weight = 20e5
            # loss_G_globalL1 = (self.criterionFeat(real_image * diff, fake_image * diff)     *   (1/(torch.sum(diff)+1))+ \
            #                   self.criterionFeat(real_image * (1 - diff), fake_image * (1 - diff))    *    (1/(torch.sum(1-diff)+1)))*weight
 
            if self.opt.netG == 'stack':  
                loss_G_globalL1 += self.criterionFeat(real_image * diff, fake_image0 * diff) * weight + \
                              self.criterionFeat(real_image * (1-diff), fake_image0 * (1-diff)) * weight * 4
                # loss_G_globalL1 = (self.criterionFeat(real_image * diff, fake_image * diff) / ((torch.sum(diff)+1))+ \
                #               self.criterionFeat(real_image * (1 - diff), fake_image * (1 - diff)) / ((torch.sum(1-diff)+1)))*weight


        
        # Only return the fake_B image if necessary to save BW
        return [ self.loss_filter( loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_G_geom, loss_G_L1Par, loss_G_globalL1, loss_D_real, loss_D_fake ),
                None if not infer else fake_image,
                pred_geom,
                fake_transform_par,
                ]
    def ReNormal(self,data):
        data[:,0,:,:] = (data[:,0,:,:] * 255-123.675)/58.395
        data[:,1,:,:] = (data[:,1,:,:] * 255-116.28)/57.12
        data[:,2,:,:] = (data[:,2,:,:] * 255-103.53)/57.375
        return data

    def inference(self, label,  image=None, label_parsing=None, depth=None, parsing=None,):
        # Encode Inputs   
        image = Variable(image) if image is not None else None
        input_label = self.encode_input(Variable(label),   real_image=image, label_parsing=label_parsing, depth=None, parsing=None, infer=True)[0]

        # Fake Generation
        if self.use_features:
            if self.opt.use_encoded_image:
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:
                # sample clusters from precomputed features  
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)  
        else:
            input_concat = input_label        
           
        if torch.__version__.startswith('0.4'):
            with torch.no_grad():
                if self.opt.netG == 'global':
                    fake_image = self.netG.forward(input_concat)
                elif self.opt.netG == 'stack':
                    fake_image0,fake_image = self.netG.forward(input_concat)
        else:
            if self.opt.netG == 'global':
                fake_image = self.netG.forward(input_concat)
            elif self.opt.netG == 'stack':
                fake_image0,fake_image = self.netG.forward(input_concat)
        return fake_image

    def sample_features(self, inst): 
        # read precomputed feature clusters 
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)        
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)                                      
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])
        for i in np.unique(inst_np):    
            label = i if i < 1000 else i//1000
            if label in features_clustered:
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0]) 
                                            
                idx = (inst == int(i)).nonzero()
                for k in range(self.opt.feat_num):                                    
                    feat_map[idx[:,0], idx[:,1] + k, idx[:,2], idx[:,3]] = feat[cluster_idx, k]
        if self.opt.data_type==16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):
        image = Variable(image.cuda(), volatile=True)
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda())
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num+1))
        for i in np.unique(inst_np):
            label = i if i < 1000 else i//1000
            idx = (inst == int(i)).nonzero()
            num = idx.size()[0]
            idx = idx[num//2,:]
            val = np.zeros((1, feat_num+1))                        
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]            
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)
        return feature

    def get_edges(self, t):
        edge = torch.cuda.ByteTensor(t.size()).zero_()
        edge[:,:,:,1:] = edge[:,:,:,1:] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,:,:-1] = edge[:,:,:,:-1] | (t[:,:,:,1:] != t[:,:,:,:-1])
        edge[:,:,1:,:] = edge[:,:,1:,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        edge[:,:,:-1,:] = edge[:,:,:-1,:] | (t[:,:,1:,:] != t[:,:,:-1,:])
        if self.opt.data_type==16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())           
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd        
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst)

        
