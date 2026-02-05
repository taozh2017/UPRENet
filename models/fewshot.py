"""
FSS via UPRE-Net
Extended from ADNet code by Hansen et al.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from models.encoder import Res101Encoder
import numpy as np
import random
import cv2
from models.moudles import  Decoder, RDF
from models import image_enhance
import matplotlib.pyplot as plt
from pdb import set_trace
from models.loss import weit_loss


class FewShotSeg(nn.Module):

    def __init__(self, pretrained_weights="deeplabv3"):
        super().__init__()

        # Encoder
        self.encoder = Res101Encoder(replace_stride_with_dilation=[True, True, False],
                                     pretrained_weights=pretrained_weights)  # or "resnet101"
        self.device = torch.device('cuda')
        self.scaler = 20.0
        self.weight = torch.tensor([0.1, 1.0])        
        self.criterion_MSE = nn.MSELoss()
        self.t_h = Parameter(torch.Tensor([-10.0]))
        self.decoder1 = Decoder(100)
        self.decoder2 = Decoder(500)
        self.stride = 8
        self.kernel = (8, 8)
        self.RDF1 = RDF(2)
        self.RDF2 = RDF(2)
        self.num_classes = 2
        self.softplus = nn.Softplus()
        self.softplus_en = nn.Softplus()
        self.thresh_uncertain = 0.8

    def forward(self, supp_imgs, supp_mask, supp_imgs_en, supp_mask_en, qry_imgs, train=False):
        """
        Args:
            supp_imgs: support images
                way x shot x [B x 3 x H x W], list of lists of tensors  1,1,1,3,256,256
            fore_mask: foreground masks for support images
                way x shot x [B x H x W], list of lists of tensors      1,1,1,256,256
            back_mask: background masks for support images
                way x shot x [B x H x W], list of lists of tensors      
            qry_imgs: query images
                N x [B x 3 x H x W], list of tensors                    1,1,3,256,256
        """
        self.n_ways = len(supp_imgs) 
        self.n_shots = len(supp_imgs[0])
        self.n_queries = len(qry_imgs)
        assert self.n_ways == 1  # for now only one-way, because not every shot has multiple sub-images
        assert self.n_queries == 1

        qry_bs = qry_imgs[0].shape[0]    
        supp_bs = supp_imgs[0][0].shape[0]      
        img_size = supp_imgs[0][0].shape[-2:]   
        
        supp_mask = torch.stack([torch.stack(way, dim=0) for way in supp_mask],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)  # B x Wa x Sh x H x W 
        supp_mask_en = torch.stack([torch.stack(way, dim=0) for way in supp_mask_en],
                                dim=0).view(supp_bs, self.n_ways, self.n_shots, *img_size)

        ###### Extract features ######
        imgs_concat = torch.cat([torch.cat(way, dim=0) for way in supp_imgs] + [torch.cat(way, dim=0) for way in supp_imgs_en]
                                + [torch.cat(qry_imgs, dim=0)] , dim=0)
        # encoder output
        img_fts, tao = self.encoder(imgs_concat)
        supp_fts = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view( 
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]    
        supp_fts = supp_fts[0]

        supp_fts_en = [img_fts[dic][self.n_ways * self.n_shots * supp_bs: 2 * self.n_ways * self.n_shots * supp_bs].view(
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]
        supp_fts_en = supp_fts_en[0]
        
        qry_fts = [img_fts[dic][2 * self.n_ways * self.n_shots * supp_bs: ].view(  # B x N x C x H' x W'
            qry_bs, self.n_queries, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)] 
        qry_fts = qry_fts[0]

        ##### Get threshold #######
        self.t_s = tao[:self.n_ways * self.n_shots * supp_bs]
        self.t_se = tao[self.n_ways * self.n_shots * supp_bs : 2 * self.n_ways * self.n_shots * supp_bs]
        self.t_q = tao[2 * self.n_ways * self.n_shots * supp_bs: 3 * self.n_ways * self.n_shots * supp_bs] 
       
        self.thresh_pred_s = [self.t_s for _ in range(self.n_ways)]     # support_image thresh  
        self.thresh_pred_se = [self.t_se for _ in range(self.n_ways)]   # support_enhance_image thresh
        self.thresh_pred_q = [self.t_q for _ in range(self.n_ways)]     # query_image thresh
        

        ###### Compute loss ######
        align_loss = torch.zeros(1).to(self.device)
        con_loss = torch.zeros(1).to(self.device)
        outputs = []
        for epi in range(supp_bs):
            ###### Extract prototypes ######
            ###### GLobal Prototypes ######
            supp_fts_ = [[F.interpolate(supp_fts[[epi], way, shot], size=img_size, mode='bilinear', align_corners=True)
                        for shot in range(self.n_shots)] for way in range(self.n_ways)]
            supp_fts_en_ = [[F.interpolate(supp_fts_en[[epi], way, shot], size=img_size, mode='bilinear', align_corners=True)
                        for shot in range(self.n_shots)] for way in range(self.n_ways)] 
            
            sup_proto_global = [[self.getFeatures(supp_fts_[way][shot], supp_mask[[epi], way, shot])
                                 for shot in range(self.n_shots)] for way in range(self.n_ways)] 
            sup_proto_global_en = [[self.getFeatures(supp_fts_en_[way][shot], supp_mask_en[[epi], way, shot])
                                 for shot in range(self.n_shots)] for way in range(self.n_ways)] 
            
            sup_proto_global_ = self.get_all_prototypes(sup_proto_global)  
            sup_proto_global_en_ = self.get_all_prototypes(sup_proto_global_en) 
            
            fg_pred_en = torch.stack([self.getPred(supp_fts_en[[epi], way, shot], sup_proto_global_[way], self.t_se)
                        for way in range(self.n_ways) for shot in range(self.n_shots)], dim=1)                      
            fg_pred = torch.stack([self.getPred(supp_fts[[epi], way, shot], sup_proto_global_en_[way], self.t_s)
                        for way in range(self.n_ways) for shot in range(self.n_shots)], dim=1)
            
            fg_pred_ = F.interpolate(fg_pred, size=img_size, mode='bilinear', align_corners=True) 
            fg_pred_en_ = F.interpolate(fg_pred_en, size=img_size, mode='bilinear', align_corners=True) 
            
            uncertainty_map = self.calculate_entropy(fg_pred_) 
            uncertainty_map_en = self.calculate_entropy(fg_pred_en_) 
            
            sup_proto = [self.UPG(supp_fts_[way][shot], supp_mask[epi][way], uncertainty_map, kernel_size=self.kernel, stride = self.stride) for way in range(self.n_ways) for shot in range(self.n_shots)]
            sup_proto_en = [self.UPG(supp_fts_en_[way][shot], supp_mask_en[epi][way], uncertainty_map_en, kernel_size=self.kernel, stride=self.stride) for way in range(self.n_ways) for shot in range(self.n_shots)]
            
            fg_proto = [sup_proto[0][0]]
            bg_proto = [sup_proto[0][1]]
            
            fg_proto_en = [sup_proto_en[0][0]]
            bg_proto_en = [sup_proto_en[0][1]]
            
            con_loss += self.criterion_MSE(fg_proto[0], fg_proto_en[0])

            fg_predition = torch.stack(
                [self.get_fg_sim(qry_fts[epi], fg_proto[way]) for way in range(self.n_ways)]).squeeze(0)
            bg_predition = torch.stack(
                [self.get_bg_sim(qry_fts[epi], bg_proto[way]) for way in range(self.n_ways)]).squeeze(0)
            fg_predition_en = torch.stack(
                [self.get_fg_sim(qry_fts[epi], fg_proto_en[way]) for way in range(self.n_ways)]).squeeze(0)
            bg_predition_en = torch.stack(
                [self.get_bg_sim(qry_fts[epi], bg_proto_en[way]) for way in range(self.n_ways)]).squeeze(0)
            
                
            fg_predition = F.interpolate(fg_predition, size=img_size, mode='bilinear', align_corners=True)
            bg_predition = F.interpolate(bg_predition, size=img_size, mode='bilinear', align_corners=True)
            fg_predition_en = F.interpolate(fg_predition_en, size=img_size, mode='bilinear', align_corners=True)
            bg_predition_en = F.interpolate(bg_predition_en, size=img_size, mode='bilinear', align_corners=True)
                
            predition_1 = torch.cat([bg_predition, fg_predition], dim=1)
            predition_2 = torch.cat([bg_predition_en, fg_predition_en], dim=1)
            predition_1_plus = self.softplus(predition_1)
            predition_2_plus = self.softplus_en(predition_2)

            evidence = dict()
            evidence[0] = predition_1_plus.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
            evidence[1] = predition_2_plus.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                
            alpha = dict()
            for v_num in range(len(evidence)):
                alpha[v_num] = evidence[v_num] + 1
                
            alpha_a, u = self.DS_Combin(alpha)
            predition = self.RDF1(predition_1, predition_2, u)
            
            predition = torch.softmax(predition, dim=1)
            
            outputs.append(predition)
            
           
            if train:
                align_loss_epi, con_loss_epi = self.align_aux_Loss([supp_fts[epi]], [qry_fts[epi]], qry_imgs, predition, 
                                                     supp_mask[epi])
                align_loss += align_loss_epi
                con_loss += con_loss_epi


        output = torch.stack(outputs, dim=1) 
        output = output.view(-1, *output.shape[2:]) 

        return output, align_loss / supp_bs, con_loss / supp_bs
        # return 0, 0, 0
    
    def align_aux_Loss(self, supp_fts, qry_fts, qry_imgs, qry_pred, sup_mask):
        """
            supp_fts: [1, 512, 64, 64]          1,1,1,512,64,64
            qry_fts: (1, 512, 64, 64)           1,1,512,64,64
            pred: [1, 2, 256, 256]              1,2,256,256     query_prediction
            fore_mask: [Way, Shot , 256, 256]   1,1,256,256     support_mask
            thresh:                             1,1,1
        """
        loss=[]
        n_ways, n_shots = len(sup_mask), len(sup_mask[0])
        
        img_size = sup_mask.shape[-2:]

        # Get query mask
        pred_mask = qry_pred.argmax(dim=1, keepdim=True).squeeze(1) 
        qry_imgs_en, pred_mask_en = image_enhance.Query_Enhance([qry_imgs], [[pred_mask]])
       
        
        pred_mask_en = pred_mask_en[0][0] 
        
        supp_bs = qry_imgs[0].shape[0]
        img_fts, tao = self.encoder(qry_imgs_en[0][0])
        qry_fts_en = [img_fts[dic][:self.n_ways * self.n_shots * supp_bs].view( 
            supp_bs, self.n_ways, self.n_shots, -1, *img_fts[dic].shape[-2:]) for _, dic in enumerate(img_fts)]     
        qry_fts_en = qry_fts_en[0]
        self.t_qe = tao[:self.n_ways * self.n_shots * supp_bs]
        self.thresh_pred_qe = [self.t_qe for _ in range(self.n_ways)]
        
        binary_masks = [pred_mask == i for i in range(1 + n_ways)]  
        binary_masks_en = [pred_mask_en == i for i in range(1 + n_ways)] 
        
        skip_ways = [i for i in range(n_ways) if binary_masks[i + 1].sum() == 0]  
        
        pred_mask = torch.stack(binary_masks, dim=0).float()
        pred_mask_en = torch.stack(binary_masks_en, dim=0).float()  
        qry_fts = [qry_fts]
        loss = torch.zeros(1).to(self.device)
        con_loss = torch.zeros(1).to(self.device)
        for way in range(n_ways):
            if way in skip_ways:
                continue
            # Get the query prototypes
            for shot in range(n_shots):
                
                qry_fts_ = [[F.interpolate(qry_fts[way][shot], size=img_size, mode='bilinear', align_corners=True)]]       
                qry_fts_en_ = [[F.interpolate(qry_fts_en[way][shot], size=img_size, mode='bilinear', align_corners=True)]] 
            
                qry_proto_global = [[self.getFeatures(qry_fts_[way][shot], pred_mask[way+1])]] 
                qry_proto_global_en = [[self.getFeatures(qry_fts_en_[way][shot], pred_mask_en[way+1])]]
            
                qry_proto_global_ = self.get_all_prototypes(qry_proto_global) 
                qry_proto_global_en_ = self.get_all_prototypes(qry_proto_global_en) 
            
                fg_pred_en = torch.stack([self.getPred(qry_fts_en[way][shot], qry_proto_global_[way], self.t_qe)], dim=1) 
                fg_pred = torch.stack([self.getPred(qry_fts[way][shot], qry_proto_global_en_[way], self.t_q)], dim=1) 
            
                fg_pred_ = F.interpolate(fg_pred, size=img_size, mode='bilinear', align_corners=True) 
                fg_pred_en_ = F.interpolate(fg_pred_en, size=img_size, mode='bilinear', align_corners=True)

            
                uncertainty_map = self.calculate_entropy(fg_pred_)                 
                uncertainty_map_en = self.calculate_entropy(fg_pred_en_)           
                 
                qry_proto = [self.UPG(qry_fts_[way][shot], pred_mask[way+1], uncertainty_map, kernel_size=self.kernel, stride = self.stride)]
                qry_proto_en = [self.UPG(qry_fts_en_[way][shot], pred_mask_en[way+1], uncertainty_map_en, kernel_size=self.kernel, stride = self.stride)] 
                
                fg_proto = [qry_proto[0][0]]
                bg_proto = [qry_proto[0][1]]
                
                fg_proto_en = [qry_proto_en[0][0]]
                bg_proto_en = [qry_proto_en[0][1]]
                
                con_loss += self.criterion_MSE(fg_proto[0], fg_proto_en[0])
                
                fg_predition = torch.stack(
                    [self.get_fg_sim(supp_fts[way][shot], fg_proto[way])]).squeeze(0)
                bg_predition = torch.stack(
                    [self.get_bg_sim(supp_fts[way][shot], bg_proto[way])]).squeeze(0)
                fg_predition_en = torch.stack(
                    [self.get_fg_sim(supp_fts[way][shot], fg_proto_en[way])]).squeeze(0)
                bg_predition_en = torch.stack(
                    [self.get_bg_sim(supp_fts[way][shot], bg_proto_en[way])]).squeeze(0)
                    
                fg_predition = F.interpolate(fg_predition, size=img_size, mode='bilinear', align_corners=True)
                bg_predition = F.interpolate(bg_predition, size=img_size, mode='bilinear', align_corners=True)
                fg_predition_en = F.interpolate(fg_predition_en, size=img_size, mode='bilinear', align_corners=True)
                bg_predition_en = F.interpolate(bg_predition_en, size=img_size, mode='bilinear', align_corners=True)
                    
                predition_1 = torch.cat([bg_predition, fg_predition], dim=1)
                predition_2 = torch.cat([bg_predition_en, fg_predition_en], dim=1)
                predition_1_plus = self.softplus(predition_1)
                predition_2_plus = self.softplus_en(predition_2)
  
                
                evidence = dict()
                evidence[0] = predition_1_plus.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                evidence[1] = predition_2_plus.permute(0, 2, 3, 1).reshape(-1, self.num_classes)
                
                alpha = dict()
                for v_num in range(len(evidence)):
                    alpha[v_num] = evidence[v_num] + 1
                
                alpha, u = self.DS_Combin(alpha)
                
                predition = self.RDF2(predition_1, predition_2, u)
                predition = torch.softmax(predition, dim=1)
                
                supp_label = torch.full_like(sup_mask[way, shot], 255, device=sup_mask.device)  
                supp_label[sup_mask[way, shot] == 1] = 1
                supp_label[sup_mask[way, shot] == 0] = 0

                query_uncertainty_map = -1 * torch.sum(predition * torch.log(predition + 1e-6), dim=1, keepdim=True)
                loss = weit_loss(predition, supp_label[None, ...].long(), query_uncertainty_map.squeeze(0))
                
        return loss, con_loss

    def get_mask(self, s_mask, s_mask_en, mask, mask_en):
        
        s_mask_ = torch.cat([s_mask, 1-s_mask], dim=1)
        s_mask_en_ = torch.cat([s_mask_en, 1-s_mask_en], dim=1)
        
        s_mask_ = torch.softmax(s_mask_, dim=1)
        s_mask_en_ = torch.softmax(s_mask_en_, dim=1)
        
        s_mask = torch.argmax(s_mask_, dim=1, keepdim=True)
        s_mask_en = torch.argmax(s_mask_en_, dim=1, keepdim=True)
        
        return mask

    def getPred(self, fts, prototype, thresh):
        """
        Calculate the distance between features and prototypes
        # (1, 512, 64, 64) (1, 512), (1, 1)
        Args:
            fts: input features
                expect shape: N x C x H x W     1, 1, 512, 64, 64
            prototype: prototype of one semantic class
                expect shape: 1 x C             1, 512
        """

        sim = -F.cosine_similarity(fts, prototype[..., None, None], dim=1) * self.scaler   
        pred = 1.0 - torch.sigmoid(0.5 * (sim - thresh))  

        return pred

    def getFeatures(self, fts, mask):
        """
        Extract foreground and background features via masked average pooling
        Args:
            fts: input features, expect shape: 1 x C x H' x W'
            mask: binary mask, expect shape: 1 x H x W
        """
        masked_fts = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                     / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)  # 1 x C

        return masked_fts

    def getPrototype(self, fg_fts):
        """
        Average the features to obtain the prototype
        Args:
            fg_fts: lists of list of foreground features for each way/shot
                expect shape: Wa x Sh x [1 x C] (1, 1, (1, 512))
            bg_fts: lists of list of background features for each way/shot
                expect shape: Wa x Sh x [1 x C]
        """
        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  # concat all fg_fts   (n_way, (1, 512))


        return fg_prototypes
    
        
    def calculate_entropy(self, pred_sup):
        """
            calculate_entropy
            pred_sup: 1, 1, 256, 256
        """
        pred_sup = torch.cat((1.0 - pred_sup, pred_sup), dim = 1)
        
        ent_map = -1 * torch.sum(pred_sup * torch.log(pred_sup + 1e-6), dim=1, keepdim=True)        
        
        return ent_map
    
      
    def get_pred_selfnet(self, fts, prototypes):
        """
            fts: 1, 512, 64, 64
            prototypes: num, 512
        """
        res=[]
        
        def safe_norm(x, p = 2, dim = 1, eps = 1e-4):
            x_norm = torch.norm(x, p = p, dim = dim) # .detach()
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x
        
        fts_ = safe_norm(fts)
        prototypes_ = safe_norm(prototypes)
        
        dists = F.conv2d(fts_, prototypes_[..., None, None]) * self.scaler
        pred = torch.sum(F.softmax(dists, dim=1)*dists, dim=1, keepdim=True)
        
        return pred

    def UPG(self, fts, mask, uncertainty_map, kernel_size=(8,8), stride = 8):
        """
            fts:1, 512, 64, 64
            mask:1,256,256
            uncertainty_map: 1,1,256,256
        """
        # img_size = mask.shape[-2:]
        # fts = F.interpolate(fts, size=img_size, mode='bilinear', align_corners=True)
        b, c, h, w = fts.size()
        kernel_h, kernel_w = kernel_size
        n_fg = mask.sum() 
        n_bg = h * w - n_fg
        fore_fts = []
        back_fts = []

        bg_mask = 1 - mask
        
        fg_prototype = torch.sum(fts * mask[None, ...], dim=(-2, -1)) \
                        / (mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        bg_prototype = torch.sum(fts * (bg_mask[None, ...]), dim=(-2, -1)) \
                        / (bg_mask[None, ...].sum(dim=(-2, -1)) + 1e-5)
        
        fore_fts.append(fg_prototype)
        back_fts.append(bg_prototype)
        

        fts_patch = F.unfold(fts, kernel_size=kernel_size, stride=stride).permute(0, 2, 1).contiguous().view(1, -1, c, kernel_h, kernel_w)
        fg_mask_patch = F.unfold(mask.unsqueeze(1), kernel_size=kernel_size, stride=stride).permute(0, 2, 1).contiguous().view(1, -1, kernel_h, kernel_w).unsqueeze(2)
        uncertainty_patch = F.unfold(uncertainty_map, kernel_size=kernel_size, stride=stride).permute(0, 2, 1).contiguous().view(1, -1, kernel_h, kernel_w).unsqueeze(2)
        bg_mask_patch = 1 - fg_mask_patch
        uncertainty_patch = torch.mean(uncertainty_patch, dim=(-2, -1)).squeeze(0).squeeze(-1)
        
        fg_mask_patch_index = torch.sum(fg_mask_patch, dim=(-2, -1)).view(-1) >= min(kernel_h * kernel_w * 0.5 , n_fg/100)
        bg_mask_patch_index = torch.sum(bg_mask_patch, dim=(-2, -1)).view(-1) >= min(kernel_h * kernel_w * 0.5 , n_bg/300)
        
        
        #extract fg_protos
        random.seed(9)
        if torch.sum(fg_mask_patch_index == 1) >= 10:
            fg_un_index = torch.masked_select(uncertainty_patch * fg_mask_patch_index, uncertainty_patch * fg_mask_patch_index!=0)
            fg_un_index_sort = torch.sort(fg_un_index)[0]
            fg_num = np.ceil(fg_un_index_sort.shape[-1] * self.thresh_uncertain)
            # fg_num = fg_un_index_sort.shape[-1]
            
            fg_un_thresh = fg_un_index_sort[int(fg_num - 1)]
            fg_certainty_patch = (uncertainty_patch <= fg_un_thresh) * fg_mask_patch_index
            
            fg_protos = torch.sum(fts_patch[:, fg_certainty_patch, :, :, :] * fg_mask_patch[:, fg_certainty_patch, :, :, :], dim=(-2, -1)) \
                 / (fg_mask_patch[:, fg_certainty_patch, :, :, :].sum(dim=(-2, -1)) + 1e-5)
            fg_protos = fg_protos.squeeze(0)
            
            if len(fg_protos) > 100:
                fg_protos = fg_protos[random.sample(range(len(fg_protos)), 100)]
            elif len(fg_protos) < 100:
                n = int(np.ceil(100.0/len(fg_protos)))
                fg_protos = fg_protos.repeat(n, 1)
                fg_protos = fg_protos[random.sample(range(len(fg_protos)), 100)]
            
        elif torch.sum(fg_mask_patch_index == 1) == 0:
            fg_protos = self.get_pts(fts, mask)
            if len(fg_protos) < 100:
                n = int(np.ceil(100.0/len(fg_protos)))
                fg_protos = fg_protos.repeat(n, 1)
                fg_protos = fg_protos[random.sample(range(len(fg_protos)), 100)]
                
        else:
            fg_protos = torch.sum(fts_patch[:, fg_mask_patch_index, :, :, :] * fg_mask_patch[:, fg_mask_patch_index, :, :, :], dim=(-2, -1)) \
                / (fg_mask_patch[:, fg_mask_patch_index, :, :, :].sum(dim=(-2, -1)) + 1e-5)
            fg_protos = fg_protos.squeeze(0)
            n = int(np.ceil(100.0/len(fg_protos)))
            fg_protos = fg_protos.repeat(n, 1)
            fg_protos = fg_protos[random.sample(range(len(fg_protos)), 100)]
        
        fore_fts.append(fg_protos)    
        
        
        if torch.sum(bg_mask_patch_index == 1) > 500:
            bg_un_index = torch.masked_select(uncertainty_patch * bg_mask_patch_index, uncertainty_patch * bg_mask_patch_index!=0)
            bg_un_index_sort = torch.sort(bg_un_index)[0]
            bg_num = np.ceil(bg_un_index_sort.shape[-1] * self.thresh_uncertain)
            # bg_num = bg_un_index_sort.shape[-1]
            
            bg_un_thresh = bg_un_index_sort[int(bg_num - 1)]
            bg_certainty_patch = (uncertainty_patch <= bg_un_thresh) * bg_mask_patch_index
            bg_protos = torch.sum(fts_patch[:, bg_certainty_patch, :, :, :] * bg_mask_patch[:, bg_certainty_patch, :, :, :], dim=(-2, -1)) \
                 / (bg_mask_patch[:, bg_certainty_patch, :, :, :].sum(dim=(-2, -1)) + 1e-5)
            bg_protos = bg_protos.squeeze(0)
            
            if len(bg_protos) > 500:
                bg_protos = bg_protos[random.sample(range(len(bg_protos)), 500)]
            elif len(bg_protos < 500):
                n = int(np.ceil(500.0/len(bg_protos)))
                bg_protos = bg_protos.repeat(n, 1)
                bg_protos = bg_protos[random.sample(range(len(bg_protos)), 500)]
                
        elif torch.sum(bg_mask_patch_index == 1) == 0:
            bg_protos = self.get_pts(fts, bg_mask)
        else:
            bg_protos = torch.sum(fts_patch[:, bg_mask_patch_index, :, :, :] * bg_mask_patch[:, bg_mask_patch_index, :, :, :], dim=(-2, -1)) \
                / (bg_mask_patch[:, bg_mask_patch_index, :, :, :].sum(dim=(-2, -1)) + 1e-5)
            bg_protos = bg_protos.squeeze(0)
            n = int(np.ceil(500.0/len(bg_protos)))
            bg_protos = bg_protos.repeat(n, 1)
            bg_protos = bg_protos[random.sample(range(len(bg_protos)), 500)]
        
        back_fts.append(bg_protos)
     
        fg_protos = torch.cat(fore_fts, dim=0)
        bg_protos = torch.cat(back_fts, dim=0)
        
        # print(fg_protos.shape)
        # print(bg_protos.shape)
        
        return [fg_protos, bg_protos]
    
        
    def DS_Combin(self, alpha):
        """
        :param alpha: All Dirichlet distribution parameters.
        :return: Combined Dirichlet distribution parameters.
        """

        def DS_Combin_two(alpha1, alpha2):
            """
            :param alpha1: Dirichlet distribution parameters of view 1
            :param alpha2: Dirichlet distribution parameters of view 2
            :return: Combined Dirichlet distribution parameters
            """
            alpha = dict()
            alpha[0], alpha[1] = alpha1, alpha2
            b, S, E, u = dict(), dict(), dict(), dict()
            for v in range(2):
                S[v] = torch.sum(alpha[v], dim=1, keepdim=True)
                E[v] = alpha[v] - 1
                b[v] = E[v] / (S[v].expand(E[v].shape))
                u[v] = self.num_classes / S[v]

            # # b^0 @ b^(0+1)
            # bb = torch.bmm(b[0].view(-1, self.num_classes, 1), b[1].view(-1, 1, self.num_classes))
            # # b^0 * u^1
            # uv1_expand = u[1].expand(b[0].shape)
            # bu = torch.mul(b[0], uv1_expand)
            # # b^1 * u^0
            # uv_expand = u[0].expand(b[0].shape)
            # ub = torch.mul(b[1], uv_expand)
            # # calculate C
            # bb_sum = torch.sum(bb, dim=(1, 2), out=None)
            # bb_diag = torch.diagonal(bb, dim1=-2, dim2=-1).sum(-1)
            # # bb_diag1 = torch.diag(torch.mm(b[v], torch.transpose(b[v+1], 0, 1)))
            # C = bb_sum - bb_diag

            # # calculate b^a
            # b_a = (torch.mul(b[0], b[1]) + bu + ub) / ((1 - C).view(-1, 1).expand(b[0].shape))
            # # calculate u^a
            # u_a = torch.mul(u[0], u[1]) / ((1 - C).view(-1, 1).expand(u[0].shape))

            # # calculate new S
            # S_a = self.num_classes / u_a
            # # calculate new e_k
            # e_a = torch.mul(b_a, S_a.expand(b_a.shape))
            # alpha_a = e_a + 1
            return alpha, u

        for v in range(len(alpha) - 1):
            if v == 0:
                u = DS_Combin_two(alpha[0], alpha[1])
            else:
                alpha_a, u = DS_Combin_two(alpha_a, alpha[v + 1])
        return u
    
    def get_pts(self, fts, mask):

        features_trans = fts.squeeze(0)
        features_trans = features_trans.permute(1, 2, 0) 
        features_trans = features_trans.view(features_trans.shape[-2] * features_trans.shape[-3],
                                             features_trans.shape[-1])
        mask = mask.squeeze(0).view(-1)
        indx = mask == 1 
        features_trans = features_trans[indx]
        if len(features_trans) > 50: 
            random.seed(9)
            features_trans = features_trans[random.sample(range(len(features_trans)), 50)]
        return features_trans

    def get_fg_sim(self, fts, prototypes):
        """
        Calculate the distance between features and prototypes
        # (1, 512, 64, 64) (102, 512)
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        
        def safe_norm(x, p = 2, dim = 1, eps = 1e-4):
            x_norm = torch.norm(x, p = p, dim = dim) # .detach()
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x
        
        fts_ = safe_norm(fts)
        prototypes_ = safe_norm(prototypes)
        
        fg_sim = F.conv2d(fts_, prototypes_[..., None, None]) * self.scaler

        fg_sim = self.decoder1(fg_sim)
    
        return fg_sim   # [1, 1, 64, 64]

    def get_bg_sim(self, fts, prototypes):
        """
        Calculate the distance between features and prototypes
        # (1, 512, 64, 64) (102, 512)
        Args:
            fts: input features
                expect shape: N x C x H x W
            prototype: prototype of one semantic class
                expect shape: 1 x C
        """
        
        def safe_norm(x, p = 2, dim = 1, eps = 1e-4):
            x_norm = torch.norm(x, p = p, dim = dim) # .detach()
            x_norm = torch.max(x_norm, torch.ones_like(x_norm).cuda() * eps)
            x = x.div(x_norm.unsqueeze(1).expand_as(x))
            return x
        
        fts_ = safe_norm(fts)
        prototypes_ = safe_norm(prototypes)
        
        bg_sim = F.conv2d(fts_, prototypes_[..., None, None]) * self.scaler
        
        # sim = -F.cosine_similarity(fts, prototypes[..., None, None], dim=1) * self.scaler
        # bg_sim = 1.0 - torch.sigmoid(0.5 * (sim - thresh))
        # bg_sim.unsqueeze(0)
        bg_sim = self.decoder2(bg_sim)

        return bg_sim  # [1, 1, 64, 64]

    def get_all_prototypes(self, fg_fts):
        """
            fg_fts: lists of list of tensor
                        expect shape: Wa x Sh x [all x C]
            fg_prototypes: [(all, 512) * way]    list of tensor
        """

        n_ways, n_shots = len(fg_fts), len(fg_fts[0])
        fg_prototypes = [torch.sum(torch.cat([tr for tr in way], dim=0), dim=0, keepdim=True) / n_shots for way in
                         fg_fts]  ## concat all fg_fts   (n_way, (1, 512))  

        return fg_prototypes