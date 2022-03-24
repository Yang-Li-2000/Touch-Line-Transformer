import json
import os
import pickle
import os.path as osp
import cv2
from IPython import embed
import copy

import torch.nn.functional as F
import torch

with open("arms.json","r") as f:
    arm_data = json.load(f)



def draw_box(img,box,img_name,arm=None,gt_box=None,box1=None):
    pt1 = (int(box[0]),int(box[1]))
    pt2 = (int(box[2]),int(box[3]))
    cv2.rectangle(img,pt1,pt2,(0,255,0),4) # green
    if arm is not None:
        pt3 = (int(arm[0][0]),int(arm[0][1]))
        pt4 = (int(arm[1][0]),int(arm[1][1]))
        cv2.line(img,pt3,pt4,(0,0,255),5) #red
    if gt_box is not None:
        pt5 = (int(gt_box[0]),int(gt_box[1]))
        pt6 = (int(gt_box[2]),int(gt_box[3]))
        cv2.rectangle(img,pt5,pt6,(255,0,0),4) #blue
    if box1 is not None: # second best box
        pt7 = (int(box1[0]),int(box1[1]))
        pt8 = (int(box1[2]),int(box1[3]))
        cv2.rectangle(img,pt7,pt8,(255,255,0),4) #light blue
    cv2.imwrite(img_name,img)



bbox_data = {}

i=0



for img_name in os.listdir('./yourefit/images'):
    pickle_file = osp.join('./yourefit/pickle',img_name[:-4]+'.p')
    pick = pickle.load(open(pickle_file, "rb" ))
    bbox_data[img_name[:-4]] = pick['bbox']


#embed()
cos_sim_list=[]
for name in arm_data.keys():
    arm = arm_data[name]
    box = bbox_data[name]
    img_file = './yourefit/images/'+name+'.jpg'
    img = cv2.imread(img_file)
    print(name)
    draw_box(img,box,'./vis/gt_vis/'+name+'.jpg',arm)


    #embed()
    # arm_tensor = torch.Tensor(arm[1])-torch.Tensor(arm[0])
    # box_center = [(box[0]+box[2])/2,(box[1]+box[3])/2]
    # box_tensor = torch.Tensor(box_center)-torch.Tensor(arm[0])
    # cos_sim = F.cosine_similarity(arm_tensor,box_tensor,dim=0)
    # cos_sim_list.append(cos_sim)


