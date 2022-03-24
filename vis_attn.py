import torch
import torch.nn as nn
import os 
import matplotlib.pyplot as plt
from IPython import embed
import cv2
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
from PIL import Image
import numpy as np
import pickle5 as pickle
from transformers import RobertaModel, RobertaTokenizerFast


img_size_pairs = torch.load('/DATA2/cxx/mdetr/img_size_pairs.pth')
img_attn_pairs = torch.load('/DATA2/cxx/mdetr/arm_attn_pairs.pth')
img_token_pairs = torch.load('/DATA2/cxx/mdetr/img_token_pairs.pth')
text_attn_pairs = torch.load('/DATA2/cxx/mdetr/text_attn_pairs.pth')

# def merge_img(jpg_img, png_img):
  
#     alpha_png = [0,0,255]
#     alpha_jpg = 0.9
    
#     for c in range(0,3):
#         jpg_img[:,:, c] = ((alpha_jpg*jpg_img[:,:,c]) + (alpha_png[c]*png_img))
 
#     return jpg_img



def add_alpha_channel(img,alpha):
    b_channel, g_channel, r_channel = cv2.split(img)
    alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * alpha 
    img_new = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_new

def merge_img(jpg_img, png_img):

    yy1 = 0
    yy2 = png_img.shape[0]
    xx1 = 0
    xx2 = png_img.shape[1]
  
    alpha_png = png_img[:,:,3] / 255.0
    alpha_jpg = 1 - alpha_png
    
    for c in range(0,3):
        jpg_img[:,:, c] = ((alpha_jpg*jpg_img[:,:,c]) + (alpha_png*png_img[:,:,c]))
 
    return jpg_img


def mask_color(mask,green=False):
    #mask =255-mask
    if green is False:
        mask[:,:,0]=0
    else:
        mask[:,:,0]=0
        mask[:,:,2]=0
    return mask


def ori_mask_color(mask,green=False):
    mask = mask * 255
    #mask =255-mask
    if green is False:
        mask[:,:,0]=0
    else:
        mask[:,:,0]=0
        mask[:,:,2]=0
    return mask


def imgwithmask(image_ori,mask_ori):
    img = add_alpha_channel(image_ori,255)
    mask = mask_ori #mask_color(mask_ori)
    mask = add_alpha_channel(mask,90)
    new_img = merge_img(img,mask)
    return new_img

#tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")
            

caption_attn_pairs = torch.load('caption_attn_pairs.pth')

caption_pairs = torch.load('caption_pairs.pth')

def get_text_attns(name):
    return caption_attn_pairs[name],caption_pairs[name]

embed()

"""
for img_name in text_attn_pairs.keys():
    print(img_name)
    attn = text_attn_pairs[img_name]
    pickle_path = '/DATA2/cxx/mdetr/yourefit/pickle/'+img_name+'.p'
    pickle_file = pickle.load(open(pickle_path, "rb" ))
    caption = pickle_file['anno_sentence'].lower()
    i = img_token_pairs[img_name]
    tokenized = tokenizer.batch_encode_plus([caption], padding="longest", return_tensors="pt")
    token_word_list = caption.split(' ')
    tokenized_list = []  
    for i in range(len(token_word_list)-1):
        token_word_list[i+1] = " "+token_word_list[i+1]

    for i in range(len(token_word_list)):
        tokenized_list.append(tokenizer(token_word_list[i])['input_ids'][1:-1])

    attentions = torch.tensor(attn)
    attentions = attentions[i][1:-1]

    cur_idx = 0
    token_word_attn = {}
    for i in range(len(token_word_list)):
        token_len = len(tokenized_list[i]) 
        attn_sum = attentions[cur_idx:cur_idx + token_len].sum()
        token_word_attn[token_word_list[i]] = float(attn_sum.cpu().numpy())
        cur_idx = cur_idx + token_len
    caption_attn_pairs[img_name] = token_word_attn

    # attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=32, mode="bilinear")[0].cpu().numpy()
    # folder = '/DATA2/cxx/mdetr/vis/attn/'+img_name
    # if not os.path.exists(folder):
    #     os.makedirs(folder)

    # fname =  os.path.join(folder ,str(i)+"text_attn.png")
    # plt.imsave(fname=fname, arr=attentions[i], format='png')
    # print(f"{fname} saved.")



# for img_name in img_size_pairs.keys():
#     size = img_size_pairs[img_name]
#     attn = img_attn_pairs[img_name]
#     img_path = '/DATA2/cxx/mdetr/yourefit/images/'+img_name+'.jpg'

#     attentions = attn.reshape(10,size[0], size[1])

#     attentions = torch.tensor(attentions)
#     attentions = nn.functional.interpolate(attentions.unsqueeze(0), scale_factor=32, mode="bilinear")[0].cpu().numpy()
#     folder = '/DATA2/cxx/mdetr/vis/attn/'+img_name
#     if not os.path.exists(folder):
#         os.makedirs(folder)

#     img = cv2.imread(img_path) 
#     i = img_token_pairs[img_name].cpu().numpy()  
#     fname =  os.path.join(folder ,str(i)+"arm_attn.png")
#     # attn = cv2.imread(fname)
#     # new_img = imgwithmask(img,attn)
#     # new_name =  os.path.join(folder ,str(i)+"_merged.jpg")
#     # cv2.imwrite(new_name,new_img)


#     # attn = attentions[i]
#     # attn = cv2.resize(attn,(img.shape[1],img.shape[0]))

#     # # attn = torch.tensor(attn).unsqueeze(2).repeat(1,1,3)

#     # attn = attn / attn.max()

#     #embed()
#     # attn_png = attn * torch.tensor([0,0,255])

#     # attn_png = attn_png.numpy()

#     #merged_img = merge_img(img,attn)


#     plt.imsave(fname=fname, arr=attentions[i], format='png')


#     #plt.imsave(fname=fname, arr=attn, format='png',cmap='Reds')

#     # attn_png = plt.imread(fname)
#     # attn_png = np.resize(attn_png,(img.shape[0],img.shape[1],4))
#     # plt.imsave(fname=fname, arr=attn_png, format='png')


#     # merged_img = merge_img(img,attn_png)
#     # plt.imsave(fname=fname, arr=merged_img, format='png')
#     # embed()
#     # img = img + attentions

#     print(f"{fname} saved.")
"""