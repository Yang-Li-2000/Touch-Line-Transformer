from copy import deepcopy
import torch
from IPython import embed




def query_num(checkpoint,num):
    dir = './pretrained/'
    temp = checkpoint['model_ema']['query_embed.weight']
    temp = temp[:num,:]
    checkpoint['model_ema']['query_embed.weight'] = temp
    torch.save(checkpoint, dir+'{0}_query_model.pth'.format(num))

def del_param(checkpoint):
    #embed()
    #keys = deepcopy(checkpoint['model_ema'].keys())
    key_list = []
    for key in checkpoint['model_ema'].keys():
        key_list.append(key)
    for key in key_list:
        print(key)
        if "transformer.decoder.layers.3" in key:
            del checkpoint['model_ema'][key]
        elif "transformer.decoder.layers.4" in key:
            del checkpoint['model_ema'][key]
        elif "transformer.decoder.layers.5" in key:
            del checkpoint['model_ema'][key]
    torch.save(checkpoint, '/./pretrained/3_layer_20_query.pth')
    return

def del_text_param(checkpoint):
    #embed()
    #keys = deepcopy(checkpoint['model_ema'].keys())
    key_list = []
    for key in checkpoint['model_ema'].keys():
        key_list.append(key)
    for key in key_list:
        print(key)
        if "transformer.tokenizer" in key:
            del checkpoint['model_ema'][key]
        elif "transformer.text_encoder" in key:
            del checkpoint['model_ema'][key]
        elif "transformer.resizer" in key:
            del checkpoint['model_ema'][key]
        elif "contrastive_align_projection" in key:
            del checkpoint['model_ema'][key]
    torch.save(checkpoint, './pretrained/no_text_20_query.pth')
    return

checkpoint = torch.load('./pretrained/20_query_model.pth',map_location="cpu")

del_text_param(checkpoint)