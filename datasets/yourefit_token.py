from distutils import file_util
import os
import re
import os.path as osp
from IPython import embed
import pickle5 as pickle

num = 0


check_list = {'the knige':[[8,13]], 
              'cellphone':[[10,14],[15,20]], 
              'the towles':[[4,10]],
              'picture frame':[[0,3]],
              'the jacket':[[4,10]],
              'scossors': [[18, 26]],
              'potato':[[4,10]],
              'box':[[12,18]],
              'chiar':[[15,20]],
              'the bad':[[8,11]],
              'purple':[[0,5]],
              'the char':[[10,15]],
              'the computer':[[5,8],[9,13]],
              'a bottle ':[[9,12]]
              }


def match_pos(sentence,word):
    sentence = sentence.lower()
    word = word.lower()
    arr = []
    split_words=word.split(' ')
    for w in split_words:
        if w == 'the':
            continue
        if len(w) == 0:
            continue
        rr = re.compile(w, re.I) 
        for match in re.finditer(rr, sentence):
            arr.append([match.start(),match.end()])
    if len(arr) == 0:
        if word == 'shoes':
            match = re.search('shoe',sentence)
            arr.append([match.start(),match.end()])  
        elif word =='box':
            if sentence == 'q-tip':
                arr.append([0,6])
            else:
                arr.append([12,18])
        elif word in check_list.keys():
            arr = check_list[word]
        else:
            exit(0)
        #arr = check_list[word]
    return arr


file_list = os.listdir('/DATA2/cxx/mdetr/yourefit/pickle')
for file in file_list:
    pickle_file = '/DATA2/cxx/mdetr/yourefit/pickle/'+file
    pick = pickle.load(open(pickle_file, "rb" ))
    #embed()
    bbox = pick['bbox']
    target_word = pick['anno_target']
    phrase = pick['anno_sentence']
    token_pos = match_pos(phrase,target_word)
    if len(token_pos)==0:
        num = num+1
        print(num)
    token_pos = [token_pos]