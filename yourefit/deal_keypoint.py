import argparse
import json
import os
 
import cv2
import numpy as np
from IPython import embed
# 骨骼关键点连接对

ref_pairs = [
    [3,4],[6,7]
]

pose_pairs = [
    [0, 1], [0, 15], [0, 16],
    [15, 17],
    [16, 18],
    [1, 2], [1, 5], [1, 8],
    [2, 3],
    [3, 4],
    [5, 6],
    [6, 7],
    [8, 9], [8, 12],
    [9, 10],
    [10, 11],
    [11, 22], [11, 24],
    [22, 23],
    [12, 13],
    [13, 14],
    [14, 21], [14, 19],
    [19, 20]
]
 
# 手部关键点连接对
hand_pairs = [
    [0, 1], [0, 5], [0, 9], [0, 13], [0, 17],
    [1, 2],
    [2, 3],
    [3, 4],
    [5, 6], [6, 7], [7, 8],
    [9, 10], [10, 11], [11, 12],
    [13, 14], [14, 15], [15, 16],
    [17, 18], [18, 19], [19, 20]
]
 
# 绘制用的颜色
pose_colors = [
    (255., 0., 85.), (255., 0., 0.), (255., 85., 0.), (255., 170., 0.),
    (255., 255., 0.), (170., 255., 0.), (85., 255., 0.), (0., 255., 0.),
    (255., 0., 0.), (0., 255., 85.), (0., 255., 170.), (0., 255., 255.),
    (0., 170., 255.), (0., 85., 255.), (0., 0., 255.), (255., 0., 170.),
    (170., 0., 255.), (255., 0., 255.), (85., 0., 255.), (0., 0., 255.),
    (0., 0., 255.), (0., 0., 255.), (0., 255.,
                                     255.), (0., 255., 255.), (0., 255., 255.)
]
 
hand_colors = [
    (100., 100., 100.),
    (100, 0, 0),
    (150, 0, 0),
    (200, 0, 0), (255, 0, 0), (100, 100, 0), (150,
                                              150, 0), (200, 200, 0), (255, 255, 0),
    (0, 100, 50), (0, 150, 75), (0, 200, 100), (0,
                                                255, 125), (0, 50, 100), (0, 75, 150),
    (0, 100, 200), (0, 125, 255), (100, 0, 100), (150, 0, 150),
    (200, 0, 200), (255, 0, 255)
]
 
 
pts = {}
def handle_json(jsonfile):
 
    #print('handle json {}'.format(jsonfile))
 
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    
    name = jsonfile.split('/')[-1][:-15]
    #img = cv2.imread('/data/cxx/YouRefIt_ERU/data/yourefit/images/'+jsonfile.split('/')[2][:-15]+'.jpg')
    pts[name] = []
    if len(data['people'])==0:
        #print(name)
        return name
    for d in [data['people'][0]]:
        kpt = np.array(d['pose_keypoints_2d']).reshape((25, 3))
        for p in ref_pairs:
            pt1 = tuple(list(map(int, kpt[p[0], 0:2])))
            c1 = kpt[p[0], 2]
            pt2 = tuple(list(map(int, kpt[p[1], 0:2])))
            c2 = kpt[p[1], 2]
            #print('== {}, {}, {}, {} =='.format(pt1, c1, pt2, c2))
 
            if c1 == 0.0 or c2 == 0.0:
                continue
            pts[name].append([pt1,pt2])
    return None
            # color = tuple(list(map(int, pose_colors[p[0]])))
            # img = cv2.line(img, pt1, pt2, color, thickness=4)
            # img = cv2.circle(img, pt1, 4, color, thickness=-
            #                  1, lineType=8, shift=0)
            # img = cv2.circle(img, pt2, 4, color, thickness=-
            #                  1, lineType=8, shift=0)
        # kpt_left_hand = np.array(d['hand_left_keypoints_2d']).reshape((21, 3))
        # for q in hand_pairs:
        #     pt1 = tuple(list(map(int, kpt_left_hand[q[0], 0:2])))
        #     c1 = kpt_left_hand[p[0], 2]
        #     pt2 = tuple(list(map(int, kpt_left_hand[q[1], 0:2])))
        #     c2 = kpt_left_hand[q[1], 2]
 
        #     # print('** {}, {}, {}, {} **'.format(pt1, c1, pt2, c2))
 
        #     if c1 == 0.0 or c2 == 0.0:
        #         continue
 
        #     color = tuple(list(map(int, hand_colors[q[0]])))
        #     img = cv2.line(img, pt1, pt2, color, thickness=4)
 
        # kpt_right_hand = np.array(
        #     d['hand_right_keypoints_2d']).reshape((21, 3))
        # for k in hand_pairs:
        #     pt1 = tuple(list(map(int, kpt_right_hand[k[0], 0:2])))
        #     c1 = kpt_right_hand[k[0], 2]
        #     pt2 = tuple(list(map(int, kpt_right_hand[k[1], 0:2])))
        #     c2 = kpt_right_hand[k[1], 2]
 
        #     print('** {}, {}, {}, {} **'.format(pt1, c1, pt2, c2))
 
        #     if c1 == 0.0 or c2 == 0.0:
        #         continue
 
        #     color = tuple(list(map(int, hand_colors[q[0]])))
        #     img = cv2.line(img, pt1, pt2, color, thickness=4)
 
    # if not os.path.exists('keypoints'):
    #     os.makedirs('keypoints')
 
    # # 保存图片
    # cv2.imwrite('keypoints/'+jsonfile.split('/')[2][:-15]+'jpg',img)
 
if __name__ == '__main__':
 
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=str,
                        default='/data/hdd01/cxx/mdetr/yourefit/keypoint', help='keypoints json directory')
    opt = parser.parse_args()
    null_list = []
    for jsonfile in os.listdir(opt.directory):
        if jsonfile.endswith('.json'):
            name = handle_json(os.path.join(opt.directory, jsonfile))
            if name is not None:
                null_list.append(name)
    embed()
    # a = np.load('ref_pose.npy',allow_pickle=True)
    # a = a.item()