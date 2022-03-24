import argparse
import json
import os
import numpy as np
from IPython import embed

def deal_arm_json(jsonfile):
    with open(jsonfile, 'r') as f:
        data = json.load(f)
    arm_line = data['shapes'][0]['points']
    assert len(arm_line)==2
    return arm_line


dir = './yourefit/arm'


arms= {}


for jsonfile in os.listdir(dir):
    if jsonfile.endswith('.json'):
        arms[jsonfile[:-5]] = deal_arm_json(os.path.join(dir,jsonfile))

with open("arms.json","w") as f:
    json.dump(arms,f)


with open("arms.json","r") as f:
    arm_data = json.load(f)
    
embed()