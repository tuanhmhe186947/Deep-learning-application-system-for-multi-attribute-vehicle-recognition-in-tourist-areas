import json
from ..config import ROOT
import os


def read_model_config_file(a):
    setup = json.load(open(os.path.join(ROOT, "resources/data/setup.json")))
    track = setup[a]
    weight = os.path.join(ROOT, track[0]['weight'])
    classes = track[0]['classes']
    imgz = track[0]['imgz']
    conf = track[0]['conf']
    device = track[0]['device'] 
    data = os.path.join(ROOT, track[0]['data'])
    print(data)
    return weight, classes, conf, imgz, device, data
