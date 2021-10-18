import argparse
import time
import numpy as np
from components.cyclegan import CycleGANModel
from components.stereo_matching_net import StereoMatchingNet
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from dataset import pittburgh_rgb_nir
from attrdict import AttrDict
import yaml
from tqdm import tqdm
from pathlib import Path


parser = argparse.ArgumentParser()
parser.add_argument('--config_file',default='./configs/pittsburgh_test.yaml',help='path to the config file')
args = parser.parse_args()

with open(args.config_file) as fp:
    config = yaml.safe_load(fp)
    config = AttrDict(config)


def test(test_loader, stereo_net, spectral_net):
    ans = [[],[],[],[],[],[],[],[]]
    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader)):
            rgb_path = data['A_paths'][0]
            splitted = rgb_path.split('/')
            collection, key = splitted[2], splitted[-1].replace("_RGBResize.png","")
            data["A"] = nn.functional.interpolate(data["A"],(420,580))
            data["B"] = nn.functional.interpolate(data["B"],(420,580))
            spectral_net.set_input(data)
            spectral_net.forward()
            fake_B, fake_A, _, _ = spectral_net.get_images()
            data["fake_A"] = fake_A
            data["fake_B"] = fake_B

            stereo_net.set_input(data)
            stereo_net.forward()
            ldisp = nn.functional.interpolate(stereo_net.ldisps[0],(429,582))[0][0].cpu().numpy()

            f = open(Path(config.basepath) / collection / 'Keypoint' / (key + '_Keypoint.txt'), 'r')
            gts = f.readlines()
            f.close()
            for gt in gts:
                x, y, d, c = gt.split()
                x = round(float(x) * 582) - 1
                x = int(max(0,min(582, x)))
                y = round(float(y) * 429) - 1
                y = int(max(0,min(429, y)))
                d = float(d) * 582
                c = int(c)
                p = max(0, ldisp[y, x] * 582)
                ans[c].append((p-d)*(p-d))

        rmse = []
        for c in range(8):
            rmse.append(pow(sum(ans[c]) / len(ans[c]), 0.5))
        print('Common    Light     Glass     Glossy  Vegetation   Skin    Clothing    Bag       Mean')
        print(round(rmse[0], 4), '  ', round(rmse[1], 4), '  ', round(rmse[2], 4), '  ', round(rmse[3], 4), '  ', round(rmse[4], 4), '  ', round(rmse[5], 4), '  ', round(rmse[6], 4), '  ', round(rmse[7], 4), '  ', round(sum(rmse) / 8.0, 4))
        print()
            

        




if __name__ == '__main__':
    dataset = pittburgh_rgb_nir(config,mode='test')
    test_loader = torch.utils.data.DataLoader(dataset,1,shuffle=True)
    stereo_net  = StereoMatchingNet(config)
    spectral_net = CycleGANModel(config)

    stereo_net.load_ckpts(config.pretrained_epoch)
    spectral_net.load_ckpts(config.pretrained_epoch)

    test(test_loader,stereo_net, spectral_net)
    
