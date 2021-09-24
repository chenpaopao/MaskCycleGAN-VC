'''
Author: lxc
Date: 2021-08-01 11:23:50
LastEditTime: 2021-09-10 09:34:00
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /MaskCycleGAN-VC（中文）（2）/mask_cyclegan_vc/distest.py
'''
import os
import pickle
import numpy as np
from tqdm import tqdm
import wave
import os
import numpy as np
from pydub import AudioSegment
import matplotlib.pylab as plt

from pydub import AudioSegment
  
  
def trans_mp3_to_wav(filepath):
    for i in os.listdir(filepath):
        print(filepath+"/"+ i)
        song = AudioSegment.from_wav(filepath+"/"+ i)
        song.export("/media/ittc-819/85083a24-bfb5-482f-a28c-b95331745df1/home/ittc-819/VC-CODE/MaskCycleGAN-VC（中文）（2）/audio_samples/2/"+i[:-4]+".mp3", format="mp3")

trans_mp3_to_wav("/media/ittc-819/85083a24-bfb5-482f-a28c-b95331745df1/home/ittc-819/VC-CODE/MaskCycleGAN-VC（中文）（2）/results/mask_cyclegan_vc_neutral-chinaman6/converted_audio")