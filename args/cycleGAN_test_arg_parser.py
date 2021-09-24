'''
Author: your name
Date: 2021-07-09 08:50:30
LastEditTime: 2021-09-24 08:44:35
LastEditors: Please set LastEditors
Description: In User Settings Edit
FilePath: /MaskCycleGAN-VC（中文）（2）/args/cycleGAN_test_arg_parser.py
'''
"""
Arguments for MaskCycleGAN-VC testing.
Inherits BaseArgParser.
"""

from args.base_arg_parser import BaseArgParser

class CycleGANTestArgParser(BaseArgParser):
    """
    Class which implements an argument parser for args used only in training MaskCycleGAN-VC.
    It inherits TrainArgParser.
    """

    def __init__(self):
        super(CycleGANTestArgParser, self).__init__()
        self.parser.add_argument('--sample_rate', type=int, default=22050, help='Sampling rate of mel-spectrograms.')
        self.parser.add_argument(
            '--speaker_A_id', type=str, default="VCC2SF3", help='Source speaker id (From VOC dataset).')
        self.parser.add_argument(
            '--speaker_B_id', type=str, default="VCC2TF1", help='Source speaker id (From VOC dataset).')
        self.parser.add_argument(
            '--preprocessed_data_dir', type=str, default="vcc2018_training_preprocessed/", help='Directory containing preprocessed dataset files.')
        self.parser.add_argument(
            '--ckpt_dir', type=str, default=None, help='Path to model ckpt.')
        self.parser.add_argument(
            '--model_name', type=str, choices=('generator_A2B', 'generator_B2A','discriminator_A','discriminator_B','discriminator_A2','discriminator_B2'), default='generator_A2B', help='Name of model to load.')
        self.parser.add_argument(
            '--real_voice_level', type=float, default=0, help='real trian voice number')