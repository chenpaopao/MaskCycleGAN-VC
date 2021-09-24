import os
import pickle
import numpy as np
from tqdm import tqdm
import torch
import torch.utils.data as data
import torchaudio

from mask_cyclegan_vc.model import Generator, Discriminator
from args.cycleGAN_test_arg_parser import CycleGANTestArgParser
from dataset.vc_dataset import VCDataset
from mask_cyclegan_vc.utils import decode_melspectrogram, vcode_melspectrogram
from logger.train_logger import TrainLogger
from saver.model_saver import ModelSaver


class MaskCycleGANVC_Distesting(object):
    """Tester for MaskCycleGAN-VC
    """

    def __init__(self, args):
        """
        Args:
            args (Namespace): Program arguments from argparser
        """
        # Store Args
        self.device = args.device
        #self.converted_audio_dir = os.path.join(args.save_dir, args.name, 'converted_audio')
        #os.makedirs(self.converted_audio_dir, exist_ok=True)
        self.model_name = args.model_name

        self.speaker_A_id = args.speaker_A_id
        self.speaker_B_id = args.speaker_B_id
        # Initialize MelGAN-Vocoder used to decode Mel-spectrograms
        self.vocoder = torch.hub.load(
            'descriptinc/melgan-neurips', 'load_melgan')
        self.sample_rate = args.sample_rate

        # Initialize speakerA's dataset
        self.dataset_A = self.loadPickleFile(os.path.join(
            args.preprocessed_data_dir, self.speaker_A_id, f"{self.speaker_A_id}_normalized.pickle"))
        dataset_A_norm_stats = np.load(os.path.join(
            args.preprocessed_data_dir, self.speaker_A_id, f"{self.speaker_A_id}_norm_stat.npz"))
        self.dataset_A_mean = dataset_A_norm_stats['mean']
        self.dataset_A_std = dataset_A_norm_stats['std']

        # Initialize speakerB's dataset
        self.dataset_B = self.loadPickleFile(os.path.join(
            args.preprocessed_data_dir, self.speaker_B_id, f"{self.speaker_B_id}_normalized.pickle"))
        dataset_B_norm_stats = np.load(os.path.join(
            args.preprocessed_data_dir, self.speaker_B_id, f"{self.speaker_B_id}_norm_stat.npz"))
        self.dataset_B_mean = dataset_B_norm_stats['mean']
        self.dataset_B_std = dataset_B_norm_stats['std']

        source_dataset = self.dataset_A if self.model_name == 'discriminator_A' else self.dataset_B
        self.dataset = VCDataset(datasetA=source_dataset,
                                 datasetB=None,
                                 valid=True)
        self.test_dataloader = torch.utils.data.DataLoader(dataset=self.dataset,
                                                           batch_size=1,
                                                           shuffle=False,
                                                           drop_last=False)

        # Generator
        self.generator = Generator().to(self.device)
        self.generator.eval()

        # Discriminator
        self.discriminator = Discriminator().to(self.device)
        self.discriminator.eval()
        # Load Discriminator from ckpt
        self.saver = ModelSaver(args)
        self.saver.load_model(self.discriminator, self.model_name)

    def loadPickleFile(self, fileName):
        """Loads a Pickle file.

        Args:
            fileName (str): pickle file path

        Returns:
            file object: The loaded pickle file object
        """
        with open(fileName, 'rb') as f:
            return pickle.load(f)

    def test(self):
        for i, sample in enumerate(tqdm(self.test_dataloader)):

            save_path = None
            if self.model_name == 'discriminator_A':
                real_A = sample
                real_A = real_A.to(self.device, dtype=torch.float)
                wav_real_A = decode_melspectrogram(self.vocoder, real_A[0].detach(
                ).cpu(), self.dataset_A_mean, self.dataset_A_std).cpu()
                melspectrogram_A = vcode_melspectrogram(self.vocoder, wav_real_A)
                result_A = self.discriminator(real_A)
                print(result_A)
            else:
                real_B = sample
                real_B = real_B.to(self.device, dtype=torch.float)
                wav_real_B = decode_melspectrogram(self.vocoder, real_B[0].detach(
                ).cpu(), self.dataset_B_mean, self.dataset_B_std).cpu()
                melspectrogram_B = vcode_melspectrogram(self.vocoder, wav_real_B)
                result_B = self.discriminator(real_B)
                print(result_B)


if __name__ == "__main__":
    parser = CycleGANTestArgParser()
    args = parser.parse_args()
    tester = MaskCycleGANVC_Distesting(args)
    tester.test()