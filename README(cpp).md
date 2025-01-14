# MaskCycleGAN-VC# MaskCycleGAN-VC
Unofficial **PyTorch** implementation of Kaneko et al.'s [**MaskCycleGAN-VC**](http://www.kecl.ntt.co.jp/people/kaneko.takuhiro/projects/maskcyclegan-vc/index.html) (2021) for non-parallel voice conversion.

MaskCycleGAN-VC is the state of the art method for non-parallel voice conversion using CycleGAN. It is trained using a novel auxiliary task of filling in frames (FIF) by applying a temporal mask to the input Mel-spectrogram. It demonstrates marked improvements over prior models such as CycleGAN-VC (2018), CycleGAN-VC2 (2019), and CycleGAN-VC3 (2020).

<p align="center">
<img src="imgs/MaskedCycleGAN-VC.png" width="500">
<br>
<b>Figure1: MaskCycleGAN-VC Training</b>
<br><br><br><br>
</p>

<p align="center">
<img src="imgs/generator.png" width="800">
<br>
<b>Figure2: MaskCycleGAN-VC Generator Architecture</b>
<br><br><br><br>
</p>

<p align="center">
<img src="imgs/discriminator.png" width="500">
<br>
<b>Figure3: MaskCycleGAN-VC PatchGAN Discriminator Architecture</b>
<br><br><br><br>
</p>

Paper: https://arxiv.org/pdf/2102.12841.pdf

Repository Contributors: [Claire Pajot](https://github.com/cmpajot), [Hikaru Hotta](https://github.com/HikaruHotta), [Sofian Zalouk](https://github.com/szalouk)


## Setup

Clone the repository.

```
git clone git@github.com:GANtastic3/MaskCycleGAN-VC.git
cd MaskCycleGAN-VC
```

Create the conda environment.
```
conda env create -f environment.yml
conda activate MaskCycleGAN-VC
```

## VCC2018 Dataset

The authors of the paper used the dataset from the Spoke task of [Voice Conversion Challenge 2018 (VCC2018)](https://datashare.ed.ac.uk/handle/10283/3061). This is a dataset of non-parallel utterances from 6 male and 6 female speakers. Each speaker utters approximately 80 sentences.

Download the dataset from the command line.
```
wget --no-check-certificate https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_training.zip?sequence=2&isAllowed=y
wget --no-check-certificate https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_evaluation.zip?sequence=3&isAllowed=y
wget --no-check-certificate https://datashare.ed.ac.uk/bitstream/handle/10283/3061/vcc2018_database_reference.zip?sequence=5&isAllowed=y
```

Unzip the dataset file.
```
mkdir vcc2018
apt-get install unzip
unzip vcc2018_database_training.zip?sequence=2 -d vcc2018/
unzip vcc2018_database_evaluation.zip?sequence=3 -d vcc2018/
unzip vcc2018_database_reference.zip?sequence=5 -d vcc2018/
mv -v vcc2018/vcc2018_reference/* vcc2018/vcc2018_evaluation
rm -rf vcc2018/vcc2018_reference
```

## Data Preprocessing

To expedite training, we preprocess the dataset by converting waveforms to melspectograms, then save the spectrograms as pickle files `<speaker_id>normalized.pickle` and normalization statistics (mean, std) as npz files `<speaker_id>_norm_stats.npz`. We convert waveforms to spectrograms using a [melgan vocoder](https://github.com/descriptinc/melgan-neurips) to ensure that you can decode voice converted spectrograms to waveform and listen to your samples during inference.

```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018/vcc2018_training \
  --preprocessed_data_directory vcc2018_preprocessed/vcc2018_training \
  --speaker_ids VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4 VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2 chinaman6 chinawomen2 chinaman3 chinaman4
```

```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018/vcc2018_evaluation \
  --preprocessed_data_directory vcc2018_preprocessed/vcc2018_evaluation \
  --speaker_ids VCC2SF1 VCC2SF2 VCC2SF3 VCC2SF4 VCC2SM1 VCC2SM2 VCC2SM3 VCC2SM4 VCC2TF1 VCC2TF2 VCC2TM1 VCC2TM2 chinaman1 chinaman2 chinaman3 chinaman4
```


## Training

Train MaskCycleGAN-VC to convert between `<speaker_A_id>` and `<speaker_B_id>`. You should start to get excellent results after only several hundred epochs.
```
python -W ignore::UserWarning -m mask_cyclegan_vc.train \
    --name mask_cyclegan_vc_chinawomen1_happy \
    --seed 0 \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training/ \
    --speaker_A_id chinawomen1 \
    --speaker_B_id happy \
    --epochs_per_save 100 \
    --epochs_per_plot 10 \
    --num_epochs 900 \
    --batch_size 1 \
    --generator_lr 2e-4 \
    --discriminator_lr 1e-4  \
    --decay_after 1e4 \
    --sample_rate 22050 \
    --num_frames 64 \
    --max_mask_len 25 \
    --gpu_ids 0 \
```

To continue training from a previous checkpoint in the case that training is suspended, add the argument `--continue_train` while keeping all others the same. The model saver class will automatically load the most recently saved checkpoint and resume training.

Launch Tensorboard in a separate terminal window.
```
tensorboard --logdir results/logs
```
远程访问tensorboard，服务器输入，本地计算机：http://服务器ip:8080
```
tensorboard --bind_all --logdir results/logs --port 8080
```

## Testing

Test your trained MaskCycleGAN-VC by converting between `<speaker_A_id>` and `<speaker_B_id>` on the evaluation dataset. Your converted .wav files are stored in `results/<name>/converted_audio`.

```
python -W ignore::UserWarning -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_chinawomen1_happy \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_evaluation \
    --gpu_ids 0 \
    --speaker_A_id chinaman6 \
    --speaker_B_id happy \ 
    --ckpt_dir results/mask_cyclegan_vc_chinawomen1_happy/ckpts \
    --load_epoch 900 \
    --model_name generator_B2A \
```
```
python -W ignore::UserWarning -m mask_cyclegan_vc.test \
    --name mask_cyclegan_vc_happy-chinaman6 \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_evaluation \
    --gpu_ids 0 \
    --speaker_A_id chinaman6 \
    --speaker_B_id happy \  
    --ckpt_dir results/m    --ckpt_dir results/mask_cyclegan_vc_chinawomen1_happy/ckpts 
    --load_epoch 900 \
    --model_name generator_B2A \
```
discriminator:
### 先预处理待测试的数据：
```
python data_preprocessing/preprocess_vcc2018.py \
  --data_directory vcc2018/vcc2018_training \
  --preprocessed_data_directory vcc2018_preprocessed/vcc2018_training \
  --speaker_ids fakeman6
```

step1:需要获得realvioce的dis辨别器的均值
运行 如果是检测A：model_name discriminator_A，否则model_name discriminator_B
```
python -W ignore::UserWarning -m mask_cyclegan_vc.distest \
    --name mask_cyclegan_vc_chinaman6_chinaman5 \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_training \
    --gpu_ids 0 \
    --speaker_A_id chinaman6 \
    --speaker_B_id chinaman5 \
    --ckpt_dir results/mask_cyclegan_vc_chinaman6_chinaman5/ckpts \
    --load_epoch 900 \
    --model_name discriminator_A \
```
获得返回的最后一个  tensor(0.0841, device='cuda:0', grad_fn=<DivBackward0>)作为real_voice_level的值输入
### 注意：这里如果判别是否是a说话人：model_name discriminator_A，speaker_A_id 为处理过的待检测的语音文件夹，speaker_B_id不用管
step2：
```
python -W ignore::UserWarning -m mask_cyclegan_vc.distest \
    --name mask_cyclegan_vc_chinaman6_chinaman5 \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_evaluation \
    --gpu_ids 0 \
    --speaker_A_id chinaman6 \
    --speaker_B_id chinaman5 \ 
    --ckpt_dir results/mask_cyclegan_vc_chinaman6_chinaman5/ckpts \
    --load_epoch 900 \
    --model_name discriminator_A \
    --real_voice_level  0.0841
```
```
python -W ignore::UserWarning -m mask_cyclegan_vc.distest \
    --name mask_cyclegan_vc_chinaman6_chinaman5 \
    --save_dir results/ \
    --preprocessed_data_dir vcc2018_preprocessed/vcc2018_evaluation \
    --gpu_ids 0 \
    --speaker_A_id chinaman6 \
    --speaker_B_id chinaman5 \
    --ckpt_dir results/mask_cyclegan_vc_chinaman6_chinaman5/ckpts \
    --load_epoch 900 \
    --model_name discriminator_A \
```
Toggle between A->B and B->A conversion by setting `--model_name` as either `generator_A2B` or `generator_B2A`.

Select the epoch to load your model from by setting `--load_epoch`.

## Code Organization
```
├── README.md                       <- Top-level README.
├── environment.yml                 <- Conda environment
├── .gitignore
├── LICENSE
|
├── args
│   ├── base_arg_parser             <- arg parser
│   ├── train_arg_parser            <- arg parser for training (inherits base_arg_parser)
│   ├── cycleGAN_train_arg_parser   <- arg parser for training MaskCycleGAN-VC (inherits train_arg_parser)
│   ├── cycleGAN_test_arg_parser    <- arg parser for testing MaskCycleGAN-VC (inherits base_arg_parser)
│
├── bash_scripts
│   ├── mask_cyclegan_train.sh      <- sample script to train MaskCycleGAN-VC
│   ├── mask_cyclegan_test.sh       <- sample script to test MaskCycleGAN-VC
│
├── data_preprocessing
│   ├── preprocess_vcc2018.py       <- preprocess VCC2018 dataset
│
├── dataset
│   ├── vc_dataset.py               <- torch dataset class for MaskCycleGAN-VC
│
├── logger
│   ├── base_logger.sh              <- logging to Tensorboard
│   ├── train_logger.sh             <- logging to Tensorboard during training (inherits base_logger)
│
├── saver
│   ├── model_saver.py              <- saves and loads models
│
├── mask_cyclegan_vc
│   ├── model.py                    <- defines MaskCycleGAN-VC model architecture
│   ├── train.py                    <- training script for MaskCycleGAN-VC
│   ├── test.py                     <- training script for MaskCycleGAN-VC
│   ├── utils.py                    <- utility functions to train and test MaskCycleGAN-VC

```

## Acknowledgements

This repository was inspired by [jackaduma](https://github.com/jackaduma)'s implementation of [CycleGAN-VC2](https://github.com/jackaduma/CycleGAN-VC2).
