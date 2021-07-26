## CycleGAN for cross spectral stereo matching 
- [Unsupervised Cross-spectral Stereo Matching by Learning to Synthesize](https://arxiv.org/pdf/1903.01078.pdf)

## Steps to train on Pittsburgh dataset:
- Set up the config file in `./config` folder, adjust the hyperparameters as per your need
- Download the pittsburgh RGB-NIR stereo dataset from [here](https://drive.google.com/file/d/1ikj7UcnQsdxUfDF5EI3YcFK-p63KHQmy/view). 
- specify the location in the config file (`basepath`)

## Qualitative Results
- **Available Soon**

## Quantitative Results
- **Available Soon**

## Trained Weights:
- **Available Soon**

## Training

To start the training, run `python train.py --config_file [PATH_TO_CONFIG_FILE]`, to use the default configuration, directly run `python train.py`
- The Network follows iterative optimization technique
- step1 & step2 --> Training Feature Extractor, Generators & The Discriminators, block gradients to all other networks (set `warmup`:True, `stereo`:False, `auxilary`:False in the config file)
- step3 --> Training Stereo Matching Network (set `warmup`: False, `stereo`: True, `auxilary`: False in the config file)
- step4 --> Train based on auxilary loss, don't forget to block Stereo Matching Network for this step! (set `warmup`: False, `stereo`: False, `auxilary`: True)
- The network is trained for 10 epochs with only step1 and step2
- Train the network for step 2 for another 10 epochs
- Train the network for another 2-3 epochs in step4 for global optimization.

## Summary
- Summaries are created inside `./summary` folder
- to view loss logs, intermediate results run `tensorboard --logdir ./summary`
