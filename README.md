## CycleGAN for cross spectral stereo matching 
- [Unsupervised Cross-spectral Stereo Matching by Learning to Synthesize](https://arxiv.org/pdf/1903.01078.pdf)

## Steps to train on Pittsburgh dataset:
- Set up the config file in `./config` folder, adjust the hyperparameters as per your need
<<<<<<< HEAD
- Download the pittsburgh RGB-NIR stereo dataset specify the location in the config file (`basepath`)
- Don't forget to wxecute  `pip install -r requirements.txt`

=======
- Download the pittsburgh RGB-NIR stereo dataset from [here](https://drive.google.com/file/d/1ikj7UcnQsdxUfDF5EI3YcFK-p63KHQmy/view). 
- specify the location in the config file (`basepath`)
>>>>>>> 38b71ccbafb0e90b71f6719e5ff5815bae51bb5a

## Qualitative Results
CycleGAN results
| Left (RGB)  | Fake Left (NIR) | Right (NIR) | Fake Right (RGB) |
| :-----------: | :------: | :-----: | :------: |
|![Left](visuals/real_A_503_10.png) | ![Left Fake](visuals/fake_B_503_10.png) |  ![Right](visuals/real_B_503_10.png) |![Right Fake](visuals/fake_A_503_10.png) |
|![Left](visuals/real_A_709_14.png) | ![Left Fake](visuals/fake_B_709_14.png) |  ![Right](visuals/real_B_709_14.png) |![Right Fake](visuals/fake_A_709_14.png) |
|![Left](visuals/real_A_739_15.png) | ![Left Fake](visuals/fake_B_739_15.png) |  ![Right](visuals/real_B_739_15.png) |![Right Fake](visuals/fake_A_739_15.png) |

**The last row is a failure case. As can be seen, the color of the bus is not properly transferred by the GAN**

- Stereo Matching Results
    - **Available very soon**

## Quantitative Results
- **Available Soon**

## Trained Weights:
- Weights are available at following [link]("https://drive.google.com/drive/folders/1g0eLttO6W9YYuFfIgPQoGF9ixR1cjPkC?usp=sharing").
- Download them and place in the folder where you will be saving your weights (`./weights` according to the default config file)
- The weights will be saved in the following a particular format, for eg. `[epoch]_net_G_A.pth` and `latest_net_G_A.pth` which signifies the latest checkpoint, you can specify the epoch you want to load weights from in the config file.

## Training
<<<<<<< HEAD
- The model follows iterative optimization technique as described in the paper.
- Only CycleGAN is trained for 10 epochs, thus `warmup: True, stereo: False, auxilary: False` in config file
- Change `warmup: False, stereo: True, auxilary: False` for training the Stereo Matching Network for another 2-3 epochs
- Change `warmup: False, stereo: False, auxilary: True` for the global optimization stage, i.e., auxilary loss
- For quantitative analysis, prepare a config file (eg. `pittsburgh_test.yaml`) and run     `python test.py --config ./configs/pittsburgh_test.yaml`
=======
>>>>>>> 38b71ccbafb0e90b71f6719e5ff5815bae51bb5a

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
- to view loss logs run `tensorboard --logdir ./summary`
