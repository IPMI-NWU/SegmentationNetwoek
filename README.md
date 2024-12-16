# UDA_pytorch
A Network for Breast Tumor Segmentation Using Multi-sequence Joint Analysis
The network utilizes dense connection networks, employing skip connections to combine deep and shallow semantic representations of features. 

Additionally, a sequence-aware module is designed to perform weighted fusion of features from different sequences. Convolutional LSTM is used to extract temporal information contained in the DCE-MRI sequences, allowing the network to simultaneously consider both morphological features of the tumor and hemodynamic features.

# Train:

3lstm_mm_unet:
`lstm_mmunet_my.py` (GPU code)
`lstm_mmunet_my_cpu` (CPU code)
`train.py` for training the network

# Test:

draw_tumor: Code for visualizing prediction results
draw_curve: Code for plotting loss and dice curves



# Requirements
Some important required packages include:
* torch == 2.3.0
* torchvision == 0.18.0
* Python == 3.10.14
* numpy == 1.26.4