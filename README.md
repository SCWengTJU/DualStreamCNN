# DualStreamCNN
Tensorflow implementation for reproducing SAR results in the paper Dual-Stream CNN for Structured Time Series Classification by Shuchen Weng*, Wenbo Li*, Yi Zhang, Siwei Lyu.

 <img src="https://github.com/SCWengTJU/DualStreamCNN/blob/master/Figures/Fig1.png" width = "900" height = "600" align=center />

## Dependencies
python 2.7  
Tensorflow  

## Data
Training data is saved in ./train_raw/
Testing data is saved in ./test_raw/

## Pretrained Model
Download here and save it to ./init_weight/

## Training
1. Interpolate training data to 68 frames, as what ./interpolation.py did. Then save processed data to ./train_67/
2. Run ./get_train.py. Numpy data will be saved in ./train_dis_np_67/
3. Run ./np2tf.py to convert '.npy' to '.tfrecord'


## Validation

## Result

## Citing AttnGAN
If you find DualStreamCNN useful in your research, please consider citing:  

@article{  
    author    = {Shuchen Weng*, Wenbo Li*, Yi Zhang, Siwei Lyu},  
    title     = {Dual-Stream CNN for Structured Time Series Classification},  
    Year = {2019},  
    booktitle = {{ICASSP}}  
}


