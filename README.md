# DualStreamCNN
Tensorflow implementation for reproducing SAR results in the paper Dual-Stream CNN for Structured Time Series Classification by Shuchen Weng*, Wenbo Li*, Yi Zhang, Siwei Lyu.

 <img src="https://github.com/SCWengTJU/DualStreamCNN/blob/master/Figures/Fig1.png" width = "900" height = "600" align=center />

## Dependencies
* python 2.7  
* Tensorflow  

## Data
Training data is saved in ./train_raw/
Testing data is saved in ./test_raw/

## Pretrained Model
Download here and save it to ./init_weight/

## Training
1. Interpolate training data to 68 frames, as what ./interpolation.py did. Then save processed data to ./train_67/
2. Run ./get_train.py and numpy data will be saved in ./train_dis_np_67/
3. Run ./np2tf.py to convert '.npy' to '.tfrecord'. The directory ./train_67_tfrecord/ will be created automatically
4. Run ./train.py to load pretrained model and start training

## Validation
1. Run ./Interplation.py to interpolate frames to 68. Processed data will be saved in ./test_67/
2. Run ./get_test.py to get test input, saved in ./test_dis_67_input/
3. Run ./test.py to get test results

## Result

| Methods | MSR Action3D | ChaLearn | 3D-SAR-140 |
| :------: | :------: | :------: | :------: |
| RR | 0.891 | 0.438 | 0.723 |
| HBRNN-L | 0.897 | 0.559 | 0.604 |
| CHARM | 0.747 | 0.476 | 0.618 |
| DBN-HMM | 0.735 | 0.628 | 0.601 |
| Lie-group | 0.866 | 0.401 | 0.745 |
| HOD | 0.844 | 0.539 | 0.657 |
| MP | 0.909 | 0.452 | 0.203 |
| SSS | 0.560 | 0.413 | 0.253 |
| HBRNN-L-T | 0.915 | 0.673 | 0.756 |
| URNN-2L-T | 0.931 | 0.753 | 0.892 |
| **Ours** | **0.963** | **0.772** | **0.896** |

## Citing AttnGAN
If you find DualStreamCNN useful in your research, please consider citing:  

@article{DualStreamCNN，</br>
&emsp; author &emsp;=&ensp; {Shuchen Weng*, Wenbo Li*, Yi Zhang, Siwei Lyu},  
&emsp; title &emsp;=&ensp; {Dual-Stream CNN for Structured Time Series Classification},  
&emsp; Year &ensp; = &ensp;{2019},  
&emsp; booktitle &ensp; = &ensp;{{ICASSP}}  
}


