# A hybrid sampling and gradient attention network for compressed image sensing

This is the PyTorch implementation of the 2022 Visual Computer paper: A hybrid sampling and gradient attention network for compressed image sensing (https://link.springer.com/article/10.1007/s00371-022-02585-0) by Xin Yang, Chunling Yang, Wenjun Chen.

South China University of Technology

## Description

(1) The training model file is saved in the check_point folder

(2) The permutation_matrix_generation.m file is the matlab code for generating the permutation matrix, corresponding to the block permutation matrix selection algorithm based on image information entropy in the paper

(3) Each permutation matrix is ​​saved under the permutation_matrix folder

(4) bsds500_train.npy, bsds500_val.npy represent the data path list of the training and verification set data of the BSDS500 dataset

(5) The dataloader_bsds500.py file loads the code for the training data

        1. image_height and image_width represent the height and width of the image respectively
        
        2. The load_filename parameter represents the parameter path list of the training data

(6) model_new.py is the model implementation file

        1. softmax_gate_net_small is used to generate gradient attention map
        
        2. sample_and_inirecon is implemented for sampling and initial reconstruction
        
        3. fusion_ini_9_small is the implementation of the initial reconstruction fusion sub-network
        
        4. Biv_Shr_small is implemented for the denoising residual block

(7) test_hsganet.py is the test code

        1. sr is the overall sampling rate, and the preset BPS ratio is 80%.
        
        2. test_video_name is used to store the test path
        
(8) The train_tvc_12phase.py file is the training code

## Citation

If you find the code helpful in your resarch or work, please cite the following papers.

   @article{yang2022hybrid,
        title={A hybrid sampling and gradient attention network for compressed image sensing},
        author={Yang, Xin and Yang, Chunling and Chen, Wenjun},
        journal={The Visual Computer},
        pages={1--14},
        year={2022},
        publisher={Springer}
      }
