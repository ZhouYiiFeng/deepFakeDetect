# Project-Deepfake Detection
Here are some reimplements of SOTA methods.
## XceptionNet
- [code](https://github.com/ZhouYiiFeng/deepFakeDetect/tree/master/baseline/xceptionNet)

<!-- ### ToDo List:
- [x] Encoding in Style:
- [x] code
	- [x] change the cuda ln -s(sudo ln -s /usr/local/cuda-10.0 /usr/local/cuda). To switch cuda version. print(torch.version.cuda) to find your own pytorch cuda version. This version need to be the same with the version of nvcc compile.
	- [x] compile the apex the acc tool for pytorch which provided by nvidia. [github](https://github.com/NVIDIA/apex)
- [x] [FaceShifter-pytorch](https://github.com/Heonozis/FaceShifter-pytorch)
 -->
 
# Project-Deepfake Generation
## Face Shifter
- [code](https://github.com/ZhouYiiFeng/deepFakeDetect/tree/master/baseline/faceshifter)
- [paper](https://arxiv.org/abs/1912.13457)

Here are some results:
<p align='center'>  
  <img src='imgs/results.png' width='440'/>  
</p>

<p align='center'>
  <img src='imgs/results2.png' width='440'/>
</p>

<p align='center'>
  <img src='imgs/results3.png' width='440'/>
</p>

The first Model is in the condition of `lr_G` = 4e-4 `lr_D` = 4e-4 `l_adv` = 1 `l_att`=10 `L_id`=10 `L_rec`=5 `Bt_s` = 2.
The results of first row if the provided code: V1.0.

In addition we train the model of batch_size of 8, but the network do not have the same down-sample layer as original paper. 
In this experiencd, we up-sample the iden code (512,1,1) to 8x8 size by 3 `ConvTranspose2d` in ADDGenerator.

Trained on RTX2080Ti for 5-7 days.