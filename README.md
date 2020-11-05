# Fake-Face-detection-Generation
Some collected paper and personal notes relevant to Fake Face Detetection and Generation.

This file refers to [592McAvoy](https://github.com/592McAvoy/fake-face-detection)


## Challenge

- [Facebook] [Deepfake Detection Challenge]( https://www.kaggle.com/c/deepfake-detection-challenge/overview )
  - [unofficial github repo](https://github.com/drbh/deepfake-detection-challenge)

## Study

## I. Dataset
1. [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics/tree/master/dataset)

	- [benchmark](http://kaldir.vc.in.tum.de/faceforensics_benchmark)
  	- [paper](https://arxiv.org/abs/1901.08971):  [ICCV 2019] FaceForensics++: Learning to Detect Manipulated Facial Images
     	- 977 downloaded videos from youtube, 1000 original extracted sequences and its manipulated version
     	- generated based on *Deep-Fakes, Face2Face, FaceSwap and NeuralTextures*  
    - [note](https://zhoef.com/2020/07/13/25_DeepfakeDetection%E4%B9%8BFaceForensics++/#more)

2. [DeepFake Forensics (Celeb-DF) Dataset](http://www.cs.albany.edu/~lsw/celeb-deepfakeforensics.html)   

    - [paper](https://arxiv.org/abs/1909.12962): [arXiv 2019] Celeb-DF: A New Dataset for DeepFake Forensics
    	- real and *DeepFake* synthesized videos having similar visual quality on par with those circulated online 
    	- 408 original videos collected from YouTube with subjects of different ages, ethic groups and genders, and 795 DeepFake videos synthesized from these real videos. 

3. [Celeb-DF-2.0]

 	- [paper](https://arxiv.org/abs/1909.12962) 
 	- [github](https://github.com/yuezunli/celeb-deepfakeforensics)
	- YouTube-real (300)
	- Celeb-synthesis (5639)
	- Celeb-real (590)
	- This dataset is produced by share encoder and seperate decoder network.

4. [DFDC] The facebook challenge dataset.

	- [paper](https://arxiv.org/abs/1910.08854)
	- [kaggel](https://www.kaggle.com/c/deepfake-detection-challenge/data)
	- Total size training (zip): 472GB.	

## II. Current Work

### (1) Special Artifact-Based


### (2) CNN-Based


### (3) Video forensics


### (4) Two Stream


### (5) Auto-encoder


### (6) Frequency Domain

### (7) General image manipulation


### (8) Novel Network or Module


### (9) GAN-fake face detection


### (10) Domain Adaptation


### (11) Metrics Learning

# Project-Deepfake Detection
## XceptionNet
- [code](https://github.com/ZhouYiiFeng/deepFakeDetect/tree/master/baseline/xceptionNet)

## Crop the facial area.
Using the face alignment network FAN to detect the facial landmarks. The original code only provide the case where batch size equal to 1. We modified the code to work on arb. batch size. In `crop_batch_face.py` the FAN is able to detect batch_size face images, which improve the processing cost.

However the frame-extracted dataset is to large, it still takes about 2days to crop the whole datasets (On two RTX2080Ti)

## Validate the Crop results
In the dataset of baseline we coded two dataset(df++ and celeb_df-v2). Each can be visualised by the `draw_batch` function.

## Dataset:
We choose three popular deepfake datasets to conduct our experiments.
- DeeperForensics-1.0 [paper](https://arxiv.org/abs/2001.03024.pdf) and [github](https://github.com/EndlessSora/DeeperForensics-1.0)
	- source-video
		- 100 people with different identity.
			- 9 different light diraction.
				- 8 different expression.
					- 7 different camera diraction.
	- manipulate-video
		- 11 different post-processing method results.
			- 1000 synthesis video (by using different source and target pair.)
	- Store in 1024 `/mnt/disk3/std/zyf/dataset/deepfake/deepforenci`
	+ [ ] original face images
	+ [x] manipulated face images

- Celeb-DF-2.0 [paper](https://arxiv.org/abs/1909.12962) and [github](https://github.com/yuezunli/celeb-deepfakeforensics)
	- YouTube-real (300)  
	- Celeb-synthesis (5639)
	- Celeb-real (590)
	- Store in 6665. `/mnt/hdd3/zyf/dataset/deepfake/celeb_df_v2`
	+ [x] original face images
	+ [x] manipulated face images

- FaceForensics++ [paper](https://arxiv.org/abs/1901.08971) and [github](https://github.com/ondyari/FaceForensics)
	- original_sequences(from youtube: 1000.) *which is regarded as the target video in DeeperForensics*
	- manipulated_sequences
		- Deepfakes (1000)
		- NeuralTextures (1000)
		- FaceSwap (1000)
		- Face2Face (1000)
	- Store in 1024/6665 `/mnt/hdd3/zyf/dataset/deepfake/faceforensics++/original_sequences`
	+ [x] original face images
	+ [x] manipulated face images

- CelebA
	- [paper](https://arxiv.org/abs/1411.7766)
	- [homepage](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
	- dir:
		- In-The-Wild Images (Img/img_celeba.7z)
		- Align & Cropped Images (Img/img_align_celeba.zip & Img/img_align_celeba_png.7z)

- CelebA-HQ
	- Generate [blog](https://www.jianshu.com/p/1fcaccfedd71)

For fake face synthesize, only need the real video. For fake face detection, we need both the fake and real video. However, using whole image to conduct the detection is time-consuming and ineffective, we focus on the face areas to imporve the performance of our methods. We use the FAN to crop the facial area for the real and fake (maniplate) images. Each facial areas is face centered and padding with 0 when it out of bounds. 

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

