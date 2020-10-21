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
	- [] original face images
	- [x] manipulated face images

- Celeb-DF-2.0 [paper](https://arxiv.org/abs/1909.12962) and [github](https://github.com/yuezunli/celeb-deepfakeforensics)
	- YouTube-real (300)
	- Celeb-synthesis (5639)
	- Celeb-real (590)
	- [] original face images
	- [] manipulated face images

- FaceForensics++ [paper](https://arxiv.org/abs/1901.08971) and [github](https://github.com/ondyari/FaceForensics)
	- original_sequences(from youtube: 1000.) *which is regarded as the target video in DeeperForensics*
	- manipulated_sequences
		- Deepfakes (1000)
		- NeuralTextures (1000)
		- FaceSwap (1000)
		- Face2Face (1000)
	- [x] original face images
	- [x] manipulated face images

For fake face synthesize, only need the real video. For fake face detection, we need both the fake and real video. However, using whole image to conduct the detection is time-consuming and effectiveless, we focus on the face areas to imporve the performance of our methods. We use the FAN to crop the facial area for the real and fake (maniplate) images. Each facial areas is face centered and padding with 0 when it out of bounds. 