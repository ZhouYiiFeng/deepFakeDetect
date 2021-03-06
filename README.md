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


