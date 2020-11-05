# Project-Deepfake Generation
## Face Shifter
- [code](https://github.com/Heonozis/FaceShifter-pytorch)
- [paper](https://arxiv.org/abs/1912.13457)

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


### Experiments:
Implement in Pytorch.
