class Path(object):
    @staticmethod
    def db_dir(database):
        if database == 'ff++':
            # folder that contains class labels
            root_img_dir = '/mnt/disk3/std/zyf/dataset/deepfake/faceforensics++'
            root_video_dir = '/mnt/disk2/liziyuan/zhouyifeng/datasets/fakesDetection/faceforensics++'

            # Save preprocess data into output_dir
            inner_test_dir = '/mnt/disk2/liziyuan/zhouyifeng/deepfakes/dcdd/baselines/xecptionNet/inner_test_results'
            output_dir = '/mnt/disk2/liziyuan/zhouyifeng/deepfakes/dcdd/baselines/xecptionNet/results'
            return root_img_dir, root_video_dir, inner_test_dir, output_dir
        elif database == 'DFDC':
            pass
        elif database == 'celeb_df_v2':
            pass
        elif database == 'deepf':
            pass
        else:
            print('Database {} not available.'.format(database))
            raise NotImplementedError

    # @staticmethod
    # def aug_dir():
    #     return './dataAug'

    # @staticmethod
    # def model_dir():
    #     # return '/path/to/Models/c3d-pretrained.pth'
    #     pass

    # @staticmethod
    # def pre_model_dir():
    #     return '/mnt/disk2/liziyuan/zhouyifeng/video/DualFlow/pretrained_models'
    #     pass