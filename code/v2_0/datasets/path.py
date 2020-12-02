class Path(object):
    @staticmethod
    def FaceForensicsPaths(port):

        # folder that contains class labels
        if port == 1024:
            root_face_dir = '/mnt/disk3/std/zyf/dataset/deepfake/faceforensics++/original_sequences/youtube/c23/face'
            root_anno_dir = '/mnt/disk3/std/zyf/dataset/deepfake/faceforensics++/original_sequences/youtube/c23/anno'

            # Save preprocess data into output_dir
            inner_test_dir = '/mnt/disk2/liziyuan/zhouyifeng/deepfakes/faceshifer/FaceShifter-pytorch/mid_results/FaceForensics'
            output_dir = '/mnt/disk2/liziyuan/zhouyifeng/deepfakes/faceshifer/FaceShifter-pytorch/results/FaceForensics'
            return root_face_dir, root_anno_dir, inner_test_dir, output_dir
        elif port == 6665:
            root_face_dir = '/mnt/hdd3/zyf/dataset/deepfake/faceforensics++/original_sequences/youtube/c23/face'
            root_anno_dir = '/mnt/hdd3/zyf/dataset/deepfake/faceforensics++/original_sequences/youtube/c23/anno'

            # Save preprocess data into output_dir
            inner_test_dir = '/mnt/hdd2/std/zyf/dfke/faceGen/FaceShifter-pytorch/mid_results/FaceForensics'
            output_dir = '/mnt/hdd2/std/zyf/dfke/faceGen/FaceShifter-pytorch/results/FaceForensics'
            return root_face_dir, root_anno_dir, inner_test_dir, output_dir
        elif port == 65000:
            root_face_dir = '/mnt/disk/zyf/datasets/deepfake/faceforensics++/original_sequences/youtube/c23/face'
            root_anno_dir = '/mnt/disk/zyf/datasets/deepfake/faceforensics++/original_sequences/youtube/c23/anno'

            # Save preprocess data into output_dir
            inner_test_dir = '/mnt/disk/zyf/dfk/deepFakeDetect/baseline/faceshifter/mid_results/FaceForensics'
            output_dir = '/mnt/disk/zyf/dfk/deepFakeDetect/baseline/faceshifter/results/FaceForensics'
            return root_face_dir, root_anno_dir, inner_test_dir, output_dir


    @staticmethod
    def aug_dir():
        return './dataAug'
