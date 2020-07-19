from tqdm import tqdm
from datetime import datetime
import os
import argparse
from torch.utils.data import DataLoader
import torch
from util import utils
from dataset.ffpp_dataset import FFPPDataset
from tensorboardX import SummaryWriter

def _main(opts):
    opts.model_dir = os.path.join(opts.checkpoints_dir, opts.exp_name)
    opts.lr_min = opts.lr_init * opts.lr_min_m
    print("========================================================")
    print("===> Save model to %s" % opts.model_dir)
    print("========================================================")
    if not os.path.isdir(opts.model_dir):
        os.makedirs(opts.model_dir)

    ### initialize model
    print('===> Initializing %s model from %s...' % (opts.model, opts.exp_name))
    # model = Xception(num_classes=2)

    if opts.which_epoch == 'latest':
        epoch_st = utils.findLatestEpoch(opts)
    else:
        epoch_st = int(opts.which_epoch)

    if epoch_st == 0:
        print('\n=====================================================================')
        print("===> Initializing Model with pretrained model from %s" % opts.pretrain_path)
        print('=====================================================================')
        # load pretrain
        model = utils.loadPretrainModel(path=opts.pretrain_path)

    ### initialize optimizer
    if opts.solver == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=opts.lr_init, momentum=opts.momentum,
                                    weight_decay=opts.weight_decay)
    elif opts.solver == 'ADAM':
        optimizer = torch.optim.Adam(model.parameters(), lr=opts.lr_init, weight_decay=opts.weight_decay,
                                     betas=(opts.beta1, opts.beta2))
    else:
        raise Exception("Not supported solver (%s)" % opts.solver)

    ### load the cp
    if epoch_st > 0:
        print('\n=====================================================================')
        print("===> Load Model from %d epoch" % epoch_st)
        print('=====================================================================')
        model, optimizer = utils.load_model(model, optimizer, opts, epoch_st)

    if opts.cuda:
        model = model.cuda()
    model.train()

    ### initialize loss writer
    loss_dir = os.path.join(opts.checkpoints_dir, opts.exp_name, 'loss')
    if not os.path.isdir(loss_dir):
        os.makedirs(loss_dir)
    loss_writer = SummaryWriter(loss_dir)

    crtirion = torch.nn.CrossEntropyLoss()
    train_data = FFPPDataset(opts, "train")
    val_data = FFPPDataset(opts, "val")
    train_loader = DataLoader(train_data, batch_size=opts.batch_size, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_data, batch_size=opts.batch_size, shuffle=True, num_workers=8)

    num_params = utils.count_network_parameters(model)
    print('\n=====================================================================')
    print("===> Model has %d parameters" % num_params)
    print('=====================================================================')
    opts.train_epoch_size = len(train_data) / opts.batch_size
    total_iter = 0
    for epoch_id in range(opts.epochs):
        # update learning rate
        current_lr = utils.learning_rate_decay(opts, epoch_id)
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr

        ts = datetime.now()
        for iteration, data in enumerate(train_loader, 1):
            total_iter += 1
            optimizer.zero_grad()
            # loss and metric
            bin_loss_batch = 0.0
            data_time = datetime.now() - ts
            ts = datetime.now()

            # forward
            image, ann_data, label = data
            label = label.long()
            if opts.cuda:
                image = image.cuda()
                ann_data = ann_data.cuda()
                label = label.cuda()

            # Model prediction
            output = model(image)
            bin_loss_batch = crtirion(output, label) # crossentropyloss include softmax
            bin_loss_batch.backward()
            optimizer.step()
            if total_iter % opts.display_ite == 0:
                network_time = datetime.now() - ts
                ### print training info
                info = "Train: [GPU %s]: " % (opts.device)
                info += "Train: Epoch %d; Batch %d / %d; " % (epoch_id, iteration, len(train_loader))
                info += "lr = %s; " % (str(current_lr))
                ## number of samples per second
                batch_freq = opts.batch_size / (data_time.total_seconds() + network_time.total_seconds())
                info += "data loading = %.3f sec, network = %.3f sec, batch = %.3f Hz\n" % (
                    data_time.total_seconds(), network_time.total_seconds(), batch_freq)
                utils.tensor_info(info, output, label, total_iter, opts, loss_writer, bin_loss_batch)

            if total_iter % opts.val_ite == 0:
                model.eval()
                with torch.no_grad():
                    TP = 0.0
                    TN = 0.0
                    FN = 0.0
                    FP = 0.0
                    info = "\tVal: phase: "
                    print(info)
                    for val_data in tqdm(val_loader):
                        data_time = datetime.now() - ts
                        ts = datetime.now()
                        # forward

                        image, ann_data, label = val_data
                        label = label.long()
                        if opts.cuda:
                            image = image.cuda()
                            ann_data = ann_data.cuda()
                            label = label.cuda()
                        # Model prediction
                        output = model(image)
                        _, prediction = torch.max(output, 1)  # argmax
                        prediction = prediction.cpu().numpy()
                        label = label.cpu().numpy()
                        TP += ((prediction == 1) & (label == 1)).sum()
                        # TN    predict 和 label 同时为0
                        TN += ((prediction == 0) & (label == 0)).sum()
                        # FN    predict 0 label 1
                        FN += ((prediction == 0) & (label == 1)).sum()
                        # FP    predict 1 label 0
                        FP += ((prediction == 1) & (label == 0)).sum()
                    p = TP / (TP + FP) if (TP + FP) != 0 else 0
                    r = TP / (TP + FN)
                    F1 = 2 * r * p / (r + p) if TP != 0 else 0
                    acc_batch = (TP + TN) / (TP + TN + FP + FN)
                    ### print training info
                    info = "Val: [GPU %s]: \n" % (opts.device)
                    info += "\t\t%25s = %f\n" % ("Val_acc", acc_batch)
                    info += "\t\t%25s = %f\n" % ("Val_F1", F1)
                    loss_writer.add_scalar('Val_acc', acc_batch, total_iter)
                    loss_writer.add_scalar('Val_F1', F1, total_iter)
                    print(info)
        utils.save_model(model, optimizer, opts, epoch_id)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DeepFake-Pytorch')

    parser.add_argument('--exp_name', type=str, default="bslNo_1", help='the experiment name')
    parser.add_argument('--model', type=str, default="XceptionNet", help='the model name')
    parser.add_argument('--pretrain_path', type=str, default="./pretrain/xception-b5690688.pth", help='pretrain weight path')
    parser.add_argument('--display_ite', type=int, default="2", help='when to display the train info')
    parser.add_argument('--val_ite', type=int, default="1000", help='when to display the val info')

    parser.add_argument('--batch-size', type=int, default=36, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--num_workers', type=int, default=8, metavar='N',
                        help='num_workers default 8')
    parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                        help='number of epochs to train (default: 10000)')
    parser.add_argument('--phase', type=str, default="train", help='phase')
    parser.add_argument('--device', type=str, default="0", help='use device ex. "0,1,2"')
    parser.add_argument('--no_pair', action='store_true',
                        help='load the dataset in the form of fakeA realA pair or random')
    parser.add_argument('--dataset_name', type=str, default="Deepfakes",
                        help='choose from Deepfakes, Face2Face, FaceSwap, NeuralTextures')
    parser.add_argument('--compression', type=str, default="c23", help='choose from raw, c23, c40')
    parser.add_argument('--keepOri', action='store_true', help='debug, no norm')
    parser.add_argument('--checkpoints_dir', type=str, default="./checkpoints", help='where to save the cp')
    parser.add_argument('--which_epoch', type=str, default='latest',
                        help='which epoch to load? set to latest to use latest cached model')

    # lr
    parser.add_argument('--lr_init', type=float, default=1e-4, help='initial learning Rate')
    parser.add_argument('--lr_offset', type=int, default=20,
                        help='epoch to start learning rate drop [-1 = no drop]')
    parser.add_argument('--lr_step', type=int, default=20, help='step size (epoch) to drop learning rate')
    parser.add_argument('--lr_drop', type=float, default=0.5, help='learning rate drop ratio')
    parser.add_argument('--lr_min_m', type=float, default=0.1,
                        help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')

    # optimizer
    parser.add_argument('--solver', type=str, default="ADAM", choices=["SGD", "ADAIM"], help="optimizer")
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
    parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for ADAM')
    parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
    parser.add_argument('--loss', type=str, default="L1", help="optimizer [Options: SGD, ADAM]")
    args = parser.parse_args()
    args.cuda = True if args.device is not None else False

    _main(args)

