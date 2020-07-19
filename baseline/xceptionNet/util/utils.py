import glob,pickle
import os
import re
import numpy as np
import torch
import math
import datetime
from model.xception import xception

def findLatestEpoch(opts):
    ### resume latest model
    name_list = glob.glob(os.path.join(opts.checkpoints_dir, opts.exp_name, "model_*.pth"))
    epoch_st = 0
    if len(name_list) > 0:
        epoch_list = []
        for name in name_list:
            s = re.findall(r'\d+', os.path.basename(name))[0]
            epoch_list.append(int(s))

        epoch_list.sort()
        epoch_st = epoch_list[-1]

    if epoch_st > 0:
        print('=====================================================================')
        print('===> Resuming model from epoch %d' % epoch_st)
        print('=====================================================================')
    return epoch_st


def load_model(model, optimizer=None, opts=None, epoch=None):
    # load model
    if opts.isTrain:
        model_filename = os.path.join(opts.model_dir, "model_epoch_%d.pth" % (epoch))
    else:
        model_filename = os.path.join(opts.model_dir, "model_epoch_%d.pth" % (epoch))
    print("Load %s" % model_filename)
    state_dict = torch.load(model_filename)

    if opts.isTrain:
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        ### move optimizer state to GPU
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()
    else:
        new_dict = {}
        model_state_dict = list(model.state_dict().keys())
        tmp_chek_layer_list = list(state_dict['model'].keys())
        for i in range(len(model_state_dict)):
            layer_name = model_state_dict[i]
            # if ("VGG" not in ch_layer_name) and ("Flow" not in ch_layer_name):
            new_dict[layer_name] = state_dict['model'][layer_name]
        model.load_state_dict(new_dict)


def count_network_parameters(model):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    N = sum([np.prod(p.size()) for p in parameters])
    return N


def learning_rate_decay(opts, epoch):
    ###             1 ~ offset              : lr_init
    ###        offset ~ offset + step       : lr_init * drop^1
    ### offset + step ~ offset + step * 2   : lr_init * drop^2
    ###              ...

    if opts.lr_drop == 0:  # constant learning rate
        decay = 0
    else:
        assert (opts.lr_step > 0)
        decay = math.floor(float(epoch) / opts.lr_step)
        decay = max(decay, 0)  ## decay = 1 for the first lr_offset iterations

    lr = opts.lr_init * math.pow(opts.lr_drop, decay)
    lr = max(lr, opts.lr_init * opts.lr_min)

    return lr


def loadPretrainModel(path, donwload=False, num_out_classes=2, dropout=0.0):
    model = xception(pretrained=donwload)
    state_dict = torch.load(path)
    for name, weights in state_dict.items():
        if 'pointwise' in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    model.load_state_dict(state_dict)
    model.last_linear = model.fc
    del model.fc
    num_ftrs = model.last_linear.in_features
    if not dropout:
        model.last_linear = torch.nn.Linear(num_ftrs, num_out_classes)
    else:
        print('Using dropout', dropout)
        model.last_linear = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(num_ftrs, num_out_classes)
        )
    return model

def tensor_info(info, output, label, total_iter, opts, loss_writer, bin_loss_batch):
    # Cast to desired
    post_func = torch.nn.Softmax(dim=1)
    output = post_func(output)

    _, prediction = torch.max(output, 1)  # argmax
    prediction = prediction.cpu().numpy()
    label = label.cpu().numpy()
    # acc_batch = sum(prediction == label) / opts.batch_size
    # TP    predict 和 label 同时为1
    TP = ((prediction == 1) & (label == 1)).sum()
    # TN    predict 和 label 同时为0
    TN = ((prediction == 0) & (label == 0)).sum()
    # FN    predict 0 label 1
    FN = ((prediction == 0) & (label == 1)).sum()
    # FP    predict 1 label 0
    FP = ((prediction == 1) & (label == 0)).sum()
    p = TP / (TP + FP) if (TP + FP) != 0 else 0
    r = TP / (TP + FN)
    F1 = 2 * r * p / (r + p) if TP != 0 else 0
    acc_batch = (TP + TN) / (TP + TN + FP + FN)

    info += "\tmodel = %s\n" % opts.model

    loss_writer.add_scalar('acc', acc_batch, total_iter)
    info += "\t\t%25s = %f\n" % ("acc", acc_batch)

    loss_writer.add_scalar('loss', bin_loss_batch.item(), total_iter)
    info += "\t\t%25s = %f\n" % ("loss", bin_loss_batch.item())

    loss_writer.add_scalar('F1', F1, total_iter)
    info += "\t\t%25s = %f\n" % ("F1", F1)
    print(info)

def save_model(model, optimizer, opts, epoch_id):
    # save opts
    opts_filename = os.path.join(opts.model_dir, "opts.pth")
    print("Save %s" % opts_filename)
    with open(opts_filename, 'wb') as f:
        pickle.dump(opts, f)

    # serialize model and optimizer to dict
    state_dict = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    model_filename = os.path.join(opts.model_dir, "model_epoch_%d.pth" % (epoch_id))
    print("Save %s" % model_filename)
    torch.save(state_dict, model_filename)
