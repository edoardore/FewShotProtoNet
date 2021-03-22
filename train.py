import numpy as np
import torch
import torch.optim
import glob
import configs
import backbone
from data.datamgr import SetDataManager
from protonet import ProtoNet
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def train(base_loader, val_loader, model, start_epoch, stop_epoch, checkpoint_dir, save_freq):
    optimizer = torch.optim.Adam(model.parameters())
    max_acc = 0

    for epoch in range(start_epoch, stop_epoch):
        model.train()
        model.train_loop(epoch, base_loader, optimizer)
        model.eval()

        acc = model.test_loop(val_loader)
        if acc > max_acc:
            print("Saving best model!")
            max_acc = acc
            outfile = os.path.join(checkpoint_dir, 'best_model.tar')
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

        if (epoch % save_freq == 0) or (epoch == stop_epoch - 1):
            outfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(epoch))
            torch.save({'epoch': epoch, 'state': model.state_dict()}, outfile)

    return model


if __name__ == '__main__':
    model_dict = dict(
        Conv4=backbone.Conv4,
        Conv4S=backbone.Conv4S,
        Conv6=backbone.Conv6,
        ResNet10=backbone.ResNet10,
        ResNet18=backbone.ResNet18,
        ResNet34=backbone.ResNet34)
    # .Conv4, .Conv6, .ResNet10, .ResNet18, .ResNet34
    model = 'Conv6'

    # CUB, omniglot, cross_char
    dataset = 'CUB'

    # class num to classify for training
    train_n_way = 5

    # class num to classify for testing (validation)
    test_n_way = 5

    # number of labeled data in each class, same as n_support
    n_shot = 1

    # perform data augmentation or not during training
    train_aug = True

    # Save frequency
    save_freq = 50

    # Starting epoch
    start_epoch = 0

    # Stopping epoch
    stop_epoch = -1

    # continue from previous trained model with largest epoch
    resume = False

    np.random.seed(10)
    if dataset == 'cross':
        base_file = configs.data_dir['miniImagenet'] + 'all.json'
        val_file = configs.data_dir['CUB'] + 'val.json'
    elif dataset == 'cross_char':
        base_file = configs.data_dir['omniglot'] + 'noLatin.json'
        val_file = configs.data_dir['emnist'] + 'val.json'
    else:
        base_file = configs.data_dir[dataset] + 'base.json'
        val_file = configs.data_dir[dataset] + 'val.json'

    if 'Conv' in model:
        if dataset in ['omniglot', 'cross_char']:
            image_size = 28
        else:
            image_size = 84
    else:
        image_size = 224

    if dataset in ['omniglot', 'cross_char']:
        assert model == 'Conv4' and not train_aug, 'omniglot only support Conv4 without augmentation'
        model = 'Conv4S'

    if stop_epoch == -1:
        if n_shot == 1:
            stop_epoch = 600
        elif n_shot == 5:
            stop_epoch = 400
        else:
            stop_epoch = 600  # default

    n_query = max(1, int(
        16 * test_n_way / train_n_way))
    train_few_shot_params = dict(n_way=train_n_way, n_support=n_shot)
    base_datamgr = SetDataManager(image_size, n_query=n_query, **train_few_shot_params)
    base_loader = base_datamgr.get_data_loader(base_file, aug=train_aug)
    test_few_shot_params = dict(n_way=test_n_way, n_support=n_shot)
    val_datamgr = SetDataManager(image_size, n_query=n_query, **test_few_shot_params)
    val_loader = val_datamgr.get_data_loader(val_file, aug=False)
    modelStr = model
    model = ProtoNet(model_dict[model], **train_few_shot_params)

    model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, dataset, modelStr, 'protonet')
    if train_aug:
        checkpoint_dir += '_aug'
    checkpoint_dir += '_%dway_%dshot' % (train_n_way, n_shot)

    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    if resume:
        filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
        if len(filelist) != 0:
            filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
            epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
            max_epoch = np.max(epochs)
            resume_file = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
        else:
            resume_file = None
        if resume_file is not None:
            tmp = torch.load(resume_file)
            start_epoch = tmp['epoch'] + 1
            model.load_state_dict(tmp['state'])

    model = train(base_loader, val_loader, model, start_epoch, stop_epoch, checkpoint_dir, save_freq)
