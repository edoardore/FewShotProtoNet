import torch
from torch.autograd import Variable
import numpy as np
import glob
import os
import h5py
import configs
import backbone
from data.datamgr import SimpleDataManager

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def save_features(model, data_loader, outfile):
    f = h5py.File(outfile, 'w')
    max_count = len(data_loader) * data_loader.batch_size
    all_labels = f.create_dataset('all_labels', (max_count,), dtype='i')
    all_feats = None
    count = 0
    for i, (x, y) in enumerate(data_loader):
        if i % 10 == 0:
            print('{:d}/{:d}'.format(i, len(data_loader)))
        x = x.cuda()
        x_var = Variable(x)
        feats = model(x_var)
        if all_feats is None:
            all_feats = f.create_dataset('all_feats', [max_count] + list(feats.size()[1:]), dtype='f')
        all_feats[count:count + feats.size(0)] = feats.data.cpu().numpy()
        all_labels[count:count + feats.size(0)] = y.cpu().numpy()
        count = count + feats.size(0)

    count_var = f.create_dataset('count', (1,), dtype='i')
    count_var[0] = count

    f.close()





if __name__ == '__main__':
    model_dict = dict(
        Conv4=backbone.Conv4,
        Conv4S=backbone.Conv4S,
        Conv6=backbone.Conv6,
        ResNet10=backbone.ResNet10,
        ResNet18=backbone.ResNet18,
        ResNet34=backbone.ResNet34)

    # .Conv4, .Conv6, .ResNet10, .ResNet18, .ResNet34
    model = 'ResNet34'

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

    # base/val/novel
    split = 'novel'

    # save feature from the model trained in x epoch, use the best model if x is -1
    save_iter = -1

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

    if dataset == 'cross':
        if split == 'base':
            loadfile = configs.data_dir['miniImagenet'] + 'all.json'
        else:
            loadfile = configs.data_dir['CUB'] + split + '.json'
    elif dataset == 'cross_char':
        if split == 'base':
            loadfile = configs.data_dir['omniglot'] + 'noLatin.json'
        else:
            loadfile = configs.data_dir['emnist'] + split + '.json'
    else:
        loadfile = configs.data_dir[dataset] + split + '.json'

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, dataset, model, 'protonet')
    if train_aug:
        checkpoint_dir += '_aug'
    checkpoint_dir += '_%dway_%dshot' % (train_n_way, n_shot)

    if save_iter != -1:
        modelfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(save_iter))
    else:
        best_file = os.path.join(checkpoint_dir, 'best_model.tar')
        if os.path.isfile(best_file):
            modelfile = best_file
        else:
            filelist = glob.glob(os.path.join(checkpoint_dir, '*.tar'))
            if len(filelist) == 0:
                modelfile= None
            else:
                filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
                epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
                max_epoch = np.max(epochs)
                modelfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))

    if save_iter != -1:
        outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"),
                               split + "_" + str(save_iter) + ".hdf5")
    else:
        outfile = os.path.join(checkpoint_dir.replace("checkpoints", "features"), split + ".hdf5")

    datamgr = SimpleDataManager(image_size, batch_size=64)
    data_loader = datamgr.get_data_loader(loadfile, aug=False)

    model = model_dict[model]()

    model = model.cuda()
    tmp = torch.load(modelfile)
    state = tmp['state']
    state_keys = list(state.keys())
    for i, key in enumerate(state_keys):
        if "feature." in key:
            newkey = key.replace("feature.",
                                 "")
            state[newkey] = state.pop(key)
        else:
            state.pop(key)

    model.load_state_dict(state)
    model.eval()
    dirname = os.path.dirname(outfile)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    save_features(model, data_loader, outfile)
