import torch
import numpy as np
import torch.optim
import torch.utils.data.sampler
import os
import random
import configs
import backbone
import data.feature_loader as feat_loader
from protonet import ProtoNet
import glob
from matplotlib import pyplot as plt
from matplotlib.ticker import PercentFormatter

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def feature_evaluation(cl_data_file, model, n_way=5, n_support=5, n_query=15, adaptation=False):
    class_list = cl_data_file.keys()

    select_class = random.sample(class_list, n_way)
    z_all = []
    for cl in select_class:
        img_feat = cl_data_file[cl]
        perm_ids = np.random.permutation(len(img_feat)).tolist()
        z_all.append([np.squeeze(img_feat[perm_ids[i]]) for i in range(n_support + n_query)])  # stack each batch

    z_all = torch.from_numpy(np.array(z_all))

    model.n_query = n_query
    if adaptation:
        scores = model.set_forward_adaptation(z_all, is_feature=True)
    else:
        scores = model.set_forward(z_all, is_feature=True)
    pred = scores.data.cpu().numpy().argmax(axis=1)
    y = np.repeat(range(n_way), n_query)
    acc = np.mean(pred == y) * 100
    return acc


def test(model, n_shot, train_aug, dataset, iter_num, adaptation):
    model_dict = dict(
        Conv4=backbone.Conv4,
        Conv4S=backbone.Conv4S,
        Conv6=backbone.Conv6,
        ResNet10=backbone.ResNet10,
        ResNet18=backbone.ResNet18,
        ResNet34=backbone.ResNet34)

    # class num to classify for training
    train_n_way = 5

    # class num to classify for testing (validation)
    test_n_way = 5

    # base/val/novel
    split = 'novel'

    # save feature from the model trained in x epoch, use the best model if x is -1
    save_iter = -1

    acc_all = []

    few_shot_params = dict(n_way=test_n_way, n_support=n_shot)

    if dataset in ['omniglot', 'cross_char']:
        assert model == 'Conv4' and not train_aug, 'omniglot only support Conv4 without augmentation'
        model = 'Conv4S'

    modelStr = model
    model = ProtoNet(model_dict[model], **few_shot_params)

    model = model.cuda()

    checkpoint_dir = '%s/checkpoints/%s/%s_%s' % (configs.save_dir, dataset, modelStr, 'protonet')
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
                modelfile = None
            else:
                filelist = [x for x in filelist if os.path.basename(x) != 'best_model.tar']
                epochs = np.array([int(os.path.splitext(os.path.basename(x))[0]) for x in filelist])
                max_epoch = np.max(epochs)
                modelfile = os.path.join(checkpoint_dir, '{:d}.tar'.format(max_epoch))
    if modelfile is not None:
        tmp = torch.load(modelfile)
        model.load_state_dict(tmp['state'])

    if save_iter != -1:
        split_str = split + "_" + str(save_iter)
    else:
        split_str = split

    novel_file = os.path.join(checkpoint_dir.replace("checkpoints", "features"),
                              split_str + ".hdf5")  # defaut split = novel, but you can also test base or val classes
    cl_data_file = feat_loader.init_loader(novel_file)

    for i in range(iter_num):
        acc = feature_evaluation(cl_data_file, model, n_query=15, adaptation=adaptation, **few_shot_params)
        acc_all.append(acc)
    return acc_all


def makeTable(headerRow, columnizedData, columnSpacing=2):
    from numpy import array, max, vectorize

    cols = array(columnizedData, dtype=str)
    colSizes = [max(vectorize(len)(col)) for col in cols]

    header = ''
    rows = ['' for i in cols[0]]

    for i in range(0, len(headerRow)):
        if len(headerRow[i]) > colSizes[i]: colSizes[i] = len(headerRow[i])
        headerRow[i] += ' ' * (colSizes[i] - len(headerRow[i]))
        header += headerRow[i]
        if not i == len(headerRow) - 1: header += ' ' * columnSpacing

        for j in range(0, len(cols[i])):
            if len(cols[i][j]) < colSizes[i]:
                cols[i][j] += ' ' * (colSizes[i] - len(cols[i][j]) + columnSpacing)
            rows[j] += cols[i][j]
            if not i == len(headerRow) - 1: rows[j] += ' ' * columnSpacing

    line = '-' * len(header)
    print(line)
    print(header)
    print(line)
    for row in rows: print(row)
    print(line)


def testCUB():
    iter_num = 600
    acc_means = []
    acc_confidences = []
    # perform data augmentation or not during training
    aug = [False, True]
    # number of labeled data in each class, same as n_support
    shot = [1, 5]
    models = ['Conv4', 'Conv6', 'ResNet10', 'ResNet18', 'ResNet34']
    for train_aug in aug:
        for n_shot in shot:
            for model in models:
                if train_aug:
                    print('Test with augmentiation, ' + model + ', n_shot = ' + str(n_shot))
                else:
                    print('Test without augmentiation, ' + model + ', n_shot = ' + str(n_shot))
                acc_all = test(model, n_shot, train_aug, 'CUB', iter_num, False)
                acc_all = np.asarray(acc_all)
                acc_mean = np.mean(acc_all)
                acc_std = np.std(acc_all)
                acc_confidence = 1.96 * acc_std / np.sqrt(iter_num)
                print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, acc_confidence))
                acc_means.append(round(acc_mean, 2))
                acc_confidences.append(round(acc_confidence, 2))

    fig, ax = plt.subplots()
    ax.plot(models, acc_means[0:5], color='orange', linestyle='solid', marker='o',
            markerfacecolor='orange', markersize=9, linewidth=2)
    ax.set(xlabel='Backbone Depth', ylabel='Accuracy', title='ProtoNet, CUB, No Augmentation, 5W-1S')
    ax.grid()
    ax.yaxis.set_major_formatter(PercentFormatter())
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(models, acc_means[5:10], color='orange', linestyle='solid', marker='o',
            markerfacecolor='orange', markersize=9, linewidth=2)
    ax.set(xlabel='Backbone Depth', ylabel='Accuracy', title='ProtoNet, CUB, No Augmentation, 5W-5S')
    ax.grid()
    ax.yaxis.set_major_formatter(PercentFormatter())
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(models, acc_means[10:15], color='orange', linestyle='solid', marker='o',
            markerfacecolor='orange', markersize=9, linewidth=2)
    ax.set(xlabel='Backbone Depth', ylabel='Accuracy', title='ProtoNet, CUB, With Augmentation, 5W-1S')
    ax.grid()
    ax.yaxis.set_major_formatter(PercentFormatter())
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(models, acc_means[15:20], color='orange', linestyle='solid', marker='o',
            markerfacecolor='orange', markersize=9, linewidth=2)
    ax.set(xlabel='Backbone Depth', ylabel='Accuracy', title='ProtoNet, CUB, With Augmentation, 5W-5S')
    ax.grid()
    ax.yaxis.set_major_formatter(PercentFormatter())
    plt.show()

    header = ['Backbone:', 'Conv4', 'Conv6', 'ResNet10', 'ResNet18', 'ResNet34']
    names = ['CUB 5W-1S NoAug', 'CUB 5W-5S NoAug', 'CUB 5W-1S Aug', 'CUB 5W-5S Aug']
    Conv4 = [str(acc_means[0]) + ' +- ' + str(acc_confidences[0]), str(acc_means[5]) + ' +- ' + str(acc_confidences[5]),
             str(acc_means[10]) + ' +- ' + str(acc_confidences[10]),
             str(acc_means[15]) + ' +- ' + str(acc_confidences[15])]
    Conv6 = [str(acc_means[1]) + ' +- ' + str(acc_confidences[1]), str(acc_means[6]) + ' +- ' + str(acc_confidences[6]),
             str(acc_means[11]) + ' +- ' + str(acc_confidences[11]),
             str(acc_means[16]) + ' +- ' + str(acc_confidences[16])]
    ResNet10 = [str(acc_means[2]) + ' +- ' + str(acc_confidences[2]),
                str(acc_means[7]) + ' +- ' + str(acc_confidences[7]),
                str(acc_means[12]) + ' +- ' + str(acc_confidences[12]),
                str(acc_means[17]) + ' +- ' + str(acc_confidences[17])]
    ResNet18 = [str(acc_means[3]) + ' +- ' + str(acc_confidences[3]),
                str(acc_means[8]) + ' +- ' + str(acc_confidences[8]),
                str(acc_means[13]) + ' +- ' + str(acc_confidences[13]),
                str(acc_means[18]) + ' +-' + str(acc_confidences[18])]
    ResNet34 = [str(acc_means[4]) + ' +- ' + str(acc_confidences[4]),
                str(acc_means[9]) + ' +- ' + str(acc_confidences[9]),
                str(acc_means[14]) + ' +- ' + str(acc_confidences[14]),
                str(acc_means[19]) + ' +- ' + str(acc_confidences[19])]
    makeTable(header, [names, Conv4, Conv6, ResNet10, ResNet18, ResNet34])


def testOmniglotAndCross():
    iter_num = 600
    acc_means = []
    acc_confidences = []
    # number of labeled data in each class, same as n_support
    shot = [1, 5]
    datasets = ['omniglot', 'cross_char']
    model = 'Conv4'
    for dataset in datasets:
        for n_shot in shot:
            print('Test ' + str(dataset) + ', ' + str(model) + ', n_shot = ' + str(n_shot))
            acc_all = test(model, n_shot, False, dataset, iter_num, False)
            acc_all = np.asarray(acc_all)
            acc_mean = np.mean(acc_all)
            acc_std = np.std(acc_all)
            acc_confidence = 1.96 * acc_std / np.sqrt(iter_num)
            print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, acc_confidence))
            acc_means.append(round(acc_mean, 2))
            acc_confidences.append(round(acc_confidence, 2))

    header = ['Omni 5W-1S', 'Omni 5W-5S', 'Omni->Emnist 5W-1S', 'Omni->Emnist 5W-5S']
    makeTable(header, [[str(acc_means[0]) + ' +- ' + str(acc_confidences[0])],
                       [str(acc_means[1]) + ' +- ' + str(acc_confidences[1])],
                       [str(acc_means[2]) + ' +- ' + str(acc_confidences[2])],
                       [str(acc_means[3]) + ' +- ' + str(acc_confidences[3])]])


def testFurtherAdaptation():
    iter_num = 600
    acc_means = []
    acc_confidences = []
    # number of labeled data in each class, same as n_support
    n_shot = 5
    dataset = 'CUB'
    model = 'ResNet18'
    adaptations = [False, True]
    for adaptation in adaptations:
        print('Test ' + str(dataset) + ', ' + str(model) + ', n_shot = ' + str(n_shot) + ', adaptation = ' + str(
            adaptation))
        acc_all = test(model, n_shot, True, dataset, iter_num, adaptation)
        acc_all = np.asarray(acc_all)
        acc_mean = np.mean(acc_all)
        acc_std = np.std(acc_all)
        acc_confidence = 1.96 * acc_std / np.sqrt(iter_num)
        print('%d Test Acc = %4.2f%% +- %4.2f%%' % (iter_num, acc_mean, acc_confidence))
        acc_means.append(round(acc_mean, 2))
        acc_confidences.append(round(acc_confidence, 2))

    header = ['CUB ResNet18 5W-5S NoAdaptation', 'CUB ResNet18 5W-5S Adaptation']
    makeTable(header, [[str(acc_means[0]) + ' +- ' + str(acc_confidences[0])],
                       [str(acc_means[1]) + ' +- ' + str(acc_confidences[1])]])

    objects = ('Without\n Adapt.', 'With\n Adapt.')
    y_pos = np.arange(len(objects))
    plt.barh(y_pos, [acc_means[0], 0], align='center', alpha=1)
    plt.barh(y_pos, [0, acc_means[1]], align='center', alpha=1)
    plt.yticks(y_pos, objects)
    plt.xlabel('Accuracy')
    plt.title('CUB, 5-Shot, ResNet18')
    plt.grid()
    plt.show()



testOmniglotAndCross()
testCUB()
testFurtherAdaptation()