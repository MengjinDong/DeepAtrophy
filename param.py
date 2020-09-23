'''
Configs for training & evaluation & testing
Written by Whalechen
'''

import argparse

import torchvision.models as models

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

def parse_opts():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    parser.add_argument(
        '--model-type',
        default=2,
        type=int,
        metavar='N',
        help='model type when loading models' +
             'model_type = 1, use self-defined model architecture' + # change args.arch
             'model_type = 2, use pretrained ResNet model architecture' # change args.model, args.depth
    )

    parser.add_argument(
        '--model',
        default='resnet',
        type=str,
        help='(resnet | preresnet | wideresnet | resnext | densenet | '
    )

    parser.add_argument(
        '--model_depth',
        default=50,
        type=int,
        help='Depth of resnet (10 | 18 | 34 | 50 | 101)'
    )

    parser.add_argument(
        '--num_date_diff_classes',
        default=5, # number of output parameters of the basic sub-network
        type=int,
        help="Number of date difference classes" # for resnet output
    )

    parser.add_argument(
        '--num_reg_labels',
        default = 4,
        type=int,
        help="Number of regression channels. Default 1, since regression gives 1 output"
    )

    parser.add_argument(
        '--num_stages',
        default=8,
        type=int,
        help="Number of stages in diagnosis, including (HC, eMCI, lMCI, and AD) x (Amyloid + and -)"
    )

    parser.add_argument(
        '--input-channels',
        default=2,
        type=float,
        metavar='M',
        help='input channel of network.\n' +
             'input_channels = 1: sample include 2 images, each with one bl or fu image.\n' +
             'input_channels = 2: sample include one image with 2 channels (bl and fu) \n' +
             'input_channels = 3: sample include one image with 3 channels, with the third blank.\n'
    )
    # End of parameters for model_type == 2.

    parser.add_argument(
        '--resume_all',
        default='',
        help='path to the latest checkpoint (default: none)'
    )

    parser.add_argument(
        '--train-stages',
        default="[0, 3, 5]", # "[0, 1, 3, 5]"
        help='Input stages to train the model'
    )

    parser.add_argument(
        '--eval-stages',
        default="[0, 1, 3, 5]",
        help='Input stages to evaluate the model'
    )

    parser.add_argument(
        '--test-stages',
        default="[0, 1, 3, 5]",
        help='Input stages to test the model'
    )

    parser.add_argument(
        '--trt-stages',
        default="[trt]",
        help='Input stages to test the model'
    )

    parser.add_argument(
        '--pretrained',
        dest='pretrained',
        action='store_true',
        # can be store_true (pretrained with no modified architecture),
        # store_false (pretrained network with modified architecture)
        help='use pre-trained model for model 2'
    )

    parser.add_argument(
        '-e', '--evaluate',
        dest='evaluate',
        action='store_true',
        help='option to evaluate before starting training'
    )

    parser.add_argument(
        '-t', '--test',
        dest='test',
        action='store_true',
        help='option to test model after training on test set (not validation set)'
    )

    parser.add_argument(
        '--test-pair',
        action='store_false',
        help='option to test only the basic sub-network after training on test set (not validation set)'
    )

    parser.add_argument(
        '--range-weight',
        default=1,
        type=int,
        metavar='N',
        help='weight of range loss'
    )

    parser.add_argument(
        '--ROOT',
        metavar='DIR',
        default="/home/mdong/Desktop/Longi_T1_final_paper",
        help='path to project'
    )

    parser.add_argument(
        '-b', '--batch-size',
        default=15,
        type=int,  # 300 (for convnet) or 60 (for resnet)
        metavar='N', help='mini-batch size (default: 20)'
    )

    parser.add_argument(
        '--early-stop',
        default=0,
        type=int,
        metavar='N', help='flag whether to do early stopping or not'
    )

    parser.add_argument(
        '-j', '--workers',
        default=4,
        type=int,
        metavar='N',
        help='number of data loading workers (default: 4)'
    )

    parser.add_argument(
        '--lr', '--learning-rate',
        default=0.001,
        type=float,
        metavar='LR',
        help='initial learning rate'
    )

    parser.add_argument(
        '--datapath',
        metavar='DIR',
        default="/media/mdong/MDONG2T/Longi_T1_2GO_final_paper/T1_Input_3d",
        help='path to dataset'
    )

    parser.add_argument(
        '--epochs',
        default=15,
        type=int,
        metavar='N',
        help='number of total epochs to run for the categorical regression'
    )

    parser.add_argument(
        '--momentum',
        default=0.9,
        type=float,
        metavar='M',
        help='momentum'
    )

    parser.add_argument(
        '--get-prec',
        default=1,
        type=float,
        metavar='M',
        help='flag on whether to get precision of training or eval data'
    )

    parser.add_argument(
        '--weight-decay', '--wd',
        default=1e-4,
        type=float,
        metavar='W', help='weight decay (default: 1e-4)'
    )

    parser.add_argument(
        '--print-freq', '-p',
        default=20,
        type=int,
        metavar='N',
        help='print frequency (default: 20)'
    )

    parser.add_argument(
        '--eval-freq',
        default=2,
        type=int,
        metavar='N',
        help='eval frequency (default: 5)'
    )

    parser.add_argument(
        '--world-size',
        default=1,
        type=int,
        help='number of distributed processes'
    )

    parser.add_argument(
        '--dist-url',
        default='tcp://224.66.41.62:23456',
        type=str,
        help='url used to set up distributed training'
    )

    parser.add_argument(
        '--dist-backend',
        default='gloo',
        type=str,
        help='distributed backend'
    )

    parser.add_argument(
        '--max_angle',
        default=15,
        type=int,
        metavar='N',
        help='max angle of rotation when doing data augmentation'
    )

    parser.add_argument(
        '--rotate_prob',
        default=0.5,
        type=int,
        metavar='N',
        help='probability of rotation in each axis when doing data augmentation'
    )

    parser.add_argument(
        '--seed',
        default=None,
        type=int,
        help='seed for initializing training. '
    )

    parser.add_argument(
        '--gpu',
        default=0,
        type=int,
        help='GPU id to use.'
    )

    parser.add_argument(
        '--multiprocessing-distributed',
        action='store_true',
        help='Use multi-processing distributed training to launch '
             'N processes per node, which has N GPUs. This is the '
             'fastest way to use PyTorch for either single node or '
             'multi node data parallel training'
    )

    # [48, 80, 64] -> [24, 40, 32]
    parser.add_argument(
        '--input_D',
    default=48,
        type=int,
        help='Input size of depth'
    )

    parser.add_argument(
        '--input_H',
        default=80,
        type=int,
        help='Input size of height'
    )

    parser.add_argument(
        '--input_W',
        default=64,
        type=int,
        help='Input size of width'
    )

    parser.add_argument(
        '--patience',
        default=20,
        type=int,
        help='Early stopping patience')

    parser.add_argument(
        '--tolerance',
        default=0,
        type=float,
        help='Early stopping tolerance of accuracy')

    parser.add_argument(
        '--new_layer_names',
        default=['conv_seg'],
        type=list,
        help='New layer except for backbone'
    )

    parser.add_argument(
        '--resnet_shortcut',
        default='B',
        type=str,
        help='Shortcut type of resnet (A | B)')

    parser.add_argument(
        '--no_cuda', action='store_true', help='If true, cuda is not used.')

    parser.add_argument(
        '--gpu_id',
        default=[0],
        nargs='+',
        type=int,
        help='Gpu id lists')


    parser.add_argument(
        '--manual_seed', default=1, type=int, help='Manually set random seed'
    )

    args = parser.parse_args()

    args.pretrain_path = "/media/mdong/StorageDevice256/MedicalNet/pretrain/{}_{}.pth".format(args.model, args.model_depth)
    
    return args
