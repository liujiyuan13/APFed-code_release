from TVFL import *
from data import get_loader
from functions import set_seed

import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
    args = get_args()

    args.devices = ["cuda:0", "cuda:0"]
    # args.devices = ["cpu", "cpu"]

    args.alg = 'tvfl'
    args.num_epochs = 300
    args.optim = 'SGD'  # or Adam

    iters = 5
    datasets = ['mnist', 'fmnist', 'cifar10', 'cifar100', 'caltech101']
    num_views_set = [2, 3]
    batch_size = 128

    for iter in range(iters):
        set_seed(iter)
        for dataset in datasets:
            for num_views in num_views_set:
                for ith_view in range(num_views):

                    args.iter, args.dataset, args.num_views, args.ith_view = iter, dataset, num_views, ith_view
                    args.batch_size = batch_size

                    train_loader = get_loader(args.dataset, args.batch_size, args.num_views, train=True)
                    test_loader = get_loader(args.dataset, args.batch_size, args.num_views, train=False)

                    # run the algorithm
                    solver = TVFL(args)
                    solver.train_and_eval(train_loader, test_loader)

                    del solver