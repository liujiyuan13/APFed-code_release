from APFed import *
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

    args.alg = 'apfedc'
    args.optim = 'SGD' # or Adam
    args.num_epochs = 300

    iters = 5
    datasets = ['mnist', 'fmnist', 'cifar10', 'cifar100', 'caltech101']
    num_views_set = [2, 3]
    trade_offs = [0, 0.1, 1, 10]
    batch_sizes = [32, 64, 128, 256, 512, 1024]

    for iter in range(iters):
        set_seed(iter)
        for dataset in datasets:
            for num_views in num_views_set:
                for ith_view in range(num_views):
                    for trade_off in trade_offs:
                        for batch_size in batch_sizes:

                            args.iter, args.dataset, args.num_views, args.ith_view = iter, dataset, num_views, ith_view
                            args.active_alone = True if trade_off == 0 else False
                            args.trade_off, args.batch_size = trade_off, batch_size

                            train_loader = get_loader(args.dataset, args.batch_size, args.num_views, train=True)
                            test_loader = get_loader(args.dataset, args.batch_size, args.num_views, train=False)

                            # run the algorithm
                            solver = APFed(args)
                            solver.train_and_eval(train_loader, test_loader)

                            del solver