import os
import argparse
import torch
from functions import classification_result
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter


def get_args():
    parser = argparse.ArgumentParser(description='APFed method and baselines')
    parser.add_argument('-f', default='', type=str)

    # Tasks
    parser.add_argument('--alg', type=str, default='tvfl',
                        help='only tvfl, use for mark')
    parser.add_argument('--iter', type=int, default=0,
                        help='the i-th run')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to use (default: mnist)')
    parser.add_argument('--num_views', type=int, default=2,
                        help='the number of data views')
    parser.add_argument('--ith_view', type=int, default=0,
                        help='the i-th view on the active client')
    parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                        help='batch size (default: 256)')
    parser.add_argument('--num_epochs', type=int, default=500,
                        help='number of epochs (default: 500)')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='learning rate for model parameters (default: 1e-3)')
    parser.add_argument('--optim', type=str, default='Adam', choices=['Adam', 'SGD'],
                        help='optimizer to use (default: Adam)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='L2 penalty factor of the optimizers')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='Momentum value for SGD')
    # parser.add_argument('--overlap', type=bool, default=False,
    #                     help='if the splitted images overlap with each other')
    parser.add_argument('--log_dir', type=str, default='./runs',
                        help='log file directory (default: ./runs)')
    args = parser.parse_args()
    return args


class TVFL(object):
    def __init__(self, hp):

        self.hp = hp
        self.name = '{}_{}_nviews{}_{}_bz_{}'.format(hp.alg, hp.dataset, hp.num_views, hp.ith_view, hp.batch_size)

        # active and passive clients
        from utils.clients_tvfl import ActiveClient, PassiveClient
        self.num_passive_clients = self.hp.num_views - 1
        self.active_client = ActiveClient(device=hp.devices[0])
        self.passive_clients = []
        for i in range(self.num_passive_clients):
            self.passive_clients.append(PassiveClient(device=hp.devices[1]))

        self.set_clients()


    def set_clients(self):

        # import nets according to different dataset
        if self.hp.dataset in ['mnist', 'fmnist']:
            from utils.nets_mnist import EncNet, TaskNet
            if self.hp.num_views == 2:
                tmp = 10 * 24
            elif self.hp.num_views == 3:
                tmp = 5 * 24
            else:
                raise NotImplementedError
            d_rep = 64 * tmp
            num_class = 10
        elif self.hp.dataset in ['cifar10', 'cifar100']:
            from utils.nets_cifar import EncNet, TaskNet
            if self.hp.num_views == 2:
                tmp = 5 * 13
            elif self.hp.num_views == 3:
                tmp = 2 * 13
            else:
                raise NotImplementedError
            d_rep = 64 * tmp
            num_class = 10 if self.hp.dataset == 'cifar10' else 100
        elif self.hp.dataset in ['caltech101']:
            from utils.nets_cifar import EncNet, TaskNet
            if self.hp.num_views == 2:
                tmp = 5 * 13
            elif self.hp.num_views == 3:
                tmp = 2 * 13
            else:
                raise NotImplementedError
            d_rep = 64 * tmp
            num_class = 102
        else:
            raise NotImplementedError

        # define the common optim and scheduler
        def get_optim_scheduler(name, net, lr, weight_decay, num_epochs=500):
            params = []
            for _, p in net.named_parameters():
                if p.requires_grad:
                    params.append(p)

            if name.lower() == 'adam':
                optim = torch.optim.Adam(params=params, lr=lr, weight_decay=weight_decay)
            elif name.lower() == 'sgd':
                optim = torch.optim.SGD(params=params, lr=lr, weight_decay=weight_decay, momentum=0.9)
            else:
                raise NotImplementedError

            scheduler = CosineAnnealingLR(optim, T_max=num_epochs)

            return optim, scheduler

        # -----------------------------------
        # set the active client
        active_enc = EncNet()
        optim_ae, scheduler_ae = get_optim_scheduler(self.hp.optim, active_enc, self.hp.lr,
                                                     self.hp.weight_decay, self.hp.num_epochs)
        task_model = TaskNet(d_rep * self.hp.num_views, num_class)
        optim_tm, scheduler_tm = get_optim_scheduler(self.hp.optim, task_model, self.hp.lr,
                                                     self.hp.weight_decay, self.hp.num_epochs)
        self.active_client.set_model([active_enc, optim_ae, scheduler_ae],
                                     [task_model, optim_tm, scheduler_tm])
        self.active_client.set_loss_fun(torch.nn.NLLLoss(reduction='mean'))

        # -----------------------------------
        # set the passive clients
        for i in range(self.num_passive_clients):
            passive_enc = EncNet()
            optim_pe, scheduler_pe = get_optim_scheduler(self.hp.optim, passive_enc, self.hp.lr,
                                                         self.hp.weight_decay, self.hp.num_epochs)
            self.passive_clients[i].set_model([passive_enc, optim_pe, scheduler_pe])

    def train_and_eval(self, train_loader, test_loader):

        print('# {}'.format(self.name))

        writer_norm = SummaryWriter('./runs/iter_{}/{}_{}'.format(self.hp.iter, self.name, 'norm'))
        writer_zero = SummaryWriter('./runs/iter_{}/{}_{}'.format(self.hp.iter, self.name, 'zero'))
        writer_mean = SummaryWriter('./runs/iter_{}/{}_{}'.format(self.hp.iter, self.name, 'mean'))
        writer_rand = SummaryWriter('./runs/iter_{}/{}_{}'.format(self.hp.iter, self.name, 'rand'))

        for epoch in range(1, self.hp.num_epochs+1):
            loss_task = 0
            preds, ys = [], []
            for i_batch, batch_data in enumerate(train_loader):
                data, y = batch_data

                # set the data of active and passive clients
                self.active_client.set_data(data=data[self.hp.ith_view], y=y)
                del data[self.hp.ith_view]
                for i in range(self.num_passive_clients):
                    self.passive_clients[i].set_data(data[i])

                # 1. active client computes representations
                self.active_client.compute_rep()

                # 2. all passive clients compute representations
                reps_passive = []
                for i in range(self.num_passive_clients):
                    reps_passive.append(self.passive_clients[i].compute_rep())

                # 3. send reps to active client and train to get gradients
                grads_out, loss_task_tmp, pred = self.active_client.train(reps_passive)

                # 4. send gradients to passive clients and update the encoders
                for i in range(self.num_passive_clients):
                    self.passive_clients[i].train(grad=grads_out[i])

                preds.append(pred)
                ys.append(y)
                loss_task += loss_task_tmp

            preds = torch.cat(preds)
            ys = torch.cat(ys)
            results_train = classification_result(ys, preds)

            # measure on test set
            results_norm, results_zero, results_mean, results_rand = self.eval(test_loader)

            print('- {:0>3d}, loss: {:.4f}, acc_train: {:.2f}%, acc_norm: {:.2f}%'.format(epoch, loss_task,
                                                                                          results_train[0],
                                                                                          results_norm[0]))

            def log_with_writer(writer, loss, results_train, results_test, epoch):
                writer.add_scalar('loss', loss, epoch)

                writer.add_scalar('acc_train', results_train[0], epoch)
                writer.add_scalar('precision_macro_train', results_train[1], epoch)
                writer.add_scalar('recall_macro_train', results_train[2], epoch)
                writer.add_scalar('f1_macro_train', results_train[3], epoch)
                writer.add_scalar('precision_weighted_train', results_train[4], epoch)
                writer.add_scalar('recall_weighted_train', results_train[5], epoch)
                writer.add_scalar('f1_weighted_train', results_train[6], epoch)

                writer.add_scalar('acc_test', results_test[0], epoch)
                writer.add_scalar('precision_macro_test', results_test[1], epoch)
                writer.add_scalar('recall_macro_test', results_test[2], epoch)
                writer.add_scalar('f1_macro_test', results_test[3], epoch)
                writer.add_scalar('precision_weighted_test', results_test[4], epoch)
                writer.add_scalar('recall_weighted_test', results_test[5], epoch)
                writer.add_scalar('f1_weighted_test', results_test[6], epoch)

            log_with_writer(writer_norm, loss_task, results_train, results_norm, epoch)
            log_with_writer(writer_zero, loss_task, results_train, results_zero, epoch)
            log_with_writer(writer_mean, loss_task, results_train, results_mean, epoch)
            log_with_writer(writer_rand, loss_task, results_train, results_rand, epoch)

        writer_norm.close()
        writer_zero.close()
        writer_mean.close()
        writer_rand.close()

        self.save_model()

    def eval(self, test_loader):
        # evaluate
        preds_norm, preds_zero, preds_mean, preds_rand, ys = [], [], [], [], []
        for i_batch, batch_data in enumerate(test_loader):
            data, y = batch_data
            # set the data of active and passive clients
            self.active_client.set_data(data=data[self.hp.ith_view], y=y)
            del data[self.hp.ith_view]
            for i in range(self.num_passive_clients):
                self.passive_clients[i].set_data(data[i])

            # 1. active client computes representations
            rep_active = self.active_client.compute_rep()

            ## norm
            # 2. all passive clients compute representations
            reps_passive_norm = []
            for i in range(self.num_passive_clients):
                reps_passive_norm.append(self.passive_clients[i].compute_rep())
            # 3. send reps to active client and get predictions
            pred = self.active_client.predict(reps_passive_norm)
            preds_norm.append(pred)

            ## zero fill
            # 2. all passive clients compute representations
            reps_passive_zero = []
            for i in range(self.num_passive_clients):
                reps_passive_zero.append(torch.zeros_like(reps_passive_norm[i]))
            # 3. send reps to active client and get predictions
            pred = self.active_client.predict(reps_passive_zero)
            preds_zero.append(pred)

            ## mean fill
            # 2. all passive clients compute representations
            reps_passive_mean = []
            mean = torch.mean(rep_active)
            for i in range(self.num_passive_clients):
                reps_passive_mean.append(torch.ones_like(reps_passive_norm[i]) * mean)
            # 3. send reps to active client and get predictions
            pred = self.active_client.predict(reps_passive_mean)
            preds_mean.append(pred)

            ## rand fill
            # 2. all passive clients compute representations
            reps_passive_rand = []
            max, min = torch.max(rep_active), torch.min(rep_active)
            for i in range(self.num_passive_clients):
                reps_passive_rand.append(torch.rand_like(reps_passive_norm[i]) * (max-min) + min)
            # 3. send reps to active client and get predictions
            pred = self.active_client.predict(reps_passive_rand)
            preds_rand.append(pred)

            ys.append(y)


        ys = torch.cat(ys)

        ## norm
        preds_norm = torch.cat(preds_norm)
        results_norm = classification_result(ys, preds_norm)

        ## zero
        preds_zero = torch.cat(preds_zero)
        results_zero = classification_result(ys, preds_zero)

        ## mean
        preds_mean = torch.cat(preds_mean)
        results_mean = classification_result(ys, preds_mean)

        ## rand
        preds_rand = torch.cat(preds_rand)
        results_rand = classification_result(ys, preds_rand)

        return results_norm, results_zero, results_mean, results_rand


    def save_model(self):
        save_dir = './models/iter_{}/{}/'.format(self.hp.iter, self.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # active client
        torch.save(self.active_client.encoder.state_dict(), save_dir + 'active_enc.pt')
        torch.save(self.active_client.task_model.state_dict(), save_dir + 'task_model.pt')
        # passive clients
        for i in range(self.num_passive_clients):
            torch.save(self.passive_clients[i].encoder.state_dict(), save_dir + 'passive_enc_{}.pt'.format(i))
        return






