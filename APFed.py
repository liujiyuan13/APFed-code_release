import os
import argparse
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from functions import contrastive_loss, classification_result

def get_args():
    parser = argparse.ArgumentParser(description='APFed method and baselines')
    parser.add_argument('-f', default='', type=str)

    # Tasks
    parser.add_argument('--alg', type=str, default='apfedc',
                        help='apfedr, apfedc')
    parser.add_argument('--iter', type=int, default=0,
                        help='the i-th run')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to use (default: mnist)')
    parser.add_argument('--num_views', type=int, default=2,
                        help='the number of data views')
    parser.add_argument('--ith_view', type=int, default=0,
                        help='the i-th view on the active client')
    parser.add_argument('--active_alone', type=bool, default=True,
                        help='training the active client without passive clients')
    parser.add_argument('--active_dec', type=bool, default=False,
                        help='using decoder in the active client')
    parser.add_argument('--trade_off', type=float, default=1,
                        help='weight parameter of combining the gradients from passive clients')
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


class APFed(object):
    def __init__(self, hp):

        self.hp = hp

        # import clients according to different alg
        if self.hp.alg == 'apfedc':
            from utils.clients_apfedc import ActiveClient, PassiveClient
            self.name = '{}_{}_nviews{}_{}_ACTalone{}_tradeoff_{}_bz_{}'.format(
                hp.alg, hp.dataset, hp.num_views, hp.ith_view, int(hp.active_alone), hp.trade_off, hp.batch_size)
        elif self.hp.alg == 'apfedr':
            from utils.clients_apfedr import ActiveClient, PassiveClient
            self.name = '{}_{}_nviews{}_{}_ACTalone{}_ACTdec{}_tradeoff_{}_bz_{}'.format(
                hp.alg, hp.dataset, hp.num_views, hp.ith_view, int(hp.active_alone),  int(hp.active_dec), hp.trade_off, hp.batch_size)
        else:
            raise NotImplementedError

        self.log_path = '{}/iter_{}/{}'.format(self.hp.log_dir, self.hp.iter, self.name)

        # init clients
        self.num_passive_clients = self.hp.num_views - 1
        self.active_client = ActiveClient(device=hp.devices[0])
        self.passive_clients = []
        for i in range(self.num_passive_clients):
            self.passive_clients.append(PassiveClient(device=hp.devices[1]))

        self.set_clients()

    def set_clients(self):

        # import nets according to different dataset
        if self.hp.dataset in ['mnist', 'fmnist']:
            from utils.nets_mnist import EncNet, TaskNet, Proj_Head, DecNet
            if self.hp.num_views == 2:
                tmp = 10 * 24
            elif self.hp.num_views == 3:
                tmp = 5 * 24
            else:
                raise NotImplementedError
            d_rep = 64 * tmp
            num_class = 10
        elif self.hp.dataset in ['cifar10', 'cifar100']:
            from utils.nets_cifar import EncNet, TaskNet, Proj_Head, DecNet
            if self.hp.num_views == 2:
                tmp = 5 * 13
            elif self.hp.num_views == 3:
                tmp = 2 * 13
            else:
                raise NotImplementedError
            d_rep = 64 * tmp
            num_class = 10 if self.hp.dataset == 'cifar10' else 100
        elif self.hp.dataset in ['caltech101']:
            from utils.nets_cifar import EncNet, TaskNet, Proj_Head, DecNet
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

        # dimension of the contrastive embedding (after the projection head)
        d_con = 64

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
        task_model = TaskNet(d_rep, num_class)
        optim_tm, scheduler_tm = get_optim_scheduler(self.hp.optim, task_model, self.hp.lr,
                                                     self.hp.weight_decay, self.hp.num_epochs)
        if self.hp.alg == 'apfedc':
            active_proj = Proj_Head(d_rep, d_con)
            optim_ap, scheduler_ap = get_optim_scheduler(self.hp.optim, active_proj, self.hp.lr,
                                                         self.hp.weight_decay, self.hp.num_epochs)
            self.active_client.set_model([active_enc, optim_ae, scheduler_ae], [task_model, optim_tm, scheduler_tm],
                                         [active_proj, optim_ap, scheduler_ap], trade_off=self.hp.trade_off)
            self.active_client.set_loss_fun(torch.nn.NLLLoss(reduction='mean'))

        elif self.hp.alg == 'apfedr':
            if self.hp.active_dec:
                active_dec = DecNet()
                optim_ad, scheduler_ad = get_optim_scheduler(self.hp.optim, active_dec, self.hp.lr,
                                                             self.hp.weight_decay, self.hp.num_epochs)
                self.active_client.set_model([active_enc, optim_ae, scheduler_ae], [task_model, optim_tm, scheduler_tm],
                                             [active_dec, optim_ad, scheduler_ad], trade_off=self.hp.trade_off)
                self.active_client.set_loss_fun(torch.nn.NLLLoss(reduction='mean'), torch.nn.MSELoss(reduction='mean'))
            else:
                self.active_client.set_model([active_enc, optim_ae, scheduler_ae], [task_model, optim_tm, scheduler_tm],
                                             trade_off=self.hp.trade_off)
                self.active_client.set_loss_fun(torch.nn.NLLLoss(reduction='mean'))
        else:
            raise NotImplementedError

        # -----------------------------------
        # set the passive clients
        if self.hp.active_alone or self.hp.trade_off == 0:
            self.hp.active_alone = True
            return

        if self.hp.alg == 'apfedc':
            for i in range(self.num_passive_clients):
                passive_enc = EncNet()
                optim_pe, scheduler_pe = get_optim_scheduler(self.hp.optim, passive_enc, self.hp.lr,
                                                             self.hp.weight_decay, self.hp.num_epochs)
                passive_proj = Proj_Head(d_rep, d_con)
                optim_pp, scheduler_pp = get_optim_scheduler(self.hp.optim, passive_proj, self.hp.lr,
                                                             self.hp.weight_decay, self.hp.num_epochs)

                self.passive_clients[i].set_model([passive_enc, optim_pe, scheduler_pe],
                                                  [passive_proj, optim_pp, scheduler_pp])
                self.passive_clients[i].set_loss_fun(loss_fun=contrastive_loss)

        elif self.hp.alg == 'apfedr':
            for i in range(self.num_passive_clients):
                passive_dec = DecNet()
                optim_pd, scheduler_pd = get_optim_scheduler(self.hp.optim, passive_dec, self.hp.lr,
                                                             self.hp.weight_decay, self.hp.num_epochs)
                self.passive_clients[i].set_model([passive_dec, optim_pd, scheduler_pd])
                self.passive_clients[i].set_loss_fun(torch.nn.MSELoss(reduction='mean'))

        else:
            raise NotImplementedError



    def train_and_eval(self, train_loader, test_loader):

        print('# {}'.format(self.name))

        writer = SummaryWriter(self.log_path)

        for epoch in range(1, self.hp.num_epochs+1):

            loss_active_task, loss_passive_sum = 0, 0
            loss_active_dec = 0
            preds, ys = [], []
            for i_batch, batch_data in enumerate(train_loader):
                data, y = batch_data

                # set the data of active and passive clients
                data_active = data[self.hp.ith_view]
                self.active_client.set_data(data=data_active, y=y)
                del data[self.hp.ith_view]
                for i in range(self.num_passive_clients):
                    self.passive_clients[i].set_data(data[i])

                # 1. active client outputs representations
                rep = self.active_client.output_rep()

                # 2. all passive clients compute gradients over reps
                grads_passive, loss_passive_tmp = [None] * (self.hp.num_views - 1), 0
                if not self.hp.active_alone:
                    for i in range(self.num_passive_clients):
                        grads_passive[i], loss_passive = self.passive_clients[i].compute_grad(rep.clone())
                        loss_passive_tmp += loss_passive

                # 3. send back the grads and update the encoder of active client
                loss_active_tmp = self.active_client.train(grads_passive)

                # test
                pred = self.active_client.predict(data_active)

                preds.append(pred)
                ys.append(y)
                # if not self.hp.active_dec:
                if self.hp.alg == 'apfedc':
                    loss_active_task += loss_active_tmp
                elif self.hp.alg == 'apfedr':
                    loss_active_task += loss_active_tmp[0]
                    loss_active_dec += loss_active_tmp[1]
                else:
                    raise NotImplementedError
                loss_passive_sum += loss_passive_tmp

            preds = torch.cat(preds)
            ys = torch.cat(ys)

            results_train = classification_result(ys, preds)
            loss = loss_active_task + loss_passive_sum + loss_active_dec

            # measure on test set
            results_test = self.eval(test_loader)

            print('- {:0>3d}, loss: {:.4f}, acc_train: {:.2f}%, acc_test: {:.2f}%'.format(epoch, loss, results_train[0],
                                                                                          results_test[0]))

            writer.add_scalar('loss', loss, epoch)
            writer.add_scalar('loss_active_task', loss_active_task, epoch)
            writer.add_scalar('loss_active_dec', loss_active_dec, epoch)
            writer.add_scalar('loss_passive_sum', loss_passive_sum, epoch)

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

        writer.close()

        self.save_model()


    def eval(self, test_loader):
        # evaluate
        preds, ys = [], []
        for i_batch, batch_data in enumerate(test_loader):
            data, y = batch_data
            pred = self.active_client.predict(data[self.hp.ith_view])
            preds.append(pred)
            ys.append(y)
        preds = torch.cat(preds)
        ys = torch.cat(ys)
        # calculate accuracy
        results = classification_result(ys, preds)
        return results


    def save_model(self):
        save_dir = './models/iter_{}/{}/'.format(self.hp.iter, self.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # active client
        torch.save(self.active_client.encoder.state_dict(), save_dir+'active_enc.pt')
        torch.save(self.active_client.task_model.state_dict(), save_dir + 'task_model.pt')
        if self.hp.alg == 'apfedc':
            torch.save(self.active_client.proj_head.state_dict(), save_dir + 'active_proj_head.pt')
        if self.hp.active_dec:
            torch.save(self.active_client.decoder.state_dict(), save_dir + 'active_dec.pt')

        # passive clients
        if not self.hp.active_alone:
            if self.hp.alg == 'apfedc':
                for i in range(self.num_passive_clients):
                    torch.save(self.passive_clients[i].encoder.state_dict(), save_dir + 'passive_enc_{}.pt'.format(i))
                    torch.save(self.passive_clients[i].proj_head.state_dict(), save_dir + 'passive_proj_head_{}.pt'.format(i))
            elif self.hp.alg == 'apfedr':
                for i in range(self.num_passive_clients):
                    torch.save(self.passive_clients[i].decoder.state_dict(), save_dir + 'passive_dec_{}.pt'.format(i))
            else:
                raise  NotImplementedError

        return
