import torch


class ActiveClient(object):
    def __init__(self, device='cpu'):
        self.name = 'active_client_tvfl'

        # device
        self.device = device

        # model
        self.encoder, self.optim_enc, self.scheduler_enc = None, None, None
        self.task_model, self.optim_task, self.scheduler_task = None, None, None

        # loss function
        self.loss_fun_task = None

        # data
        self.data = None
        self.y = None

        # temporary variables
        self.rep, self.rep_task = None, None

    def set_model(self, encoder_set, task_model_set):
        self.encoder, self.optim_enc, self.scheduler_enc = encoder_set
        self.encoder.to(self.device)

        self.task_model, self.optim_task, self.scheduler_task = task_model_set
        self.task_model.to(self.device)

        self.set_train()

    def set_loss_fun(self, loss_fun_task):
        self.loss_fun_task = loss_fun_task

    def set_train(self):
        self.encoder.train()
        self.task_model.train()

    def set_data(self, data, y=None):
        if isinstance(data, list):
            self.data = []
            for i in range(len(data)):
                if len(data) == 2 and i == 1:  # specific to visual and acoustic modality of mosi and mosei datasets
                    self.data.append(data[i])
                else:
                    self.data.append(data[i].to(self.device))
        else:
            self.data = data.to(self.device)

        if y is not None:
            self.y = y.to(self.device)

    def compute_rep(self):
        self.rep = self.encoder(self.data)
        self.rep_task = self.rep.detach().clone()
        return self.rep_task

    def train(self, reps_out: list):
        # compute the gradient from task model

        # task model net
        self.rep_task.requires_grad_()

        rep_task_flat = self.rep_task.view(self.rep_task.shape[0], -1)
        reps_list = [rep_task_flat]
        for i in range(len(reps_out)):
            reps_out[i] = reps_out[i].to(self.device)
            reps_out[i].requires_grad_()
        for rep in reps_out:
            reps_list.append(rep.view(rep.shape[0], -1))
        rep_tmp = torch.cat(reps_list, dim=1)

        # forward
        pred = self.task_model(rep_tmp)
        loss_task = self.loss_fun_task(pred, self.y)

        # backward: compute gradients
        self.optim_task.zero_grad()
        loss_task.backward()
        self.optim_task.step()

        # backward the grads from the task: update the encoder
        self.optim_enc.zero_grad()
        self.rep.backward(self.rep_task.grad)
        self.optim_enc.step()

        # self.scheduler_enc.step()
        # self.scheduler_task.step()

        # obtain the grads of outside
        grads_out = []
        for rep in reps_out:
            grads_out.append(rep.grad)

        pred = pred.detach().data.max(1, keepdim=True)[1]
        return grads_out, loss_task.item(), pred

    def predict(self, reps_out):
        # eval mode
        self.encoder.eval()
        self.task_model.eval()
        # predict
        with torch.no_grad():
            rep_task = self.encoder(self.data)
            rep_task_flat = rep_task.view(rep_task.shape[0], -1)
            reps_list = [rep_task_flat]
            for rep in reps_out:
                rep = rep.to(self.device)
                reps_list.append(rep.view(rep.shape[0], -1))
            rep_tmp = torch.cat(reps_list, dim=1)
            pred = self.task_model(rep_tmp)
            pred = pred.data.max(1, keepdim=True)[1]
        # reset the train mode
        self.set_train()

        return pred


class PassiveClient(object):
    def __init__(self, device='cpu'):
        self.name = 'passive_client_vfl'

        # device
        self.device = device

        # encoder
        self.encoder, self.optim_enc, self.scheduler_enc = None, None, None

        # data
        self.data = None

        self.rep = None

    def set_train(self):
        self.encoder.train()

    def set_model(self, encoder_set, train_enc=True):
        self.encoder, self.optim_enc, self.scheduler_enc = encoder_set
        self.encoder.to(self.device)

        self.train_enc = train_enc
        self.set_train()

    def set_data(self, data):
        if isinstance(data, list):
            self.data = []
            for i in range(len(data)):
                if len(data) == 2 and i == 1:  # specific to visual and acoustic modality of mosi and mosei datasets
                    self.data.append(data[i])
                else:
                    self.data.append(data[i].to(self.device))
        else:
            self.data = data.to(self.device)

    def compute_rep(self):
        self.rep = self.encoder(self.data)
        return self.rep.detach().clone()

    def train(self, grad):

        self.optim_enc.zero_grad()
        self.rep.backward(grad)
        self.optim_enc.step()

        # self.scheduler.step()

        return
