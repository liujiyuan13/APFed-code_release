import torch

class ActiveClient(object):
    def __init__(self, device='cpu'):
        self.name = 'active_client'

        # device
        self.device = device

        # model
        self.encoder, self.optim_enc, self.scheduler_enc = None, None, None
        self.task_model, self.optim_task, self.scheduler_task = None, None, None
        self.proj_head, self.optim_proj, self.scheduler_proj = None, None, None
        self.decoder, self.optim_dec, self.scheduler_dec = None, None, None

        self.trade_off = 1

        # loss function
        self.loss_fun_task = None
        self.loss_fun_dec =None

        # data
        self.data = None
        self.y = None

        # temporary variables
        self.rep, self.rep_task, self.rep_con, self.emb_con = None, None, None, None

    def set_model(self, encoder_set, task_model_set, decoder_set=None, proj_head_set=None, trade_off=1):
        self.encoder, self.optim_enc, self.scheduler_enc = encoder_set
        self.encoder.to(self.device)

        self.task_model, self.optim_task, self.scheduler_task = task_model_set
        self.task_model.to(self.device)

        self.decoder, self.optim_dec, self.scheduler_dec = decoder_set
        self.decoder.to(self.device) if self.decoder is not None else 0

        self.proj_head, self.optim_proj, self.scheduler_proj = proj_head_set
        self.proj_head.to(self.device) if self.proj_head is not None else 0

        self.trade_off = trade_off

        self.set_train()

    def set_loss_fun(self, loss_fun_task, loss_fun_dec=None):
        self.loss_fun_task = loss_fun_task
        self.loss_fun_dec = loss_fun_dec

    def set_train(self):
        self.encoder.train()
        self.task_model.train()
        self.proj_head.train() if self.proj_head is not None else 0
        self.decoder.train() if self.decoder is not None else 0

    def set_data(self, data, y=None):
        self.data = data.to(self.device)
        self.y = y.to(self.device) if y is not None else None

    def output_rep(self):
        self.rep = self.encoder(self.data)
        self.rep_task = self.rep.detach().clone()

        if self.proj_head is not None:
            self.rep_con = self.rep.detach().clone()
            self.rep_con.requires_grad_()
            self.emb_con = self.proj_head(self.rep_con)

        emb_con_out = self.emb_con.detach().clone() if self.emb_con is not None else None

        return self.rep.detach().clone(), emb_con_out

    def train(self, grads_out: list, types: list):
        # compute the gradient from task model

        # task model net
        self.rep_task.requires_grad_()
        # forward
        pred = self.task_model(self.rep_task)
        loss_task = self.loss_fun_task(pred, self.y)

        loss_dec = 0
        if self.decoder is not None:
            dec = self.decoder(self.rep_task)
            loss_dec = self.loss_fun_dec(dec, self.data)

        loss = loss_task + loss_dec

        # backward: compute gradients
        self.optim_task.zero_grad()
        self.optim_dec.zero_grad() if self.decoder is not None else 0
        loss.backward()
        self.optim_task.step()
        self.optim_dec.step() if self.decoder is not None else 0

        rep_grad_task = self.rep_task.grad

        # backward the grads from contrastive learning
        grad_sum_con, grad_sum_dec = 0, 0
        for ig, grad in enumerate(grads_out):
            if grad is not None:
                if types[ig] == 'con':
                    grad_sum_con += grad.to(self.device) * self.trade_off
                elif types[ig] == 'dec':
                    grad_sum_dec += grad.to(self.device) * self.trade_off

        if isinstance(grad_sum_con, int):  # this means grad_sum is 0
            rep_grad_con = 0
        else:
            self.optim_proj.zero_grad()
            self.emb_con.backward(grad_sum_con)
            self.optim_proj.step()
            rep_grad_con = self.rep_con.grad

        # backward the grads from the task and constrastive learning: update the encoder
        grad_sum = rep_grad_con + grad_sum_dec + rep_grad_task
        self.optim_enc.zero_grad()
        self.rep.backward(grad_sum)
        self.optim_enc.step()

        # self.scheduler_enc.step()
        # self.scheduler_task.step()

        return loss_task.item(), loss_dec.item() if self.decoder is not None else 0

    def predict(self, data):
        # get data
        self.set_data(data)
        # eval mode
        self.encoder.eval()
        self.task_model.eval()
        # predict
        with torch.no_grad():
            pred = self.task_model(self.encoder(self.data))
            pred = pred.data.max(1, keepdim=True)[1]
        # reset the train mode
        self.set_train()

        return pred


class PassiveClient_Con(object):
    def __init__(self, device='cpu'):
        self.name = 'passive_client_con'

        # device
        self.device = device

        # encoder
        self.encoder, self.optim_enc, self.scheduler_enc = None, None, None
        self.proj_head, self.optim_proj, self.scheduler_proj = None, None, None

        # loss functions
        self.loss_fun = None

        # data
        self.data = None

    def set_train(self):
        self.encoder.train()
        self.proj_head.train()

    def set_model(self, encoder_set, proj_head_set):
        self.encoder, self.optim_enc, self.scheduler_enc = encoder_set
        self.encoder.to(self.device)

        self.proj_head, self.optim_proj, self.scheduler_proj = proj_head_set
        self.proj_head.to(self.device)

        self.set_train()

    def set_loss_fun(self, loss_fun):
        self.loss_fun = loss_fun

    def set_data(self, data):
        self.data = data.to(self.device)

    def compute_grad(self, rep):
        # receive rep
        rep = rep.to(self.device)
        rep.requires_grad_()

        # forward to get loss
        embedding = self.encoder(self.data)
        rep_self = self.proj_head(embedding)
        x = rep.view(rep.size(0), -1)
        loss = self.loss_fun([x, rep_self])

        self.optim_proj.zero_grad()
        self.optim_enc.zero_grad()
        loss.backward()
        self.optim_proj.step()
        self.optim_enc.step()

        # self.scheduler.step()

        return rep.grad, loss.item()


class PassiveClient_Dec(object):
    def __init__(self, device='cpu'):
        self.name = 'passive_client_dec'

        # device
        self.device = device

        # encoder
        self.decoder, self.optim, self.scheduler = None, None, None

        # loss functions
        self.loss_fun = None

        # data
        self.data = None


    def set_train(self):
        self.decoder.train()

    def set_model(self, decoder_set):
        self.decoder, self.optim, self.scheduler = decoder_set
        self.decoder.to(self.device)

        self.set_train()

    def set_loss_fun(self, loss_fun):
        self.loss_fun = loss_fun

    def set_data(self, data):
        self.data = data.to(self.device)

    def compute_grad(self, rep):
        # receive rep
        rep = rep.to(self.device)
        rep.requires_grad_()

        # forward to get loss
        data_out = self.decoder(rep)
        loss = self.loss_fun(data_out, self.data)

        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        # self.scheduler.step()

        return rep.grad, loss.item()