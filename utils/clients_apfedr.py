import torch


class ActiveClient(object):
    def __init__(self, device='cpu'):
        self.name = 'active_client_apfedr'

        # device
        self.device = device

        # model
        self.encoder, self.optim_enc, self.scheduler_enc = None, None, None
        self.task_model, self.optim_task, self.scheduler_task = None, None, None
        self.decoder, self.optim_dec, self.scheduler_dec = None, None, None

        self.trade_off = 1

        # loss function
        self.loss_fun_task, self.loss_fun_dec = None, None

        # data
        self.data = None
        self.y = None

        # temporary variables
        self.rep, self.rep_task, self.rep_dec = None, None, None

        # if the encoders of the models are needed to be trained
        self.trainable = True

    def set_model(self, ancoder_set, task_model_set, decoder_set=None, trade_off=1):
        self.encoder, self.optim_enc, self.scheduler_enc = ancoder_set
        self.task_model, self.optim_task, self.scheduler_task = task_model_set
        self.encoder.to(self.device)
        self.task_model.to(self.device)

        if decoder_set is not None:
            self.decoder, self.optim_dec, self.scheduler_dec = decoder_set
            self.decoder.to(self.device)

        self.set_train(train=self.trainable)

        self.trade_off = trade_off

    def set_loss_fun(self, loss_fun_task, loss_fun_dec=None):
        self.loss_fun_task = loss_fun_task
        self.loss_fun_dec = loss_fun_dec

    def set_train(self, train=True):
        if train:
            self.encoder.train()
            self.task_model.train()
            self.decoder.train() if self.decoder is not None else None
        else:
            self.encoder.eval()
            self.task_model.eval()
            self.decoder.eval() if self.decoder is not None else None

    def set_data(self, data, y=None):
        self.data = data.to(self.device)
        if y is not None:
            self.y = y.to(self.device)

    def output_rep(self):
        self.rep = self.encoder(self.data)
        rep_detach = self.rep.detach()
        self.rep_task = rep_detach.clone()
        self.rep_dec = rep_detach.clone() if self.decoder is not None else None
        return rep_detach.clone()

    def train(self, grads_out: list):
        # compute the gradient from task model

        # task model net
        self.rep_task.requires_grad_()
        # forward
        pred = self.task_model(self.rep_task)
        loss_task = self.loss_fun_task(pred, self.y)
        # backward: compute gradients
        self.optim_task.zero_grad()
        loss_task.backward()
        self.optim_task.step()
        rep_grad_task = self.rep_task.grad

        # decoder net
        rep_grad_dec, loss_dec = 0, 0
        if self.decoder is not None:
            self.rep_dec.requires_grad_()
            # forward
            dec = self.decoder(self.rep_dec)
            loss_dec = self.loss_fun_dec(dec, self.data)
            # backward: compute gradients
            self.optim_dec.zero_grad()
            loss_dec.backward()
            self.optim_dec.step()
            rep_grad_dec = self.rep_dec.grad

        # simply sum the grads
        grad_sum = rep_grad_task + rep_grad_dec * self.trade_off
        for grad in grads_out:
            if grad is not None:
                grad_sum += grad.to(self.device) * self.trade_off

        # backward: update the encoder
        self.optim_enc.zero_grad()
        self.rep.backward(grad_sum)
        self.optim_enc.step()

        # self.scheduler_enc.step()
        # self.scheduler_task.step()

        return [loss_task.item(), loss_dec.item() if self.decoder is not None else loss_dec]

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
        self.set_train(train=self.trainable)

        return pred

    def reconstruct(self, data):
        # get data
        self.set_data(data)
        # eval mode
        self.encoder.eval()
        self.task_model.eval()
        # predict
        with torch.no_grad():
            dec = self.decoder(self.encoder(self.data))
        # reset the train mode
        self.set_train(train=self.trainable)

        return dec


class PassiveClient(object):
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

        # if the encoders of the models are needed to be trained
        self.trainable = True

    def set_train(self, train=True):
        if train:
            self.decoder.train()
        else:
            self.decoder.eval()

    def set_model(self, decoder_set):
        self.decoder, self.optim, self.scheduler = decoder_set
        self.decoder.to(self.device)

        self.set_train(train=self.trainable)

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