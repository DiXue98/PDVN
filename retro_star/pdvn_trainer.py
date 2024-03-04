import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from mlp_retrosyn.mlp_policies import preprocess
class PDVNTrainer:
    def __init__(self, args, prior_net, value_net, one_step, device):
        self.args = args
        self.prior_net = prior_net
        self.value_net = value_net
        self.one_step = one_step
        self.lr = args.lr
        self.device = device
        self.minibatch_size = args.minibatch_size
        self.advantage = args.advantage

        self.prior_optim = optim.Adam(list(prior_net.parameters()), lr=self.lr)
        self.value_optim = optim.Adam(list(value_net.parameters()), lr=self.lr)
        self.actor_optim = optim.Adam(list(one_step.net.parameters()), lr=self.lr)
        self.optimizers = [self.prior_optim, self.value_optim, self.actor_optim]

    def fit_prior(self, mols, pris, epoch_num=1, once=True):
        fps = np.array([preprocess(mol, fp_dim=self.args.fp_dim) for mol in mols])
        pris = np.array(pris)
        loss = [] # Append all batches in all epochs. 
        if self.args.dropout:
            self.prior_net.train()
        else:
            self.prior_net.eval()

        if once: # move all collected mols and pris from cpu to gpu once. 
            fps = torch.FloatTensor(fps).to(self.device)
            pris = torch.FloatTensor(pris).to(self.device)

        for _ in range(epoch_num):
            fps, pris = self._shuffle_data(fps, pris)
            for i in range(0, len(mols), self.minibatch_size):
                fps_batch = fps[i:i+self.minibatch_size]
                pris_batch = pris[i:i+self.minibatch_size].unsqueeze(-1)
                if not once: # move current batch from cpu to gpu. 
                    fps_batch = torch.FloatTensor(fps_batch).to(self.device)
                    pris_batch = torch.FloatTensor(pris_batch).to(self.device)

                pris_batch_pred = self.prior_net(fps_batch)
                loss_minibatch = F.binary_cross_entropy(pris_batch_pred, pris_batch)
                loss.append(loss_minibatch.item())
                self._optimize(self.prior_net, loss_minibatch, self.prior_optim)

        self.prior_net.eval()
        return np.mean(loss)

    def fit_value(self, mols, vals, epoch_num=1, once=True):
        fps = np.array([preprocess(mol, fp_dim=self.args.fp_dim) for mol in mols])
        vals = np.array(vals)
        loss = [] # Append all batches in all epochs. 
        if self.args.dropout:
            self.value_net.train()
        else:
            self.value_net.eval()

        if once: # move all collected mols and pris from cpu to gpu once. 
            fps = torch.FloatTensor(fps).to(self.device)
            vals = torch.FloatTensor(vals).to(self.device)

        for _ in range(epoch_num):
            fps, vals = self._shuffle_data(fps, vals)
            for i in range(0, len(mols), self.minibatch_size):
                fps_batch = fps[i:i+self.minibatch_size]
                vals_batch = vals[i:i+self.minibatch_size].unsqueeze(-1)
                if not once: # move current batch from cpu to gpu. 
                    fps_batch = torch.FloatTensor(fps_batch).to(self.device)
                    vals_batch = torch.FloatTensor(vals_batch).to(self.device)

                vals_batch_pred = self.value_net(fps_batch)
                loss_minibatch = F.mse_loss(vals_batch_pred, vals_batch)
                loss.append(loss_minibatch.item())
                self._optimize(self.value_net, loss_minibatch, self.value_optim)

        self.value_net.eval()
        return np.mean(loss)

    def fit_model(self, mols, tmps, epoch_num=1, once=True):
        fps = np.array([preprocess(mol, fp_dim=self.args.fp_dim) for mol in mols])
        tmps = np.array(tmps)
        loss = []
        if self.args.dropout:
            self.one_step.net.train()
            for m in self.one_step.net.modules():
                if isinstance(m, torch.nn.BatchNorm1d):
                    m.eval()
        else:
            self.one_step.net.eval()
        
        if once: # move all collected mols and pris from cpu to gpu once. 
            fps = torch.FloatTensor(fps).to(self.device)
            tmps = torch.LongTensor(tmps).to(self.device)

        for _ in range(epoch_num):
            fps, tmps = self._shuffle_data(fps, tmps)
            for i in range(0, len(mols), self.minibatch_size):

                fps_batch = fps[i:i+self.minibatch_size]
                tmps_batch = tmps[i:i+self.minibatch_size]
                if not once: # move current batch from cpu to gpu. 
                    fps_batch = torch.FloatTensor(fps_batch).to(self.device)
                    tmps_batch = torch.LongTensor(tmps_batch).to(self.device)

                if self.args.realistic_filter:
                    action_logits, _, _ = self.one_step.forward_topk(fps_batch, topk=self.args.expansion_topk)
                else:
                    action_logits = self.one_step.net(fps_batch)
                dist = Categorical(logits=action_logits)
                # action_entropy_batch = dist.entropy().mean()
                loss_minibatch = F.cross_entropy(action_logits, tmps_batch)
                # loss_minibatch -= self.args.entropy_coef * action_entropy_batch
                loss.append(loss_minibatch.item())
                self._optimize(self.one_step.net, loss_minibatch, self.actor_optim)

        self.one_step.net.eval()
        return np.mean(loss)

    def _optimize(self, net, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), self.args.gradient_clip)
        optimizer.step()

    def _shuffle_data(self, features, labels):
        shuffled_idx = np.arange(len(features))
        np.random.shuffle(shuffled_idx)
        features = features[shuffled_idx]
        labels = labels[shuffled_idx]
        return features, labels