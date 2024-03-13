import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as pyd
import copy
import math

import models.proto_utils as utils
from copy import deepcopy


class Projector(nn.Module):
    def __init__(self, pred_dim, proj_dim):
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(pred_dim, proj_dim), nn.ReLU(),
                                   nn.Linear(proj_dim, pred_dim))

        self.apply(utils.weight_init)

    def forward(self, x):
        return self.trunk(x)


class Encoder(nn.Module):
    def __init__(self, obs_shape, proj_dim):
        super().__init__()

        assert len(obs_shape) == 3

        self.conv = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1),
                                  nn.ReLU())

        self.repr_dim = 32 * 35 * 35

        self.projector = nn.Linear(self.repr_dim, proj_dim)

        self.apply(utils.weight_init)

    def encode(self, obs):
        obs = obs / 255.
        h = self.conv(obs)
        h = h.view(h.shape[0], -1)
        return h

    def forward(self, obs):
        h = self.encode(obs)
        z = self.projector(h)
        return z

class Proto(nn.Module):
    def __init__(self, proj_dim, pred_dim, T, num_protos, num_iters, topk):
        super().__init__()

        self.predictor = nn.Sequential(nn.Linear(proj_dim,
                                                 pred_dim), nn.ReLU(),
                                       nn.Linear(pred_dim, proj_dim))
        self.predictor_target = deepcopy(self.predictor)

        # TODO : if we want to add a projector it should be in Phi net (only if use_proto is True)
        # self.projector = Projector(pred_dim, proj_dim)
        # self.projector.apply(utils.weight_init)


        # self.encoder = Encoder(obs_shape).to(device)
        # self.encoder_target = deepcopy(self.encoder)

        self.num_iters = 3
        self.temp = T
        self.topk = topk  # not needed for us
        self.num_protos = num_protos

        self.protos = nn.Linear(proj_dim, num_protos, bias=False)
        # # candidate queue
        # self.register_buffer('queue', torch.zeros(queue_size, proj_dim))
        # self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

        self.outputs = dict()
        self.apply(utils.weight_init)


    def normalize_protos(self):
        C = self.protos.weight.data.clone()
        C = F.normalize(C, dim=1, p=2)
        self.protos.weight.data.copy_(C)

    def forward(self, s, t):
        # normalize prototypes
        self.normalize_protos()
        # s = self.encoder(obs)
        s = self.predictor(s)
        # s = self.projector(s)
        s = F.normalize(s, dim=1, p=2)

        scores_s = self.protos(s)
        log_p_s = F.log_softmax(scores_s / self.temp, dim=1)


        with torch.no_grad():
            # t = self.encoder_target(next_obs)
            # t = self.predictor_target(next_obs) # this should be the target of predictor(s)
            t = F.normalize(t, dim=1, p=2)
            scores_t = self.protos(t)
            q_t = self.sinkhorn(scores_t)
            # get the protos for s and t
            _, s_proto_idx = torch.max(scores_s, dim=1)
            _, t_proto_idx = torch.max(scores_t, dim=1)
            z_s = self.protos.weight.data[s_proto_idx]
            z_t = self.protos.weight.data[t_proto_idx]

        loss = -(q_t * log_p_s).sum(dim=1).mean()
        return z_s, z_t, loss



    def sinkhorn(self, scores):
        def remove_infs(x):
            m = x[torch.isfinite(x)].max().item()
            x[torch.isinf(x)] = m
            return x

        Q = scores / self.temp
        Q -= Q.max()

        Q = torch.exp(Q).T
        Q = remove_infs(Q)
        Q /= Q.sum()

        r = torch.ones(Q.shape[0], device=Q.device) / Q.shape[0]
        c = torch.ones(Q.shape[1], device=Q.device) / Q.shape[1]
        for it in range(self.num_iters):
            u = Q.sum(dim=1)
            u = remove_infs(r / u)
            Q *= u.unsqueeze(dim=1)
            Q *= (c / Q.sum(dim=0)).unsqueeze(dim=0)
        Q = Q / Q.sum(dim=0, keepdim=True)
        return Q.T

    def get_protos(self, phi, phi_target, obs, next_obs):
        z = phi(obs)
        with torch.no_grad():
            next_z = phi_target(next_obs)
        loss = self.forward(z, next_z)
        return loss

# class ProtoAgent(object):
#     def __init__(self, obs_shape, action_shape, action_range, device,
#                  encoder_cfg, critic_cfg, actor_cfg, proto_cfg, discount,
#                  init_temperature, lr, actor_update_frequency,
#                  critic_target_tau, critic_target_update_frequency,
#                  encoder_target_tau, encoder_update_frequency, batch_size,
#                  task_agnostic, intr_coef, num_seed_steps):
#         self.action_range = action_range
#         self.device = device
#         self.discount = discount
#         self.actor_update_frequency = actor_update_frequency
#         self.critic_target_tau = critic_target_tau
#         self.critic_target_update_frequency = critic_target_update_frequency
#         self.encoder_target_tau = encoder_target_tau
#         self.encoder_update_frequency = encoder_update_frequency
#         self.batch_size = batch_size
#         self.task_agnostic = task_agnostic
#         self.intr_coef = intr_coef
#         self.num_seed_steps = num_seed_steps
#         self.lr = lr

#         self.encoder = hydra.utils.instantiate(encoder_cfg).to(self.device)
#         self.encoder_target = hydra.utils.instantiate(encoder_cfg).to(
#             self.device)
#         self.encoder_target.load_state_dict(self.encoder.state_dict())

#         actor_cfg.params.repr_dim = self.encoder.repr_dim
#         critic_cfg.params.repr_dim = self.encoder.repr_dim

#         self.actor = hydra.utils.instantiate(actor_cfg).to(self.device)

#         self.critic = hydra.utils.instantiate(critic_cfg).to(self.device)
#         self.critic_target = hydra.utils.instantiate(critic_cfg).to(
#             self.device)
#         self.critic_target.load_state_dict(self.critic.state_dict())

#         self.proto = hydra.utils.instantiate(proto_cfg).to(self.device)

#         self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
#         self.log_alpha.requires_grad = True

#         self.aug = nn.Sequential(nn.ReplicationPad2d(4),
#                                  kornia.augmentation.RandomCrop((84, 84)))
#         # set target entropy to -|A|
#         self.target_entropy = -action_shape[0]
#         # optimizers
#         self.init_optimizers(lr)

#         self.train()
#         self.critic_target.train()
#         self.encoder_target.train()

#     def init_optimizers(self, lr):
#         # self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
#         # self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
#         #                                          lr=lr)
#         self.proto_optimizer = torch.optim.Adam(utils.chain(
#             self.encoder.parameters(), self.proto.parameters()),
#                                                 lr=lr)
#         # self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

#     def assign_modules_from(self, other):
#         self.encoder = other.encoder
#         self.encoder_target = other.encoder_target
#         self.proto = other.proto
#         self.actor = other.actor
#         # init opts
#         self.init_optimizers(self.lr)

#     def train(self, training=True):
#         self.training = training
#         self.actor.train(training)
#         self.critic.train(training)
#         self.encoder.train(training)
#         self.proto.train(training)

    # @property
    # def alpha(self):
    #     return self.log_alpha.exp()

    # def act(self, obs, sample=False):
    #     obs = torch.FloatTensor(obs).to(self.device)
    #     obs = obs.unsqueeze(0)
    #     obs = self.encoder.encode(obs)
    #     dist = self.actor(obs)
    #     action = dist.sample() if sample else dist.mean
    #     action = action.clamp(*self.action_range)
    #     assert action.ndim == 2 and action.shape[0] == 1
    #     return utils.to_np(action[0])

    # def update_repr(self, obs, next_obs, step):
    #     z = self.encoder(obs)
    #     with torch.no_grad():
    #         next_z = self.encoder_target(next_obs)

    #     loss = self.proto(z, next_z)
    #     self.proto_optimizer.zero_grad()
    #     loss.backward()
    #     self.proto_optimizer.step()

    # def compute_reward(self, next_obs, step):
    #     with torch.no_grad():
    #         y = self.encoder(next_obs)
    #         reward = self.proto.compute_reward(y)
    #     return reward

    # def update_critic(self, obs, action, reward, next_obs, discount, step):
    #     with torch.no_grad():
    #         dist = self.actor(next_obs)
    #         next_action = dist.rsample()
    #         log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
    #         target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
    #         target_V = torch.min(target_Q1,
    #                              target_Q2) - self.alpha.detach() * log_prob
    #         target_Q = reward + (discount * target_V)

    #     # get current Q estimates
    #     Q1, Q2 = self.critic(obs, action)
    #     critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

    #     # optimize the critic
    #     self.critic_optimizer.zero_grad()
    #     critic_loss.backward()
    #     self.critic_optimizer.step()



    # def update_actor_and_alpha(self, obs, step):
    #     dist = self.actor(obs)
    #     action = dist.rsample()
    #     log_prob = dist.log_prob(action).sum(-1, keepdim=True)
    #     actor_Q1, actor_Q2 = self.critic(obs, action)

    #     actor_Q = torch.min(actor_Q1, actor_Q2)

    #     actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

    #     # optimize the actor
    #     self.actor_optimizer.zero_grad()
    #     actor_loss.backward()
    #     self.actor_optimizer.step()

    #     self.log_alpha_optimizer.zero_grad()
    #     alpha_loss = (self.alpha *
    #                   (-log_prob - self.target_entropy).detach()).mean()
    #     alpha_loss.backward()
    #     self.log_alpha_optimizer.step()


    # def update(self, replay_buffer, step):
    #     if len(replay_buffer) < self.num_seed_steps:
    #         return

    #     obs, action, extr_reward, next_obs, discount = replay_buffer.sample(
    #         self.batch_size, self.discount)

    #     obs = self.aug(obs)
    #     next_obs = self.aug(next_obs)

    #     # train representation only during the task-agnostic phase
    #     if self.task_agnostic:
    #         if step % self.encoder_update_frequency == 0:
    #             self.update_repr(obs, next_obs, step)

    #             utils.soft_update_params(self.encoder, self.encoder_target,
    #                                      self.encoder_target_tau)

    #     with torch.no_grad():
    #         intr_reward = self.compute_reward(next_obs, step)

    #     if self.task_agnostic:
    #         reward = intr_reward
    #     else:
    #         reward = extr_reward + self.intr_coef * intr_reward

    #     # decouple representation
    #     with torch.no_grad():
    #         obs = self.encoder.encode(obs)
    #         next_obs = self.encoder.encode(next_obs)

    #     self.update_critic(obs, action, reward, next_obs, discount, step)

    #     if step % self.actor_update_frequency == 0:
    #         self.update_actor_and_alpha(obs, step)

    #     if step % self.critic_target_update_frequency == 0:
    #         utils.soft_update_params(self.critic, self.critic_target,
    #                                  self.critic_target_tau)
