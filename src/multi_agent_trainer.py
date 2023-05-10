import time
from pathlib import Path
import shutil
import sys
import time
import numpy as np
from typing import Any, Dict, Optional, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter
from utils import set_seed
from einops import rearrange

import cProfile

from PPO import PPO

################################### Training ###################################
class MultiAgentTrainer():
    def __init__(self, cfg: DictConfig):
        wandb.init(
            config=OmegaConf.to_container(cfg, resolve=True),
            reinit=True,
            resume=True,
            **cfg.wandb
        )

        if cfg.common.seed is not None:
            set_seed(cfg.common.seed)

        self.cfg = cfg
        self.start_epoch = 1
        self.device = torch.device(cfg.common.device)

        self.ckpt_dir = Path('checkpoints')
        self.media_dir = Path('media')
        self.episode_dir = self.media_dir / 'episodes'
        self.reconstructions_dir = self.media_dir / 'reconstructions'

        if not cfg.common.resume:
            config_dir = Path('config')
            config_path = config_dir / 'trainer.yaml'
            config_dir.mkdir(exist_ok=False, parents=False)
            shutil.copy('.hydra/config.yaml', config_path)
            wandb.save(str(config_path))
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "src"), dst="./src")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.episode_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)

        if self.cfg.training.should:
            self.train_env = instantiate(cfg.env.train)

        if self.cfg.evaluation.should:
            # test_env = create_env(cfg.env.test, cfg.collection.test.num_envs)
            self.test_env = self.train_env  # TODO: Using same env for training and testing

        assert self.cfg.training.should or self.cfg.evaluation.should
        env = self.train_env if self.cfg.training.should else self.test_env

        # initialize a PPO agent
        self.agent = PPO(**cfg.agent, action_dim=env.action_space.n, device=self.device)
        
        print(f'{sum(p.numel() for p in self.agent.policy.parameters())} parameters in agent.policy')
        
        self.n_agents = self.cfg.env.train.n_agents
        
        self.train_total_epoch_rewards = []
        self.train_epoch_rewards = []
        self.test_total_epoch_rewards = []
        self.test_epoch_rewards = []

        if cfg.common.resume:
            self.load_checkpoint()
            
        self.writer = SummaryWriter(log_dir=".")
    
    def run(self):
        # training loop
        try:
            for epoch in tqdm(range(self.start_epoch, 1 + self.cfg.common.epochs), desc=f'Training Epoch'):
                start_time = time.time()
                to_log = []
                
                if self.cfg.training.should:
                    if epoch <= self.cfg.collection.train.stop_after_epochs:
                        logs_train, epoch_reward = self.collect_experience(self.train_env, epoch, mode='train')
                        self.train_epoch_rewards.append(epoch_reward)
                        self.train_total_epoch_rewards.append(epoch_reward.mean())
                        to_log += logs_train
                
                        # update PPO agent
                        self.agent.update()
                    
                if self.cfg.evaluation.should and (epoch % self.cfg.evaluation.every == 0):
                    logs_test, epoch_reward = self.collect_experience(self.test_env, epoch, mode='test')
                    self.test_epoch_rewards.append(epoch_reward)
                    self.test_total_epoch_rewards.append(epoch_reward.mean())
                    to_log += logs_test
                    
                    self.agent.buffer.clear()
                
                if self.cfg.training.should:
                    reward_dict = {'train': self.train_epoch_rewards, 'train_total': self.train_total_epoch_rewards,
                                   'test': self.test_epoch_rewards, 'test_total': self.test_total_epoch_rewards}
                    self.save_checkpoint(epoch, reward_dict, save_agent_only=not self.cfg.common.do_checkpoint)

                to_log.append({'duration': (time.time() - start_time) / 3600})
                for metrics in to_log:
                    wandb.log({'epoch': epoch, **metrics})
                    for k, v in metrics.items():
                        if k[-9:] == 'histogram':
                            self.writer.add_histogram(k, v, epoch)
                        else:
                            self.writer.add_scalar(k, v, epoch)
            
        finally:
            self.finish()
            
    def collect_experience(self, env, epoch, mode):
        to_log = []
        metrics_collect = {}
        
        obs = env.reset()
        current_ep_reward = np.zeros(self.n_agents, dtype=np.float32)

        for _ in tqdm(range(self.cfg.collection.train.config.num_steps), desc=f'Experience collection ({mode}) Epoch {epoch}'):

            # select action with policy
            rearranged_obs = rearrange(torch.tensor(obs, dtype=torch.float32, device=self.device).div(255), 'n h w c -> n c h w')
            action = self.agent.select_action(rearranged_obs)
            obs, rewards, dones, _ = env.step(action)

            # saving reward and is_terminals
            self.agent.buffer.rewards.append(rewards)
            self.agent.buffer.is_terminals.append(dones)
            current_ep_reward += rewards
        
        env.close()
        
        print(f'Episode average reward: {current_ep_reward.mean()}')
        metrics_collect['total_epoch_returns'] = current_ep_reward.mean()
        metrics_collect = {f'{mode}/{k}': v for k, v in metrics_collect.items()}
        to_log.append(metrics_collect)
        
        return to_log, current_ep_reward
        
    def finish(self):
        self.writer.close()
        wandb.finish()
        
    def save_checkpoint(self, epoch: int, reward_dict: dict, save_agent_only: bool):
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, reward_dict, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)
        
    def _save_checkpoint(self, epoch: int, reward_dict: dict, save_agent_only: bool):
        self.agent.save(self.ckpt_dir, save_agent_only)
        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir / 'epoch.pt')
            torch.save(reward_dict, self.ckpt_dir / 'rewards.pt')
            
    def load_checkpoint(self):
        assert self.ckpt_dir.is_dir()
        self.agent.load(self.ckpt_dir)
        
        reward_dict = torch.load(self.ckpt_dir / 'rewards.pt')
        self.train_epoch_rewards = reward_dict['train']
        self.train_total_epoch_rewards = reward_dict['train_total']
        self.test_epoch_rewards = reward_dict['test']
        self.test_total_epoch_rewards = reward_dict['test_total']
        
        self.start_epoch = torch.load(self.ckpt_dir / 'epoch.pt') + 1
        print(f'Successfully loaded model, optimizer, epoch and rewards from {self.ckpt_dir.absolute()}.')


# if __name__ == '__main__':
#     multi_agent_train()
    
    
    
    
    
    
    
