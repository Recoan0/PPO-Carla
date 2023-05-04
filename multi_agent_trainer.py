import os
import glob
import time
from datetime import datetime
from collections import defaultdict
from functools import partial
from pathlib import Path
import shutil
import sys
import time
from typing import Any, Dict, Optional, Tuple

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
from torch.utils.tensorboard import SummaryWriter

import cProfile

from PPO import PPO
from utils import set_seed, EpisodeDirManager

################################### Training ###################################
class MultiAgentTrainer():
    def __init__(self, cfg: DictConfig) -> None:
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
            shutil.copytree(src=(Path(hydra.utils.get_original_cwd()) / "scripts"), dst="./scripts")
            self.ckpt_dir.mkdir(exist_ok=False, parents=False)
            self.media_dir.mkdir(exist_ok=False, parents=False)
            self.episode_dir.mkdir(exist_ok=False, parents=False)
            self.reconstructions_dir.mkdir(exist_ok=False, parents=False)

        episode_manager_train = EpisodeDirManager(self.episode_dir / 'train', max_num_episodes=cfg.collection.train.num_episodes_to_save)
        episode_manager_test = EpisodeDirManager(self.episode_dir / 'test', max_num_episodes=cfg.collection.test.num_episodes_to_save)
        self.episode_manager_imagination = EpisodeDirManager(self.episode_dir / 'imagination', max_num_episodes=cfg.evaluation.actor_critic.num_episodes_to_save)

        def create_env(cfg_env, num_envs):
            env_fn = partial(instantiate, config=cfg_env)
            return MultiProcessEnv(env_fn, num_envs, should_wait_num_envs_ratio=1.0) if num_envs > 1 else SingleProcessEnv(env_fn)

        if self.cfg.training.should:
            train_env = create_env(cfg.env.train, cfg.collection.train.num_envs)
            self.train_dataset = instantiate(cfg.datasets.train)
            self.train_collector = instantiate(cfg.collector.train, env=train_env, dataset=self.train_dataset, episode_dir_manager=episode_manager_train)

        if self.cfg.evaluation.should:
            # test_env = create_env(cfg.env.test, cfg.collection.test.num_envs)
            test_env = train_env  # TODO: Using same env for training and testing
            
            self.test_dataset = instantiate(cfg.datasets.test)
            self.test_collector = instantiate(cfg.collector.train, env=test_env, dataset=self.test_dataset, episode_dir_manager=episode_manager_test)

        assert self.cfg.training.should or self.cfg.evaluation.should
        env = train_env if self.cfg.training.should else test_env

        # initialize a PPO agent
        self.agent = PPO(**cfg.agent, action_dim=env.num_actions, device=self.device)
        
        print(f'{sum(p.numel() for p in self.agent.policy.parameters())} parameters in agent.policy')

        if cfg.common.resume:
            self.agent.load(self.ckpt_dir)
        
        self.writer = SummaryWriter(log_dir=".")
    
    def run(self):
        # training loop
        try:
            epoch_rewards = []
            for epoch in tqdm(range(self.start_epoch, 1 + self.cfg.common.epochs), desc=f'Training Epoch'):
                start_time = time.time()
                to_log = []
                if self.cfg.training.should:
                    if epoch <= self.cfg.collection.train.stop_after_epochs:
                        add_to_log, epoch_reward = self.collect_experience(self.env)
                        to_log += add_to_log
                        epoch_rewards.append(epoch_reward)
                
                        # update PPO agent
                        self.agent.update()
                    
                    self.save_checkpoint(epoch, save_agent_only=not self.cfg.common.do_checkpoint)

                to_log.append({'duration': (time.time() - start_time) / 3600})

            self.env.close()
            
        finally:
            self.finish()
            
    def collect_experience(self, env, epoch):
        state = env.reset()
        current_ep_reward = 0

        for i in tqdm(range(self.cfg.collection.train.config.num_steps), desc=f'Experience collection (train) Epoch {epoch}'):

            # select action with policy
            action = self.agent.select_action(state)
            state, reward, done, _ = env.step(action)

            # saving reward and is_terminals
            self.agent.buffer.rewards.append(reward)
            self.agent.buffer.is_terminals.append(done)
            current_ep_reward += reward

            # log in logging file
            if time_step % log_freq == 0:

                # log average reward till last episode
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)

                log_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                log_f.flush()

                log_running_reward = 0
                log_running_episodes = 0
        
        self.env.close()
        
        return {}, current_ep_reward
        
    def finish(self) -> None:
        self.writer.close()
        if self.cfg.training.should:
            self.train_collector.close()
        if self.cfg.evaluation.should:
            self.test_collector.close()
        wandb.finish()
        
    def save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        tmp_checkpoint_dir = Path('checkpoints_tmp')
        shutil.copytree(src=self.ckpt_dir, dst=tmp_checkpoint_dir, ignore=shutil.ignore_patterns('dataset'))
        self._save_checkpoint(epoch, save_agent_only)
        shutil.rmtree(tmp_checkpoint_dir)
        
    def _save_checkpoint(self, epoch: int, save_agent_only: bool) -> None:
        self.agent.save(self.ckpt_dir, save_agent_only)
        if not save_agent_only:
            torch.save(epoch, self.ckpt_dir / 'epoch.pt')


# if __name__ == '__main__':
#     multi_agent_train()
    
    
    
    
    
    
    
