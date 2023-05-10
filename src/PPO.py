import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from tqdm import tqdm
import numpy as np


################################## PPO Policy ##################################
class RolloutBuffer(Dataset):
    def __init__(self, device):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
        self.prepared = False
        self.device = device
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]
        self.prepared = False
        
    def assert_valid(self):
        assert len(self.states) == len(self.actions) == len(self.rewards) == len(self.is_terminals) \
            == len(self.state_values) == len(self.logprobs), "Buffer not updated correctly, properties have different lengths"
            
    def generate_monte_carlo_returns(self, gamma):
        # Monte Carlo estimate of returns
        all_returns = []
        for i in range(len(self.states[0])):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(self.rewards), reversed(self.is_terminals)):
                if is_terminal[i]:
                    discounted_reward = 0
                discounted_reward = reward[i] + (gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            all_returns.append(rewards)
            
        mc_returns = np.vstack(all_returns, dtype=np.float32)
        
        # Normalizing the rewards
        self.mc_returns = (mc_returns - mc_returns.mean()) / (mc_returns.std() + 1e-7)
        
    def prepare_data(self, gamma):
        self.assert_valid()
        assert not self.prepared, "Data already prepared"
        
        self.generate_monte_carlo_returns(gamma)
        self.convert_data_shapes()
        
        # Flatten tensors
        self.mc_returns = self.mc_returns.reshape(-1, *self.mc_returns.shape[2:])
        self.old_states = self.old_states.reshape(-1, *self.old_states.shape[2:])
        self.old_actions = self.old_actions.reshape(-1, *self.old_actions.shape[2:])
        self.old_logprobs = self.old_logprobs.reshape(-1, *self.old_logprobs.shape[2:])
        self.old_state_values = self.old_state_values.reshape(-1, *self.old_state_values.shape[2:])
        
        # Calculate advantages
        self.advantages = self.mc_returns - self.old_state_values
        
        self.prepared = True
        
    def convert_data_shapes(self):
        # convert list to tensor
        self.old_states = np.squeeze(np.stack(self.states, axis=1))
        self.old_actions = np.squeeze(np.stack(self.actions, axis=1))
        self.old_logprobs = np.squeeze(np.stack(self.logprobs, axis=1))
        self.old_state_values = np.squeeze(np.stack(self.state_values, axis=1))
        
    def __getitem__(self, index):
        return self.old_states[index], self.old_actions[index], self.old_logprobs[index], \
            self.advantages[index], self.mc_returns[index]
            
    def __len__(self):
        assert len(self.states) * len(self.states[0]) == len(self.mc_returns), \
            f"{len(self.states)}, {len(self.mc_returns)}"
        return len(self.states)   


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        assert tuple(state_dim) == (3, 64, 64), "states need to be of size (3, 64, 64)"
        
        super(ActorCritic, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, stride=1, padding=1)
        self.maxp1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 3, stride=1, padding=1)
        self.maxp2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.maxp3 = nn.MaxPool2d(2, 2)
        self.conv4 = nn.Conv2d(64, 64, 3, stride=1, padding=1)
        self.maxp4 = nn.MaxPool2d(2, 2)
        
        self.common_linear = nn.Linear(1024, 512)
        
        self.critic_linear = nn.Linear(512, 1)
        self.actor_linear = nn.Linear(512, action_dim)
        
        # self.actor = nn.Sequential(
        #                 nn.Linear(state_dim, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, action_dim),
        #                 nn.Softmax(dim=-1)
        #             )
        # # critic
        # self.critic = nn.Sequential(
        #                 nn.Linear(state_dim, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 64),
        #                 nn.Tanh(),
        #                 nn.Linear(64, 1)
        #             )

    def forward(self, inputs: torch.FloatTensor):
        assert inputs.ndim == 4 and tuple(inputs.shape[1:]) == (3, 64, 64)
        mini, maxi = inputs.min().item(), inputs.max().item()
        assert 0 <= mini <= 1 and 0 <= maxi <= 1

        x = inputs.mul(2).sub(1)
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)
        
        x = F.relu(self.common_linear(x))
        
        logits_actions = self.actor_linear(x)
        means_values = self.critic_linear(x)
        
        return logits_actions, means_values
        
    
    def act(self, state):
        logits_actions, means_values = self(state)
        dist = Categorical(logits=logits_actions)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach(), means_values.detach()
    
    def evaluate(self, state, action):
        logits_actions, means_values = self(state)
        dist = Categorical(logits=logits_actions)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        
        return action_logprobs, means_values, dist_entropy


class PPO:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        self.buffer = RolloutBuffer(device)

        self.policy = ActorCritic(state_dim, action_dim).to(device)
        # self.optimizer = torch.optim.Adam([
        #                 {'params': self.policy.actor.parameters(), 'lr': lr_actor},
        #                 {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        #             ])
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr_actor)

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        assert torch.is_tensor(state)
        with torch.no_grad():
            # No need to convert state to Tensor, trainer already passes tensor
            action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.states.append(state.cpu().numpy())
        self.buffer.actions.append(action.cpu().numpy())
        self.buffer.logprobs.append(action_logprob.cpu().numpy())
        self.buffer.state_values.append(state_val.cpu().numpy())

        return action.cpu().numpy()

    def update(self):
        self.buffer.prepare_data(self.gamma)
        # loader = DataLoader(self.buffer, batch_size=64, shuffle=True, 
        #                     collate_fn=lambda x: tuple(x_.to(self.device) for x_ in default_collate(x)))
        loader = DataLoader(self.buffer, batch_size=len(self.buffer), shuffle=True, 
                            collate_fn=lambda x: tuple(x_.to(self.device) for x_ in default_collate(x)))

        # Optimize policy for K epochs
        for _ in tqdm(range(self.K_epochs), desc=f"Training PPO"):
            for old_states, old_actions, old_logprobs, advantages, mc_returns in loader:
                # Evaluating old actions and values
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

                # match state_values tensor dimensions with rewards tensor
                state_values = torch.squeeze(state_values)
                
                # Finding the ratio (pi_theta / pi_theta__old)
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, mc_returns) - 0.01 * dist_entropy
                
                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    def save(self, ckpt_dir, save_agent_only=False):
        torch.save(self.policy_old.state_dict(), ckpt_dir / 'last.pt')
        
        if not save_agent_only:
            torch.save(self.optimizer.state_dict(), ckpt_dir / 'optimizer.pt')
   
    def load(self, checkpoint_path):
        assert checkpoint_path.is_dir()
        
        agent_state_dict = torch.load(checkpoint_path / 'last.pt', map_location=lambda storage, loc: storage)
        self.policy_old.load_state_dict(agent_state_dict)
        self.policy.load_state_dict(agent_state_dict)
        
        self.optimizer.load_state_dict(torch.load(checkpoint_path / 'optimizer.pt', map_location=self.device))
        
        # for i in range(len(self.optimizer.param_groups)):
        #     self.optimizer.param_groups[i]['capturable'] = True
        
        print(f'Successfully loaded model and optimizer from {checkpoint_path.absolute()}.')
        
        
        
       


