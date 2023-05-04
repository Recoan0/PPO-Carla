import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


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
        
        self.critic_linear = nn.Linear(1024, 1)
        self.actor_linear = nn.Linear(1024, action_dim)
        
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
        assert inputs.ndim == 4 and inputs.shape[1:] == (3, 64, 64)
        assert 0 <= inputs.min() <= 1 and 0 <= inputs.max() <= 1

        x = inputs.mul(2).sub(1)
        x = F.relu(self.maxp1(self.conv1(x)))
        x = F.relu(self.maxp2(self.conv2(x)))
        x = F.relu(self.maxp3(self.conv3(x)))
        x = F.relu(self.maxp4(self.conv4(x)))
        x = torch.flatten(x, start_dim=1)
        
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
        
        self.buffer = RolloutBuffer()

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
        with torch.no_grad():
            state = torch.FloatTensor(state, device=self.device)
            action, action_logprob, state_val = self.policy_old.act(state)
        
        self.buffer.states.append(state)
        self.buffer.actions.append(action)
        self.buffer.logprobs.append(action_logprob)
        self.buffer.state_values.append(state_val)

        return action.item()

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(self.device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(self.device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(self.device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(self.device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

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
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
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
        assert self.ckpt_dir.is_dir()
        
        agent_state_dict = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        self.policy_old.load_state_dict(agent_state_dict)
        self.policy.load_state_dict(agent_state_dict)
        
        self.optimizer.load_state_dict(torch.load(self.ckpt_dir / 'optimizer.pt', map_location=self.device))
        
        for i in range(len(self.optimizer.param_groups)):
            self.optimizer.param_groups[i]['capturable'] = True
        
        print(f'Successfully loaded model and optimizer from {self.ckpt_dir.absolute()}.')
        
        
        
       


