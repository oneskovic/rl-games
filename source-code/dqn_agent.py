# Implements a dqn agent using pytorch
import torch
from config import TORCH_DEVICE
class DQNAgent:
    def __init__(self, input_shape, n_actions, gamma = 0.99, eps = 1.0, eps_min = 0.01, eps_decay = 0.999, lr = 0.0001, target_update_interval = 20, architecture = None):
        self.input_shape = input_shape
        self.n_actions = n_actions

        # Hyperparameters
        self.gamma = gamma
        self.eps = eps
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.learning_rate = lr
        self.target_update_interval = target_update_interval
        self.target_update_remaing = self.target_update_interval

        if architecture is None:
            # Model architecture
            architecture = [
                torch.nn.Linear(input_shape[0], 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, n_actions)
            ]

        # Initialize the model and target model, optimizer and loss function
        self.model = torch.nn.Sequential(*architecture).to(TORCH_DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.target_model = torch.nn.Sequential(*architecture).to(TORCH_DEVICE)
        self.target_model.load_state_dict(self.model.state_dict())
        self.loss_fn = torch.nn.SmoothL1Loss()

    def select_action(self, state, greedy=False):
        state = torch.from_numpy(state).float().unsqueeze(0).to(TORCH_DEVICE)
        #state /= 255.0
        
        # Select action greedily (used when evaluating)
        if greedy: 
            with torch.no_grad():
                return self.model(state).argmax()
        # Select action epsilon greedy
        else:      
            if torch.rand(1) > self.eps:
                with torch.no_grad():
                    return self.model(state).argmax()
            else:
                return torch.randint(0, self.n_actions, (1, 1))

    def update_batch(self, batch_samples):
        batch_states, selected_actions, batch_rewards, batch_next_states, batch_done = batch_samples
        #batch_states /= 255.0
        #batch_next_states /= 255.0
        selected_actions = selected_actions.reshape(-1, 1)

        # Predict Q values using current model
        pred_q = self.model(batch_states)
        # Select only the Q values for the actions that were executed
        pred_q = pred_q.gather(1, selected_actions).flatten()

        # Predict Q values using the target model 
        true_q = self.gamma * self.target_model(batch_next_states).max(1)[0] * (~batch_done) + batch_rewards

        # Compute loss and backprop
        loss = self.loss_fn(pred_q, true_q)
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradient
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        # Decay epsilon
        if self.eps > self.eps_min:
            self.eps *= self.eps_decay

        # Update target model if needed
        self.target_update_remaing -= 1
        if self.target_update_remaing <= 0:
            self.update_target_model()
            self.target_update_remaing = self.target_update_interval

        return loss.item()

    def update_target_model(self):
        self.target_model.load_state_dict(self.model.state_dict())
    
    def get_q_value(self, state, action):
        return self.model(state)[action]