# Implements a dqn agent using pytorch
import torch
class DQNAgent:
    def __init__(self, input_shape, n_actions):
        self.input_shape = input_shape
        self.n_actions = n_actions

        # Hyperparameters
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_decay = 0.999
        self.eps_min = 0.001
        self.learning_rate = 0.0001
        self.target_update_interval = 20
        self.target_update_remaing = self.target_update_interval

        # Model architecture
        architecture = [
            torch.nn.Linear(input_shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_actions)
        ]

        # Initialize the model and target model, optimizer and loss function
        self.model = torch.nn.Sequential(*architecture)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.target_model = torch.nn.Sequential(*architecture)
        self.target_model.load_state_dict(self.model.state_dict())
        self.loss_fn = torch.nn.MSELoss()

    def select_action(self, state, greedy=False):
        if greedy: # Select action greedily (used when evaluating)
            with torch.no_grad():
                return self.model(torch.tensor(state).float()).argmax()
        else:      # Select action epsilon greedy
            if torch.rand(1) > self.eps:
                with torch.no_grad():
                    return self.model(torch.tensor(state).float()).argmax()
            else:
                return torch.randint(0, self.n_actions, (1, 1))

    def update_batch(self, batch_samples):
        batch_states, selected_actions, batch_rewards, batch_next_states, batch_done = batch_samples
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