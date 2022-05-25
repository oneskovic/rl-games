# Implements a dqn agent using pytorch
import torch
class DQNAgent:
    def __init__(self, input_shape, n_actions):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_decay = 0.995
        self.eps_min = 0.1
        # architecture = [
        #     torch.nn.Conv2d(3, 32, kernel_size=8, stride=4),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(32, 64, kernel_size=4, stride=2),
        #     torch.nn.ReLU(),
        #     torch.nn.Conv2d(64, 64, kernel_size=3, stride=1),
        #     torch.nn.ReLU(),
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(64 * 7 * 7, 512),
        #     torch.nn.ReLU(),
        #     torch.nn.Linear(512, n_actions)]
        architecture = [
            torch.nn.Linear(input_shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_actions)
        ]
        self.model = torch.nn.Sequential(*architecture)
        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.target_model = torch.nn.Sequential(*architecture)
        self.target_optimizer = torch.optim.Adam(self.target_model.parameters())
        self.loss_fn = torch.nn.MSELoss()

    def select_action(self, state, greedy=False):
        if greedy:
            with torch.no_grad():
                return self.model(torch.tensor(state).float()).argmax()
        else:
            if self.eps > self.eps_min:
                self.eps *= self.eps_decay

            if torch.rand(1) > self.eps:
                with torch.no_grad():
                    return self.model(torch.tensor(state).float()).argmax()
            else:
                return torch.randint(0, self.n_actions, (1, 1))

    def update(self, state, action, reward, next_state, done):
        state_action = self.model(state)[action]
        next_state_values = torch.zeros(1, 1)
        if not done:
            next_state_values = self.target_model(next_state).max(1)[0].detach()
        expected_state_action_values = reward + self.gamma * next_state_values
        loss = self.loss_fn(state_action, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def update_batch(self, batch_samples):
        batch_states = torch.stack([sample[0] for sample in batch_samples])
        pred_q = torch.zeros(len(batch_samples))
        for i, sample in enumerate(batch_samples):
            state, action, reward, next_state, done = sample
            pred_q[i] = self.model(state)[action]

        true_q = torch.zeros(len(batch_samples))
        for i, sample in enumerate(batch_samples):
            state, action, reward, next_state, done = sample
            next_q = torch.zeros(1)
            if not done:
                next_q = self.target_model(next_state).max().detach()
            true_q[i] = reward + self.gamma * next_q
        loss = self.loss_fn(pred_q, true_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def swap_models(self):
        self.model, self.target_model = self.target_model, self.model
        self.optimizer, self.target_optimizer = self.target_optimizer, self.optimizer
    
    def get_q_value(self, state, action):
        return self.model(state)[action]