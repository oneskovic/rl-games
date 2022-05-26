# Implements a dqn agent using pytorch
import torch
class DQNAgent:
    def __init__(self, input_shape, n_actions):
        self.input_shape = input_shape
        self.n_actions = n_actions
        self.gamma = 0.99
        self.eps = 1.0
        self.eps_decay = 0.999
        self.eps_min = 0.001
        self.learning_rate = 0.0001
        architecture = [
            torch.nn.Linear(input_shape[0], 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, n_actions)
        ]
        self.model = torch.nn.Sequential(*architecture)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr = self.learning_rate)
        self.target_model = torch.nn.Sequential(*architecture)
        self.target_model.load_state_dict(self.model.state_dict())
        self.loss_fn = torch.nn.MSELoss()

    def select_action(self, state, greedy=False):
        if greedy:
            with torch.no_grad():
                return self.model(torch.tensor(state).float()).argmax()
        else:
            if torch.rand(1) > self.eps:
                with torch.no_grad():
                    return self.model(torch.tensor(state).float()).argmax()
            else:
                return torch.randint(0, self.n_actions, (1, 1))

    def update(self, state, action, reward, next_state, done):
        state_action = self.model(state)[action]
        next_state_values = torch.zeros(1, 1)
        if not done:
            next_state_values = self.model(next_state).max(1)[0]
        expected_state_action_values = reward + self.gamma * next_state_values
        loss = self.loss_fn(state_action, expected_state_action_values)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss.item()

    def update_batch(self, batch_samples):
        batch_states, selected_actions, batch_rewards, batch_next_states, batch_done = batch_samples
        selected_actions = selected_actions.reshape(-1, 1)

        pred_q = self.model(batch_states)
        pred_q = pred_q.gather(1, selected_actions).flatten()

        true_q = self.gamma * self.model(batch_next_states).max(1)[0] * (~batch_done) + batch_rewards

        loss = self.loss_fn(pred_q, true_q)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        if self.eps > self.eps_min:
            self.eps *= self.eps_decay
        return loss.item()

    def swap_models(self):
        pass
        #self.target_model.load_state_dict(self.model.state_dict())
    
    def get_q_value(self, state, action):
        return self.model(state)[action]