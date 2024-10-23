from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class EarlyStoppingCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq=10000, patience=5, verbose=1):
        super(EarlyStoppingCallback, self).__init__(verbose)
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.patience = patience
        self.best_mean_reward = -np.inf
        self.no_improvement_steps = 0

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            mean_reward = self._evaluate_policy()
            if mean_reward > self.best_mean_reward:
                self.best_mean_reward = mean_reward
                self.no_improvement_steps = 0
            else:
                self.no_improvement_steps += 1

            if self.no_improvement_steps >= self.patience:
                if self.verbose > 0:
                    print(f"Stopping training early after {self.no_improvement_steps} evaluations with no improvement.")
                return False
        return True

    def _evaluate_policy(self):
        episode_rewards = []
        for _ in range(5):  # Evaluate for 5 episodes
            obs = self.eval_env.reset()
            done = False
            total_reward = 0.0
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, _ = self.eval_env.step(action)
                total_reward += reward
            episode_rewards.append(total_reward)
        mean_reward = np.mean(episode_rewards)
        return mean_reward