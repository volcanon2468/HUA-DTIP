import os
import copy
import numpy as np
import torch
import torch.nn.functional as F
from collections import deque
import hydra
from omegaconf import DictConfig, OmegaConf

from src.twin.bayesian_vae import BayesianVAE
from src.twin.latent_sde import LatentNeuralSDE
from src.rl.sac_networks import TwinGymEnv, SquashedGaussianActor, TwinCritic
from src.rl.reward import MultiObjectiveReward
from src.rl.safety import SafetyGuard
from src.utils.seed import set_seed
from src.utils.logger import init_run, log_metrics, log_model, finish_run


class ReplayBuffer:
    def __init__(self, capacity: int = 100000, state_dim: int = 20, action_dim: int = 5):
        self.capacity = capacity
        self.states  = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones   = np.zeros(capacity, dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr]      = state
        self.actions[self.ptr]     = action
        self.rewards[self.ptr]     = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr]       = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx], device=device),
            torch.tensor(self.actions[idx], device=device),
            torch.tensor(self.rewards[idx], device=device).unsqueeze(-1),
            torch.tensor(self.next_states[idx], device=device),
            torch.tensor(self.dones[idx], device=device).unsqueeze(-1),
        )


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float = 0.005):
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)


def train_sac(cfg: DictConfig, device: torch.device):
    vae = BayesianVAE().to(device)
    sde = LatentNeuralSDE().to(device)
    ckpt_dir = cfg.checkpoints.dir
    for name, model in [("twin_vae", vae), ("twin_sde", sde)]:
        p = os.path.join(ckpt_dir, f"{name}.pt")
        if os.path.exists(p):
            model.load_state_dict(torch.load(p, map_location=device))
    vae.eval(); sde.eval()

    env = TwinGymEnv(vae, sde, episode_len=cfg.training.rl.episode_length, device=str(device))

    state_dim  = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    actor  = SquashedGaussianActor(state_dim, action_dim, cfg.model.rl.actor_hidden).to(device)
    critic = TwinCritic(state_dim, action_dim, cfg.model.rl.critic_hidden).to(device)
    target_critic = copy.deepcopy(critic).to(device)
    for p in target_critic.parameters():
        p.requires_grad = False

    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    target_entropy = -float(action_dim)

    opt_actor  = torch.optim.Adam(actor.parameters(),  lr=cfg.training.rl.actor_lr)
    opt_critic = torch.optim.Adam(critic.parameters(), lr=cfg.training.rl.critic_lr)
    opt_alpha  = torch.optim.Adam([log_alpha],         lr=cfg.training.rl.alpha_lr)

    buffer = ReplayBuffer(capacity=cfg.training.rl.buffer_size, state_dim=state_dim, action_dim=action_dim)
    reward_fn = MultiObjectiveReward()
    safety    = SafetyGuard()

    ep_rewards = deque(maxlen=100)
    best_avg   = -float("inf")

    for episode in range(cfg.training.rl.n_episodes):
        state, _ = env.reset()
        safety.reset()
        ep_reward = 0.0
        prev_z_mu = None

        for step in range(env.episode_len):
            with torch.no_grad():
                s_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                action, _ = actor.sample(s_t)
                action_np = action.squeeze(0).cpu().numpy()

            z_mu  = state[:env.latent_dim]
            z_std = state[env.latent_dim:]
            action_np = safety.check_and_clip(action_np, z_mu, z_std)

            next_state, env_reward, terminated, truncated, info = env.step(action_np)
            shaped_reward = reward_fn.compute(
                info["z_mu"].squeeze(), info["z_std"].squeeze(), action_np, prev_z_mu
            )
            penalty = safety.compute_penalty(action_np, z_mu, z_std)
            total_reward = shaped_reward - penalty

            done = terminated or truncated
            buffer.add(state, action_np, total_reward, next_state, done)

            ep_reward += total_reward
            prev_z_mu = z_mu.copy()
            state = next_state

            if buffer.size >= cfg.training.rl.warmup_steps:
                _update(actor, critic, target_critic, log_alpha, target_entropy,
                        opt_actor, opt_critic, opt_alpha, buffer,
                        cfg.training.rl.batch_size, cfg.training.rl.gamma,
                        cfg.training.rl.tau, device)

        ep_rewards.append(ep_reward)
        avg_reward = np.mean(ep_rewards)
        log_metrics({
            "rl/ep_reward": ep_reward,
            "rl/avg_reward_100": avg_reward,
            "rl/alpha": log_alpha.exp().item(),
            "rl/safety_violations": safety.episode_violations,
        }, step=episode)

        if avg_reward > best_avg and episode > 100:
            best_avg = avg_reward
            log_model(actor, "rl_actor", cfg)
            log_model(critic, "rl_critic", cfg)

        if episode % 100 == 0:
            print(f"  Episode {episode:5d}  reward={ep_reward:.3f}  "
                  f"avg100={avg_reward:.3f}  α={log_alpha.exp().item():.3f}  "
                  f"violations={safety.episode_violations}")

    return actor, critic


def _update(actor, critic, target_critic, log_alpha, target_entropy,
            opt_actor, opt_critic, opt_alpha, buffer,
            batch_size, gamma, tau, device):
    states, actions, rewards, next_states, dones = buffer.sample(batch_size, device)
    alpha = log_alpha.exp().detach()

    with torch.no_grad():
        next_actions, next_log_probs = actor.sample(next_states)
        q1_next, q2_next = target_critic(next_states, next_actions)
        q_next = torch.min(q1_next, q2_next) - alpha * next_log_probs
        q_target = rewards + gamma * (1 - dones) * q_next

    q1, q2 = critic(states, actions)
    critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
    opt_critic.zero_grad(); critic_loss.backward(); opt_critic.step()

    new_actions, log_probs = actor.sample(states)
    q1_new, q2_new = critic(states, new_actions)
    q_new = torch.min(q1_new, q2_new)
    actor_loss = (alpha * log_probs - q_new).mean()
    opt_actor.zero_grad(); actor_loss.backward(); opt_actor.step()

    alpha_loss = -(log_alpha * (log_probs.detach() + target_entropy)).mean()
    opt_alpha.zero_grad(); alpha_loss.backward(); opt_alpha.step()

    soft_update(target_critic, critic, tau)


@hydra.main(config_path="../configs", config_name="training", version_base=None)
def main(cfg: DictConfig):
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    init_run(cfg, name="rl-sac-training")
    train_sac(cfg, device)
    print("SAC training complete.")
    finish_run()


if __name__ == "__main__":
    main()
