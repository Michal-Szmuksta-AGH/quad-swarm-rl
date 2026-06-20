"""
Headless evaluation of a trained policy with aggregated success/collision metrics.

Mirrors sample_factory's enjoy() loop, but:
  * forces headless mode (no_render=True),
  * collects episode_extra_stats from infos at every done flag,
  * prints mean / std / n for every metric across all (agent, episode) samples,
  * writes a JSON summary next to the checkpoint.

Usage (see eval_hero_baseline.sh for an example):
    python -m swarm_rl.eval_metrics \
        --algo=APPO --env=quadrotor_multi \
        --experiment=forest_8drones_dr \
        --max_num_episodes=50 \
        ...
"""

import json
import sys
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import torch
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.utils import log

from swarm_rl.train import parse_swarm_cfg, register_swarm_components


# Paper Huang et al. ICRA 2024 — Table I, base setting (8 robots, 20% density, 0.6 m).
PAPER_TARGETS = {
    "metric/agent_success_rate": 0.97,
    "metric/agent_col_rate": 0.03,
}


def evaluate(cfg):
    # Load the saved training config from train_dir/<experiment>/cfg.json so that
    # the model architecture matches what was actually trained. CLI overrides
    # (e.g. quads_obst_density) are preserved.
    cfg = load_from_checkpoint(cfg)
    cfg.no_render = True
    cfg.num_envs = 1

    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )
    env_info = extract_env_info(env, cfg)

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])

    log.info("Loaded checkpoint, starting eval: %d episodes, %d drones",
             cfg.max_num_episodes, env.num_agents)

    extra_stats_all = defaultdict(list)
    episode_rewards = [deque([], maxlen=10000) for _ in range(env.num_agents)]
    env_episodes_done = 0
    num_frames = 0

    obs, infos = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    episode_reward = None

    with torch.no_grad():
        while env_episodes_done < cfg.max_num_episodes:
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            actions = policy_outputs["actions"]
            if cfg.eval_deterministic:
                action_distribution = actor_critic.action_distribution()
                actions = argmax_actions(action_distribution)
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)

            rnn_states = policy_outputs["new_rnn_states"]

            obs, rew, terminated, truncated, infos = env.step(actions)
            dones = make_dones(terminated, truncated)
            num_frames += 1

            if episode_reward is None:
                episode_reward = rew.float().clone()
            else:
                episode_reward += rew.float()

            dones_np = dones.cpu().numpy()
            for agent_i, done_flag in enumerate(dones_np):
                if done_flag:
                    episode_rewards[agent_i].append(episode_reward[agent_i].item())
                    info_i = infos[agent_i] if isinstance(infos, (list, tuple)) else {}
                    extra = info_i.get("episode_extra_stats", {})
                    for k, v in extra.items():
                        try:
                            extra_stats_all[k].append(float(v))
                        except (TypeError, ValueError):
                            pass
                    rnn_states[agent_i] = torch.zeros(
                        [get_rnn_size(cfg)], dtype=torch.float32, device=device,
                    )
                    episode_reward[agent_i] = 0

            if all(dones_np):
                env_episodes_done += 1
                if env_episodes_done % 5 == 0 or env_episodes_done == cfg.max_num_episodes:
                    log.info("Env episodes done: %d / %d  (frame %d)",
                             env_episodes_done, cfg.max_num_episodes, num_frames)

    env.close()

    print("\n" + "=" * 78)
    print(f"  EVALUATION SUMMARY  —  experiment: {cfg.experiment}")
    print("=" * 78)
    print(f"  Drones / env:      {env.num_agents}")
    print(f"  Env episodes done: {env_episodes_done}")
    print(f"  Total frames:      {num_frames}")
    print(f"  Room dims:         {cfg.quads_room_dims}")
    print(f"  Obst spawn area:   {cfg.quads_obst_spawn_area}")
    print(f"  Obst density/size: {cfg.quads_obst_density} / {cfg.quads_obst_size} m")
    print(f"  Quads mode:        {cfg.quads_mode}")
    print(f"  Domain random:     {cfg.quads_domain_random}")
    print(f"  Episode duration:  {cfg.quads_episode_duration} s")
    print("-" * 78)

    summary = {}
    for k in sorted(extra_stats_all.keys()):
        values = np.array(extra_stats_all[k])
        mean, std, n = float(values.mean()), float(values.std()), int(len(values))
        summary[k] = {"mean": mean, "std": std, "n": n}
        marker = ""
        if k in PAPER_TARGETS:
            target = PAPER_TARGETS[k]
            marker = f"  [paper: {target:.3f}]"
        print(f"  {k:55s} mean={mean:.4f}  std={std:.4f}  n={n}{marker}")
    print("=" * 78)

    avg_rew = np.mean([np.mean(r) for r in episode_rewards if len(r)])
    print(f"  Avg episode reward (across agents): {avg_rew:.3f}")
    print("=" * 78)

    succ = summary.get("metric/agent_success_rate", {}).get("mean")
    coll = summary.get("metric/agent_col_rate", {}).get("mean")
    if succ is not None and coll is not None:
        if succ >= 0.85 and coll <= 0.10:
            verdict = "OK — close to paper. Hero model is a reasonable baseline."
        elif succ >= 0.60:
            verdict = "MID — usable but worse than paper. Investigate before retraining."
        else:
            verdict = "BAD — far from paper. Check env config vs training config."
        print(f"  VERDICT: {verdict}")
        print("=" * 78)

    out_path = Path(cfg.train_dir) / cfg.experiment / "eval_metrics.json"
    payload = {
        "experiment": cfg.experiment,
        "num_env_episodes": env_episodes_done,
        "num_frames": num_frames,
        "num_drones": env.num_agents,
        "config_used": {
            "quads_room_dims": list(cfg.quads_room_dims),
            "quads_obst_spawn_area": list(cfg.quads_obst_spawn_area),
            "quads_obst_density": cfg.quads_obst_density,
            "quads_obst_size": cfg.quads_obst_size,
            "quads_obst_sensor_range": cfg.quads_obst_sensor_range,
            "quads_mode": cfg.quads_mode,
            "quads_episode_duration": cfg.quads_episode_duration,
            "quads_domain_random": cfg.quads_domain_random,
        },
        "metrics": summary,
        "avg_episode_reward": float(avg_rew),
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Saved JSON: {out_path}")
    print()

    return ExperimentStatus.SUCCESS, 0.0


def main():
    register_swarm_components()
    cfg = parse_swarm_cfg(evaluation=True)
    status, _ = evaluate(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
