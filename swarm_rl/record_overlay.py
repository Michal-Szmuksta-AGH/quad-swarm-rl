"""
Nagrywanie filmu z nałożonym tekstowym HUD-em (licznik kolizji + epizod + czas).

Wzorowane na sample_factory.enjoy, ale:
  * renderuje przez env.render() w trybie rgb_array,
  * po każdym kroku nakłada tekst OpenCV (cv2.putText) na surową klatkę,
  * zapisuje bezpośrednio przez cv2.VideoWriter (mp4v) — bez ffmpega
    (post-processing slow-mo + downscale robi shell wrapper).

Liczniki: drone-drone collisions, drone-obstacle collisions, success rate,
liczba dronów które dotarły do celu.

Usage (przez record_video.sh, nie bezpośrednio):
    python -m swarm_rl.record_overlay \
        --algo=APPO --env=quadrotor_multi \
        --experiment=paper_baseline_8drones_s0 \
        --max_num_episodes=3 \
        --video_output=path/to/output.mp4 \
        ...
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
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


# --- HUD layout ------------------------------------------------------------
# Białe kolory dla zwykłych metryk, pomarańczowy gdy coś się stało.
COLOR_WHITE = (255, 255, 255)
COLOR_OK = (180, 255, 180)      # zielony — sukces
COLOR_WARN = (60, 180, 255)     # pomarańczowy — kolizja wystąpiła
COLOR_BG = (0, 0, 0)            # czarne tło półprzezroczyste pod tekstem

FONT = cv2.FONT_HERSHEY_SIMPLEX


def _put_text(frame, text, x, y, scale, color, thickness=2):
    """Tekst z czarną otoczką (lepsza czytelność na każdym tle)."""
    # Outline
    cv2.putText(frame, text, (x, y), FONT, scale, COLOR_BG, thickness + 3, cv2.LINE_AA)
    # Main text
    cv2.putText(frame, text, (x, y), FONT, scale, color, thickness, cv2.LINE_AA)


def add_hud(frame, ep_idx, ep_total, ep_tick, tick_dt,
            col_dd, col_obst, num_at_goal, num_agents):
    """Wszystkie wartości to liczniki narastające w epizodzie."""
    h, w = frame.shape[:2]

    # Skaluj rozmiar tekstu do rozdzielczości (1080p baseline → bigger przy 1440p)
    scale = h / 1080.0
    title_scale = 1.0 * scale
    body_scale = 0.7 * scale
    line_h = int(38 * scale)
    margin = int(25 * scale)

    # Górny lewy: epizod + czas
    y = margin + int(20 * scale)
    _put_text(frame, f"Epizod {ep_idx}/{ep_total}", margin, y, title_scale, COLOR_WHITE)
    y += line_h
    _put_text(frame, f"Czas: {ep_tick * tick_dt:5.1f} s", margin, y, body_scale, COLOR_WHITE)

    # Górny prawy: liczniki kolizji
    col_dd_color = COLOR_WARN if col_dd > 0 else COLOR_WHITE
    col_obst_color = COLOR_WARN if col_obst > 0 else COLOR_WHITE

    text_col_dd = f"Kolizje dron-dron: {col_dd}"
    text_col_obst = f"Kolizje dron-przeszkoda: {col_obst}"
    text_at_goal = f"Drony przy celu: {num_at_goal}/{num_agents}"

    # Width measurement to align right
    (tw_dd, _), _ = cv2.getTextSize(text_col_dd, FONT, body_scale, 2)
    (tw_obst, _), _ = cv2.getTextSize(text_col_obst, FONT, body_scale, 2)
    (tw_goal, _), _ = cv2.getTextSize(text_at_goal, FONT, body_scale, 2)
    max_w = max(tw_dd, tw_obst, tw_goal)
    x_right = w - margin - max_w

    y = margin + int(20 * scale)
    _put_text(frame, text_col_dd, x_right, y, body_scale, col_dd_color)
    y += line_h
    _put_text(frame, text_col_obst, x_right, y, body_scale, col_obst_color)
    y += line_h
    _put_text(frame, text_at_goal, x_right, y, body_scale,
              COLOR_OK if num_at_goal == num_agents else COLOR_WHITE)

    return frame


def _live_counters(env):
    """Czyta liczniki z underlying QuadrotorEnvMulti."""
    # Unwrap do underlying env
    e = env
    while hasattr(e, "env"):
        e = e.env
    if hasattr(e, "unwrapped"):
        e = e.unwrapped

    col_dd = getattr(e, "collisions_per_episode", 0)
    col_obst = getattr(e, "obst_quad_collisions_per_episode", 0)
    tick = e.envs[0].tick if hasattr(e, "envs") and len(e.envs) > 0 else 0
    tick_dt = 1.0 / getattr(e, "control_freq", 100)

    # Liczba dronów które dotarły do celu (poniżej np. 0.5m od celu)
    num_at_goal = 0
    if hasattr(e, "envs"):
        for sub_env in e.envs:
            try:
                pos = sub_env.dynamics.pos
                goal = sub_env.goal[:3] if hasattr(sub_env, "goal") else None
                if goal is not None:
                    if np.linalg.norm(pos - goal) < 0.5:
                        num_at_goal += 1
            except Exception:
                pass

    return col_dd, col_obst, tick, tick_dt, num_at_goal


def record(cfg):
    cfg = load_from_checkpoint(cfg)
    cfg.no_render = False
    cfg.num_envs = 1

    env = make_env_func_batched(
        cfg,
        env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode="rgb_array",
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

    out_path = Path(cfg.video_output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("Nagrywam: %d epizodów do %s", cfg.max_num_episodes, out_path)

    video_writer = None
    fps = int(cfg.fps) if cfg.fps > 0 else 30

    obs, infos = env.reset()
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)

    episodes_done = 0
    frame_count = 0

    with torch.no_grad():
        while episodes_done < cfg.max_num_episodes:
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)

            actions = policy_outputs["actions"]
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)
            rnn_states = policy_outputs["new_rnn_states"]

            obs, rew, terminated, truncated, infos = env.step(actions)
            dones = make_dones(terminated, truncated)

            frame = env.render()
            if frame is not None:
                col_dd, col_obst, tick, tick_dt, num_at_goal = _live_counters(env)
                frame = add_hud(
                    frame.copy(), episodes_done + 1, cfg.max_num_episodes,
                    tick, tick_dt, int(col_dd), int(col_obst),
                    num_at_goal, env.num_agents,
                )

                if video_writer is None:
                    h, w = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    video_writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h))
                    log.info("Video: %dx%d @ %d FPS", w, h, fps)

                # OpenCV oczekuje BGR, env zwraca RGB
                video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_count += 1

            dones_np = dones.cpu().numpy()
            if all(dones_np):
                episodes_done += 1
                log.info("Epizod %d/%d done (frames: %d)",
                         episodes_done, cfg.max_num_episodes, frame_count)
                rnn_states = torch.zeros(
                    [env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device,
                )

    if video_writer:
        video_writer.release()
    env.close()

    log.info("Zapisano: %s (%d klatek)", out_path, frame_count)
    return ExperimentStatus.SUCCESS, 0.0


def main():
    register_swarm_components()
    # Inject --video_output argument manually before parsing
    import argparse
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--video_output", type=str, required=True)
    pre_args, remaining = pre_parser.parse_known_args()

    sys.argv = [sys.argv[0]] + remaining
    cfg = parse_swarm_cfg(evaluation=True)
    cfg.video_output = pre_args.video_output

    status, _ = record(cfg)
    return status


if __name__ == "__main__":
    sys.exit(main())
