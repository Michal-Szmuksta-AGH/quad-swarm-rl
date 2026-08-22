"""
Wizualizacja 16 (4x4) epizodow POJEDYNCZEGO modelu na WYBRANEJ topologii —
trajektorie dronow (top-down) + przeszkody + start/goal markers.

Uzycie:
    python figures/topology_trajectories.py                                    # multiranger + building (default)
    python figures/topology_trajectories.py --model paper_baseline_8drones_s0 --topology grid
    python figures/topology_trajectories.py --topology cluster --rows 3 --cols 5

Output: figures/topology_trajectories_<model>_<topology>.png

Uzywamy TOPOLOGY_SEED=42 (fair-comparison: te same 16 topologii dla dowolnego
modelu) — mozna porownywac trajektorie roznych modeli na IDENTYCZNYCH ukladach.
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from swarm_rl.train import parse_swarm_cfg, register_swarm_components


SPAWN_AREA = 8.0
OBST_SIZE_DEFAULT = 0.6
DRONE_RADIUS = 0.046  # Crazyflie 2.1 body arm-length


TOPO_LABELS_PL = {
    'grid': 'Siatka regularna',
    'poisson': 'Rozklad Poissona',
    'cluster': 'Klastry gaussowskie',
    'building': 'Wnetrze budynku (BSP)',
    'mix': 'Mix (grid/poisson/cluster/building)',
}

MODEL_LABELS_SHORT = {
    'paper_baseline_8drones_s0': 'Paper baseline (SDF)',
    'perception_limited_r1.0_8drones_s0': 'Perception r=1.0',
    'perception_limited_r0.2_8drones_s0': 'Perception r=0.2',
    'multiranger_r4.0_8drones_s0': 'Multiranger',
}


def build_cfg(model, topology, train_dir, seed_start):
    """Prepare Sample Factory cfg mirroring eval_metrics.py setup."""
    argv = [
        '--algo=APPO', '--env=quadrotor_multi',
        f'--experiment={model}',
        f'--train_dir={train_dir}',
        '--device=cpu',                # fast for 16 episodes
        '--max_num_episodes=1',        # nie uzywamy — reset() sami
        '--eval_deterministic=False',
        '--no_render',
        '--quads_render=False',
        '--quads_use_numba=True',

        '--quads_episode_duration=15.0',
        '--quads_mode=mix',
        '--quads_room_dims', '10', '10', '10',

        '--quads_use_obstacles=True',
        '--quads_obst_spawn_area', str(SPAWN_AREA), str(SPAWN_AREA),
        '--quads_obst_density=0.2',
        '--quads_obst_size=0.6',
        f'--quads_obst_topology={topology}',
        f'--quads_topology_seed={seed_start}',

        '--quads_domain_random=False',
        '--quads_obst_density_random=False',
        '--quads_obst_size_random=False',
        '--replay_buffer_sample_prob=0.0',
        '--with_wandb=False',
    ]
    cfg = parse_swarm_cfg(argv=argv, evaluation=True)
    cfg = load_from_checkpoint(cfg)
    # Force overrides from CLI (load_from_checkpoint restores training values)
    cfg.no_render = True
    cfg.num_envs = 1
    cfg.quads_render = False
    cfg.quads_obst_topology = topology
    cfg.quads_topology_seed = seed_start
    cfg.quads_obst_spawn_area = [SPAWN_AREA, SPAWN_AREA]
    cfg.quads_obst_density = 0.2
    cfg.quads_obst_size = 0.6
    return cfg


def load_policy(cfg, env):
    """Load trained checkpoint as actor-critic model."""
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)
    name_prefix = dict(latest='checkpoint', best='best')[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(
        Learner.checkpoint_dir(cfg, cfg.policy_index), f'{name_prefix}_*')
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])
    return actor_critic, device


def run_episode(env, actor_critic, device, cfg, env_info):
    """Play one episode. Returns:
        traj:      (T, n_agents, 3)  drone positions per timestep
        obst_pos:  (n_obst, 3)       obstacle positions (world frame, XY visualized)
        goals:     (n_agents, 3)     per-agent goal positions
        crashes:   dict of bool arrays per-agent:
                   {'wall': [n], 'floor': [n], 'obst': [n]} — cumulative flags
    """
    obs, _ = env.reset()
    inner = env.unwrapped
    n_agents = inner.num_agents
    goals = np.array([inner.envs[i].goal for i in range(n_agents)])
    obst_pos = np.array(inner.obstacles.pos_arr) if inner.obstacles is not None else np.zeros((0, 3))

    rnn_states = torch.zeros([n_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    traj = [inner.pos.copy()]
    crashes = {
        'wall': np.zeros(n_agents, dtype=bool),
        'floor': np.zeros(n_agents, dtype=bool),
        'obst': np.zeros(n_agents, dtype=bool),
    }
    # Miejsca kazdego collision event (x, y) per drone
    collision_events = {'obst': [[] for _ in range(n_agents)],
                        'wall': [[] for _ in range(n_agents)],
                        'floor': [[] for _ in range(n_agents)]}
    # Kluczowa flaga: SF batched env auto-resetuje per-agent po done, wiec
    # inner.pos[i] po done drona i to jego NOWY spawn (nie miejsce lądowania).
    # Trzymamy cumulative done flag i "zamrazamy" trajektorie po pierwszym done,
    # zeby uniknac artefaktu "prostej linii od finish do nowej pozycji spawn".
    per_drone_done = np.zeros(n_agents, dtype=bool)
    # Cumulative reached_goal: env.reached_goal[i] uzywa SF native definicji
    # (mean dist_to_goal w ostatnich 5 tickach < approch_goal_metric ~0.5m).
    # Zbieramy per-tick zeby uniknac utraty flagi po done+reset.
    reached_flags = np.zeros(n_agents, dtype=bool)

    with torch.no_grad():
        while True:
            normalized_obs = prepare_and_normalize_obs(actor_critic, obs)
            policy_outputs = actor_critic(normalized_obs, rnn_states)
            actions = policy_outputs['actions']
            if cfg.eval_deterministic:
                actions = argmax_actions(actor_critic.action_distribution())
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)
            rnn_states = policy_outputs['new_rnn_states']

            # Zapisz pozycje PRZED step — dla dronow ktore w tym ticku zostana done,
            # inner.pos po step moze byc juz resetted (SF batched env auto-reset).
            pre_step_pos = inner.pos.copy()
            was_done = per_drone_done.copy()

            obs, rew, terminated, truncated, infos = env.step(actions)
            dones = make_dones(terminated, truncated).cpu().numpy()

            # Crash detection — tylko dla dronow ktore BYLY aktywne przed step.
            # Pozycje kolizji zapisujemy z pre-step (real location, nie post-reset).
            for i, e in enumerate(inner.envs):
                if was_done[i]:
                    continue
                if e.dynamics.crashed_wall and not crashes['wall'][i]:
                    crashes['wall'][i] = True
                    collision_events['wall'][i].append(tuple(pre_step_pos[i, :2]))
                if e.dynamics.crashed_floor and not crashes['floor'][i]:
                    crashes['floor'][i] = True
                    collision_events['floor'][i].append(tuple(pre_step_pos[i, :2]))
            for i in inner.curr_quad_col:
                idx = int(i)
                if was_done[idx]:
                    continue
                crashes['obst'][idx] = True
                collision_events['obst'][idx].append(tuple(pre_step_pos[idx, :2]))

            # Zbierz reached_goal flags (cumulative) — tylko dla wciaz aktywnych
            for i in range(n_agents):
                if not was_done[i] and inner.reached_goal[i]:
                    reached_flags[i] = True

            # Zbuduj klatke trajektorii:
            #  - already done drony (was_done[i]=True): pozycja zamrozona (poprzednia klatka)
            #  - drony ktore wlasnie stały sie done: uzyj PRE-STEP pos (last valid, nie reset)
            #  - aktywne drony: uzyj inner.pos (fresh)
            curr_pos = inner.pos.copy()
            prev_frame = traj[-1]
            for i in range(n_agents):
                if was_done[i]:
                    curr_pos[i] = prev_frame[i]
                elif dones[i]:
                    curr_pos[i] = pre_step_pos[i]
                    per_drone_done[i] = True
            traj.append(curr_pos)

            if bool(np.all(per_drone_done)):
                break

    return np.array(traj), obst_pos, goals, crashes, collision_events, reached_flags


def draw_episode(ax, traj, obst_pos, goals, crashes, collision_events,
                  reached_flags, obst_size, ep_idx):
    """Draw one episode: obstacles, drone trajectories (speed-gradient),
    start markers, crash-aware end markers."""
    hx = SPAWN_AREA / 2
    hy = SPAWN_AREA / 2

    # Spawn boundary
    ax.add_patch(patches.Rectangle(
        (-hx, -hy), SPAWN_AREA, SPAWN_AREA,
        linewidth=1.0, edgecolor='dimgray', linestyle='--',
        facecolor='#fafafa', zorder=0,
    ))

    # Faint grid
    for x in np.arange(-hx, hx + 0.1, 1.0):
        ax.axvline(x, color='lightgray', linewidth=0.3, alpha=0.4, zorder=1)
    for y in np.arange(-hy, hy + 0.1, 1.0):
        ax.axhline(y, color='lightgray', linewidth=0.3, alpha=0.4, zorder=1)

    # Obstacles
    for x, y, _ in obst_pos:
        ax.add_patch(patches.Circle(
            (x, y), obst_size / 2,
            facecolor='#2E2E2E', edgecolor='#111', linewidth=0.4,
            alpha=0.85, zorder=2,
        ))

    n_agents = traj.shape[1]
    tab_cmap = plt.get_cmap('tab10')

    for i in range(n_agents):
        drone_color = tab_cmap(i % 10)
        xs = traj[:, i, 0]
        ys = traj[:, i, 1]

        # Trajectory — solid line w kolorze drona (per-drone identity)
        ax.plot(xs, ys, color=drone_color, linewidth=1.2, alpha=0.85, zorder=3)

        # Start marker (drone-color triangle for identity)
        ax.plot(xs[0], ys[0], marker='^', color=drone_color, markersize=7,
                markeredgecolor='black', markeredgewidth=0.5, zorder=5)

        # GOAL marker — kolorowe kolo w pozycji celu (statyczne, nie miejsce lądowania).
        # Dla scenariuszy same_goal wiele dronow ma ten sam goal — pokazuje sie w
        # tej samej pozycji (przekrywajace sie kola).
        gx, gy = goals[i, 0], goals[i, 1]
        ax.add_patch(patches.Circle(
            (gx, gy), DRONE_RADIUS * 2.5,
            facecolor=drone_color, edgecolor='black', linewidth=0.6,
            alpha=0.75, zorder=5,
        ))

        # Collision events (miejsca gdzie doszlo do kolizji)
        for cx, cy in collision_events['obst'][i]:
            ax.plot(cx, cy, marker='*', color='#D62728', markersize=7,
                    markeredgecolor='black', markeredgewidth=0.3,
                    alpha=0.9, zorder=4)
        for cx, cy in collision_events['wall'][i]:
            ax.plot(cx, cy, marker='X', color='#D62728', markersize=6,
                    markeredgecolor='black', markeredgewidth=0.3,
                    alpha=0.9, zorder=4)

        # Miejsce zakonczenia epizodu (koncowa pozycja trajektorii) — zaznacz TYLKO
        # gdy drone crashed (nie dolecial). Dla success brak markera — trajektoria
        # po prostu prowadzi do goala. Dla crashed_obst również brak dodatkowego
        # markera koncowego (obst crashes juz jako * na trajektorii).
        end_x, end_y = xs[-1], ys[-1]
        if crashes['floor'][i]:
            ax.plot(end_x, end_y, marker='v', color='#D62728',
                    markersize=9, markeredgecolor='black', markeredgewidth=0.6,
                    zorder=6)
        elif crashes['wall'][i]:
            ax.plot(end_x, end_y, marker='X', color='#D62728',
                    markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                    zorder=6)

    # Episode label — reached = SF native definition (mean dist to goal < ~0.5m
    # w ostatnich 5 tickach). "stuck" = drone nie doleciał ani nie crashed
    # (utknął przy przeszkodzie).
    n_floor = int(crashes['floor'].sum())
    n_wall = int(crashes['wall'].sum())
    n_obst_events = sum(len(collision_events['obst'][i]) for i in range(n_agents))
    n_reached = int(reached_flags.sum())
    n_crashed = int((crashes['floor'] | crashes['wall']).sum())
    n_stuck = n_agents - n_reached - n_crashed
    label = f'ep={ep_idx}  reached:{n_reached}/{n_agents}'
    if n_stuck: label += f' stuck:{n_stuck}'
    if n_obst_events: label += f' *:{n_obst_events}'
    if n_floor: label += f' fl:{n_floor}'
    if n_wall: label += f' w:{n_wall}'
    ax.text(hx - 0.3, -hy + 0.3, label,
            fontsize=7, ha='right', va='bottom', color='dimgray',
            bbox=dict(boxstyle='round,pad=0.2',
                     facecolor='white', edgecolor='lightgray', alpha=0.9))

    ax.set_xlim(-hx - 0.4, hx + 0.4)
    ax.set_ylim(-hy - 0.4, hy + 0.4)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='multiranger_r4.0_8drones_s0',
                        help='Nazwa katalogu w train_dir/ (default: multiranger)')
    parser.add_argument('--topology', default='building',
                        choices=['grid', 'poisson', 'cluster', 'building', 'mix'],
                        help='Topologia do wizualizacji (default: building)')
    parser.add_argument('--rows', type=int, default=4)
    parser.add_argument('--cols', type=int, default=4)
    parser.add_argument('--seed_start', type=int, default=42,
                        help='TOPOLOGY_SEED — kazdy epizod uzywa (seed_start + ep_idx)')
    parser.add_argument('--train_dir', default='train_dir')
    args = parser.parse_args()

    register_swarm_components()
    cfg = build_cfg(args.model, args.topology, args.train_dir, args.seed_start)

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode=None,
    )
    env_info = extract_env_info(env, cfg)
    actor_critic, device = load_policy(cfg, env)

    n = args.rows * args.cols
    print(f'Running {n} episodes: model={args.model}, topology={args.topology}, seed_start={args.seed_start}')

    fig, axes = plt.subplots(args.rows, args.cols,
                              figsize=(3.5 * args.cols, 3.5 * args.rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n):
        traj, obst_pos, goals, crashes, coll_events, reached = run_episode(
            env, actor_critic, device, cfg, env_info)
        draw_episode(axes[i], traj, obst_pos, goals, crashes, coll_events,
                      reached, cfg.quads_obst_size, ep_idx=i)
        n_agents = crashes['floor'].shape[0]
        n_reached = int(reached.sum())
        n_crashed = int((crashes['floor'] | crashes['wall']).sum())
        n_stuck = n_agents - n_reached - n_crashed
        n_obst_events = sum(len(evs) for evs in coll_events['obst'])
        print(f'  ep {i+1}/{n}: T={len(traj)} frames, n_obst={len(obst_pos)}, '
              f'reached={n_reached}/{n_agents}, stuck={n_stuck}, obst_col_events={n_obst_events}, '
              f'floor={int(crashes["floor"].sum())}, wall={int(crashes["wall"].sum())}')

    env.close()

    model_label = MODEL_LABELS_SHORT.get(args.model, args.model)
    topo_label = TOPO_LABELS_PL.get(args.topology, args.topology)
    fig.suptitle(
        f'Trajektorie dronow: {model_label} @ {topo_label}\n'
        f'{n} epizodow (seedy {args.seed_start}-{args.seed_start + n - 1}) '
        f'| kazdy dron = inny kolor | ▲ start | ● goal | '
        f'* obst_col | ▼ floor_crash | X wall_crash',
        fontsize=12, y=0.995, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'topology_trajectories_{args.model}_{args.topology}.png'
    )
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
