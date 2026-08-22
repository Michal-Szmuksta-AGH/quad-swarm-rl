"""
Wizualizacja long-range flight: 8 dronow startuje w formacji 2x4 w JEDNYM ROGU
duzego srodowiska (5x standard = 40x40m), leca do WSPOLNEGO celu w rogu
przeciwnym. Domyslnie 1 epizod (jeden duzy panel 10x10 cali).

Hack approach: standardowy env, po env.reset() force'ujemy positions dronow
+ goal (nadpisujemy scenariusz). Pierwsze klatki polityki moga byc "dziwne"
(stale obs z initial reset), ale polityka szybko sie stabilizuje.

Uzycie:
    python figures/long_flight_trajectories.py                           # multiranger, building, 1 epizod
    python figures/long_flight_trajectories.py --model paper_baseline_8drones_s0
    python figures/long_flight_trajectories.py --topology grid
    python figures/long_flight_trajectories.py --rows 2 --cols 2         # 4 epizody dla porownania

Output: figures/long_flight_<model>_<topology>.png
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


# Big-env parameters (5x standard: default 8x8m -> 40x40m)
BIG_SPAWN_AREA = 40.0
BIG_ROOM_XY = 50.0
BIG_ROOM_Z = 10.0
EPISODE_DURATION = 90.0  # s -> ~6000 tickow @ 100Hz; drone at 3m/s pokona 40m w 15s

FORMATION_SPACING = 1.5  # m miedzy dronami w formacji
DRONE_RADIUS = 0.046


TOPO_LABELS_PL = {
    'grid': 'Siatka regularna',
    'poisson': 'Rozklad Poissona',
    'cluster': 'Klastry gaussowskie',
    'building': 'Wnetrze budynku (BSP)',
    'mix': 'Mix topologii',
}

MODEL_LABELS_SHORT = {
    'paper_baseline_8drones_s0': 'Paper baseline (SDF)',
    'perception_limited_r1.0_8drones_s0': 'Perception r=1.0',
    'perception_limited_r0.2_8drones_s0': 'Perception r=0.2',
    'multiranger_r4.0_8drones_s0': 'Multiranger',
}


def formation_2x4(center_pos, spacing=FORMATION_SPACING):
    """Zwraca 8 pozycji formacji 2x4 (2 rows x 4 cols), centered at center_pos."""
    positions = []
    for row in range(2):
        for col in range(4):
            dx = (col - 1.5) * spacing
            dy = (row - 0.5) * spacing
            positions.append(center_pos + np.array([dx, dy, 0.0]))
    return np.array(positions)


def build_cfg(model, topology, train_dir, seed_start):
    argv = [
        '--algo=APPO', '--env=quadrotor_multi',
        f'--experiment={model}', f'--train_dir={train_dir}',
        '--device=cpu',
        '--max_num_episodes=1', '--eval_deterministic=False',
        '--no_render', '--quads_render=False', '--quads_use_numba=True',
        f'--quads_episode_duration={EPISODE_DURATION}',
        '--quads_mode=mix',
        '--quads_room_dims', str(BIG_ROOM_XY), str(BIG_ROOM_XY), str(BIG_ROOM_Z),
        '--quads_use_obstacles=True',
        '--quads_obst_spawn_area', str(BIG_SPAWN_AREA), str(BIG_SPAWN_AREA),
        '--quads_obst_density=0.2', '--quads_obst_size=0.6',
        f'--quads_obst_topology={topology}',
        f'--quads_topology_seed={seed_start}',
        '--quads_domain_random=False', '--quads_obst_density_random=False',
        '--quads_obst_size_random=False',
        '--replay_buffer_sample_prob=0.0', '--with_wandb=False',
    ]
    cfg = parse_swarm_cfg(argv=argv, evaluation=True)
    cfg = load_from_checkpoint(cfg)
    cfg.no_render = True
    cfg.num_envs = 1
    cfg.quads_render = False
    cfg.quads_room_dims = [BIG_ROOM_XY, BIG_ROOM_XY, BIG_ROOM_Z]
    cfg.quads_obst_spawn_area = [BIG_SPAWN_AREA, BIG_SPAWN_AREA]
    cfg.quads_episode_duration = EPISODE_DURATION
    cfg.quads_obst_density = 0.2
    cfg.quads_obst_size = 0.6
    cfg.quads_obst_topology = topology
    cfg.quads_topology_seed = seed_start
    return cfg


def load_policy(cfg, env):
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


def run_episode(env, actor_critic, device, cfg, env_info,
                start_center, goal_pos):
    """Zwraca (traj, obst_pos, goals, crashes, collision_events, reached_flags).

    HACK: po env.reset() nadpisujemy pozycje dronow (formacja 2x4) oraz
    ich goals (wspolny cel). Pierwsze klatki obs polityki sa stale
    (z pre-reset), ale polityka szybko sie stabilizuje.
    """
    obs, _ = env.reset()
    inner = env.unwrapped
    n_agents = inner.num_agents

    # Nadpisujemy positions i goals PO env.reset() (bez ponownego reset())
    formation_pos = formation_2x4(start_center)
    for i, e in enumerate(inner.envs):
        e.dynamics.pos = formation_pos[i].astype(np.float64)
        e.dynamics.vel = np.zeros(3)
        e.dynamics.rot = np.eye(3)
        e.dynamics.omega = np.zeros(3)
        e.goal = goal_pos.copy()
        inner.pos[i, :] = formation_pos[i]

    goals = np.array([goal_pos.copy() for _ in range(n_agents)])
    obst_pos = np.array(inner.obstacles.pos_arr) if inner.obstacles is not None else np.zeros((0, 3))

    rnn_states = torch.zeros([n_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    traj = [inner.pos.copy()]
    crashes = {'wall': np.zeros(n_agents, dtype=bool),
               'floor': np.zeros(n_agents, dtype=bool),
               'obst': np.zeros(n_agents, dtype=bool)}
    collision_events = {'obst': [[] for _ in range(n_agents)],
                        'wall': [[] for _ in range(n_agents)],
                        'floor': [[] for _ in range(n_agents)]}
    per_drone_done = np.zeros(n_agents, dtype=bool)
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

            pre_step_pos = inner.pos.copy()
            was_done = per_drone_done.copy()

            obs, rew, terminated, truncated, infos = env.step(actions)
            dones = make_dones(terminated, truncated).cpu().numpy()

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

            # Zaznacz reached, i automatycznie "zamroz" drona po dotarciu do celu
            # — bez tego drone po reached moze kontynuowac lot i spasc, zaburzajac
            # crash statistics (drone i tak dotarl, sukces).
            just_reached = np.zeros(n_agents, dtype=bool)
            for i in range(n_agents):
                if not was_done[i] and inner.reached_goal[i] and not reached_flags[i]:
                    reached_flags[i] = True
                    just_reached[i] = True

            curr_pos = inner.pos.copy()
            prev_frame = traj[-1]
            for i in range(n_agents):
                if was_done[i]:
                    curr_pos[i] = prev_frame[i]
                elif just_reached[i]:
                    # Zamrazamy w miejscu gdzie drone dotarl do celu
                    curr_pos[i] = pre_step_pos[i]
                    per_drone_done[i] = True
                elif dones[i]:
                    curr_pos[i] = pre_step_pos[i]
                    per_drone_done[i] = True
            traj.append(curr_pos)

            if bool(np.all(per_drone_done)):
                break

    return np.array(traj), obst_pos, goals, crashes, collision_events, reached_flags


def draw_episode(ax, traj, obst_pos, goals, crashes, collision_events,
                  reached_flags, obst_size, ep_idx, start_center, goal_pos):
    hx = BIG_SPAWN_AREA / 2
    hy = BIG_SPAWN_AREA / 2

    ax.add_patch(patches.Rectangle(
        (-hx, -hy), BIG_SPAWN_AREA, BIG_SPAWN_AREA,
        linewidth=1.0, edgecolor='dimgray', linestyle='--',
        facecolor='#fafafa', zorder=0,
    ))

    # Faint grid co 5m dla big env
    for x in np.arange(-hx, hx + 0.1, 5.0):
        ax.axvline(x, color='lightgray', linewidth=0.3, alpha=0.4, zorder=1)
    for y in np.arange(-hy, hy + 0.1, 5.0):
        ax.axhline(y, color='lightgray', linewidth=0.3, alpha=0.4, zorder=1)

    # Formation-start area highlighted
    ax.add_patch(patches.Circle(
        (start_center[0], start_center[1]), 3.0,
        facecolor='lightgreen', edgecolor='green', linestyle=':',
        linewidth=1.0, alpha=0.25, zorder=0.5,
    ))
    ax.text(start_center[0], start_center[1] - 3.5, 'START',
            fontsize=8, ha='center', color='darkgreen', fontweight='bold', zorder=1)

    # Goal region highlighted
    ax.add_patch(patches.Circle(
        (goal_pos[0], goal_pos[1]), 2.0,
        facecolor='lightyellow', edgecolor='goldenrod', linestyle=':',
        linewidth=1.0, alpha=0.35, zorder=0.5,
    ))
    ax.text(goal_pos[0], goal_pos[1] + 2.5, 'GOAL',
            fontsize=8, ha='center', color='darkgoldenrod', fontweight='bold', zorder=1)

    # Obstacles
    for x, y, _ in obst_pos:
        ax.add_patch(patches.Circle(
            (x, y), obst_size / 2,
            facecolor='#2E2E2E', edgecolor='#111', linewidth=0.3,
            alpha=0.75, zorder=2,
        ))

    n_agents = traj.shape[1]
    tab_cmap = plt.get_cmap('tab10')

    for i in range(n_agents):
        drone_color = tab_cmap(i % 10)
        xs = traj[:, i, 0]
        ys = traj[:, i, 1]

        ax.plot(xs, ys, color=drone_color, linewidth=1.0, alpha=0.85, zorder=3)
        ax.plot(xs[0], ys[0], marker='^', color=drone_color, markersize=6,
                markeredgecolor='black', markeredgewidth=0.4, zorder=5)

        # Wspolny goal (nakladajace sie kolka) — tylko raz per goal
        gx, gy = goals[i, 0], goals[i, 1]
        ax.add_patch(patches.Circle(
            (gx, gy), DRONE_RADIUS * 4,
            facecolor=drone_color, edgecolor='black', linewidth=0.4,
            alpha=0.6, zorder=5,
        ))

        for cx, cy in collision_events['obst'][i]:
            ax.plot(cx, cy, marker='*', color='#D62728', markersize=6,
                    markeredgecolor='black', markeredgewidth=0.3,
                    alpha=0.9, zorder=4)
        for cx, cy in collision_events['wall'][i]:
            ax.plot(cx, cy, marker='X', color='#D62728', markersize=5,
                    markeredgecolor='black', markeredgewidth=0.3,
                    alpha=0.9, zorder=4)

        end_x, end_y = xs[-1], ys[-1]
        if crashes['floor'][i]:
            ax.plot(end_x, end_y, marker='v', color='#D62728',
                    markersize=8, markeredgecolor='black', markeredgewidth=0.5,
                    zorder=6)
        elif crashes['wall'][i]:
            ax.plot(end_x, end_y, marker='X', color='#D62728',
                    markersize=7, markeredgecolor='black', markeredgewidth=0.4,
                    zorder=6)

    # Label
    n_floor = int(crashes['floor'].sum())
    n_wall = int(crashes['wall'].sum())
    n_obst_events = sum(len(collision_events['obst'][i]) for i in range(n_agents))
    # Reached ma priorytet nad crashed (drone ktorego reached_flags=True zostal
    # zamrozony w tej pozycji, wiec crashy z pozniejszych klatek juz nie zbieramy;
    # ale zabezpieczamy przed edge case: reached AND crashed w tej samej klatce).
    reached_only = reached_flags & ~(crashes['floor'] | crashes['wall'])
    crashed_only = (crashes['floor'] | crashes['wall']) & ~reached_flags
    n_reached = int(reached_only.sum())
    n_crashed = int(crashed_only.sum())
    n_stuck = max(0, n_agents - n_reached - n_crashed)
    label = f'ep={ep_idx}  reached:{n_reached}/{n_agents}'
    if n_stuck: label += f' stuck:{n_stuck}'
    if n_obst_events: label += f' *:{n_obst_events}'
    if n_floor: label += f' fl:{n_floor}'
    if n_wall: label += f' w:{n_wall}'
    ax.text(hx - 1.0, -hy + 1.0, label,
            fontsize=7, ha='right', va='bottom', color='dimgray',
            bbox=dict(boxstyle='round,pad=0.25',
                     facecolor='white', edgecolor='lightgray', alpha=0.9))

    ax.set_xlim(-hx - 1.5, hx + 1.5)
    ax.set_ylim(-hy - 1.5, hy + 1.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--model', default='multiranger_r4.0_8drones_s0')
    parser.add_argument('--topology', default='building',
                        choices=['grid', 'poisson', 'cluster', 'building', 'mix'])
    parser.add_argument('--rows', type=int, default=1,
                        help='Domyslnie 1 (jeden epizod = 1 duzy panel). Zwieksz jesli chcesz porownac wiele seedow.')
    parser.add_argument('--cols', type=int, default=1)
    parser.add_argument('--seed_start', type=int, default=42)
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

    # Formacja startowa w rogu (-x, -y), cel w przeciwlegly rog (+x, +y).
    # Margines 4m od brzegu spawn area, wysokosc 1.5m (bezpieczna nad podloga).
    start_center = np.array([-BIG_SPAWN_AREA / 2 + 4.0, -BIG_SPAWN_AREA / 2 + 4.0, 1.5])
    goal_pos = np.array([BIG_SPAWN_AREA / 2 - 4.0, BIG_SPAWN_AREA / 2 - 4.0, 1.5])
    distance = float(np.linalg.norm(goal_pos - start_center))

    n = args.rows * args.cols
    print(f'Long-flight: model={args.model}, topology={args.topology}')
    print(f'  Environment: spawn={BIG_SPAWN_AREA}x{BIG_SPAWN_AREA}m, room={BIG_ROOM_XY}x{BIG_ROOM_XY}m')
    print(f'  Start center: {start_center.tolist()}')
    print(f'  Goal: {goal_pos.tolist()}')
    print(f'  Diagonal distance: {distance:.1f}m')
    print(f'  Episode duration: {EPISODE_DURATION}s')
    print(f'  Running {n} epizodow (seedy {args.seed_start}..{args.seed_start + n - 1})')

    # Skalowanie: pojedynczy panel = 10x10 cali (duzy). Wielokrotne = 4x4 per panel.
    if n == 1:
        panel_size = 10
    else:
        panel_size = 4
    fig, axes = plt.subplots(args.rows, args.cols,
                              figsize=(panel_size * args.cols, panel_size * args.rows))
    if n == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i in range(n):
        traj, obst_pos, goals, crashes, coll_events, reached = run_episode(
            env, actor_critic, device, cfg, env_info, start_center, goal_pos)
        draw_episode(axes[i], traj, obst_pos, goals, crashes, coll_events,
                      reached, cfg.quads_obst_size, ep_idx=i,
                      start_center=start_center, goal_pos=goal_pos)
        n_agents = crashes['floor'].shape[0]
        reached_only = reached & ~(crashes['floor'] | crashes['wall'])
        crashed_only = (crashes['floor'] | crashes['wall']) & ~reached
        n_reached = int(reached_only.sum())
        n_crashed = int(crashed_only.sum())
        n_stuck = max(0, n_agents - n_reached - n_crashed)
        n_obst_events = sum(len(evs) for evs in coll_events['obst'])
        print(f'  ep {i+1}/{n}: T={len(traj)} frames, n_obst={len(obst_pos)}, '
              f'reached={n_reached}/{n_agents}, stuck={n_stuck}, '
              f'obst_col_events={n_obst_events}, fl={int(crashes["floor"].sum())}, '
              f'w={int(crashes["wall"].sum())}')

    env.close()

    model_label = MODEL_LABELS_SHORT.get(args.model, args.model)
    topo_label = TOPO_LABELS_PL.get(args.topology, args.topology)
    fig.suptitle(
        f'Long-range flight: {model_label} @ {topo_label}\n'
        f'{n} epizodow (seedy {args.seed_start}..{args.seed_start + n - 1}), '
        f'obszar {int(BIG_SPAWN_AREA)}x{int(BIG_SPAWN_AREA)}m, '
        f'dystans START→GOAL {distance:.1f}m',
        fontsize=12, y=0.995, fontweight='bold',
    )
    plt.tight_layout(rect=[0, 0, 1, 0.98])

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'long_flight_{args.model}_{args.topology}.png'
    )
    plt.savefig(out_path, dpi=140, bbox_inches='tight', facecolor='white')
    print(f'Wrote: {out_path}')


if __name__ == '__main__':
    main()
