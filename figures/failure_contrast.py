"""
Kontrast trybow porazki: dwa modele, TEN SAM epizod, obok siebie.

Cel rysunku: zamienic w obrazek roznice zmierzona liczbowo — model bazowy
forsuje przejscie i uderza w przeszkode, topomix zatrzymuje sie przed nia
(mniej kolizji, wiecej zakleszczen).

Oba panele dostaja IDENTYCZNE warunki:
  - ta sama topologia i to samo ziarno topologii,
  - te same pozycje startowe i te same cele (globalny np.random seedowany
    tuz przed env.reset(); skrypt WERYFIKUJE zgodnosc i ostrzega przy rozjezdzie),
  - ten sam numer epizodu.

Jeden dron jest wyrozniony pelnym kolorem, pozostalych siedem rysowanych
jasnoszaro — bez nich manewr omijania wyglada na bezprzyczynowy.

Definicja trybow (zgodna z metryka, patrz WYNIKI-EKSPERYMENTOW.md sekcja 4.8):
    kolizja      = zderzenie z przeszkoda LUB z dronem (po karencji 1.5 s)
    sukces       = dotarl do celu AND brak kolizji
    zakleszczenie = NIE dotarl do celu AND brak kolizji

Uzycie:
    # 1) przejrzyj ziarna i wybierz epizod z widocznym kontrastem
    python figures/failure_contrast.py --scan 16

    # 2) narysuj wybrany (--drone -1 = wybor automatyczny)
    python figures/failure_contrast.py --seed 1234 --drone -1

Output: figures/failure_contrast_<topologia>.png
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
DRONE_RADIUS = 0.046

# Lewy panel, prawy panel. Topomix z checkpointu 1B — rowny budzet treningowy
# z modelem bazowym (patrz WYNIKI-EKSPERYMENTOW.md sekcja 4.7).
DEFAULT_MODELS = [
    ('paper_baseline_8drones_s0', 'Model bazowy'),
    ('multiranger_r4.0_topomix_8drones_s0_1B', 'Multi-ranger, mieszanka topologii'),
]

TOPO_LABELS_PL = {
    'grid': 'siatka regularna',
    'poisson': 'rozkład Poissona',
    'cluster': 'klastry gaussowskie',
    'building': 'wnętrze budynku',
}

HIGHLIGHT_COLOR = '#1F5FBF'
OTHERS_COLOR = '#C7C7C7'


# --------------------------------------------------------------------------
# Symulacja
# --------------------------------------------------------------------------
def build_cfg(model, topology, train_dir, topology_seed):
    argv = [
        '--algo=APPO', '--env=quadrotor_multi',
        f'--experiment={model}', f'--train_dir={train_dir}',
        '--device=cpu',
        '--max_num_episodes=1', '--eval_deterministic=False',
        '--no_render', '--quads_render=False', '--quads_use_numba=True',
        '--quads_episode_duration=15.0', '--quads_mode=mix',
        '--quads_room_dims', '10', '10', '10',
        '--quads_use_obstacles=True',
        '--quads_obst_spawn_area', str(SPAWN_AREA), str(SPAWN_AREA),
        '--quads_obst_density=0.2', '--quads_obst_size=0.6',
        f'--quads_obst_topology={topology}',
        f'--quads_topology_seed={topology_seed}',
        '--quads_domain_random=False', '--quads_obst_density_random=False',
        '--quads_obst_size_random=False',
        '--replay_buffer_sample_prob=0.0', '--with_wandb=False',
    ]
    cfg = parse_swarm_cfg(argv=argv, evaluation=True)
    cfg = load_from_checkpoint(cfg)
    cfg.no_render = True
    cfg.num_envs = 1
    cfg.quads_render = False
    cfg.quads_obst_topology = topology
    cfg.quads_topology_seed = topology_seed
    cfg.quads_obst_spawn_area = [SPAWN_AREA, SPAWN_AREA]
    cfg.quads_obst_density = 0.2
    cfg.quads_obst_size = 0.6
    return cfg


def load_policy(cfg, env):
    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()
    device = torch.device('cpu')
    actor_critic.model_to_device(device)
    name_prefix = dict(latest='checkpoint', best='best')[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(
        Learner.checkpoint_dir(cfg, cfg.policy_index), f'{name_prefix}_*')
    ckpt = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(ckpt['model'])
    return actor_critic, device


def run_episode(env, actor_critic, device, cfg, env_info, episode_seed):
    """Odtwarza jeden epizod. Zwraca dict z trajektoriami i flagami trybow.

    Determinizm epizodu wymaga zaseedowania DWOCH niezaleznych generatorow:

      1. globalny np.random — z niego scenariusz losuje cele i punkty startowe
         (scenarios/base.py, scenarios/obstacles/o_base.py),
      2. per-env generator Gymnasium `np_random` — z niego bierze sie szum
         dodawany do punktu startowego (quadrotor_single.py:402:
         `self.np_random.uniform(-box, box) + self.spawn_point`).

    Zaseedowanie tylko pierwszego daje zgodne cele, ale rozjezdzone starty
    (roznica rzedu 0.15 m), bo drugi generator zyje wlasnym zyciem miedzy
    resetami. Zgodnosc obu paneli jest pozniej weryfikowana w main().
    """
    inner = env.unwrapped
    np.random.seed(episode_seed)
    for i, e in enumerate(inner.envs):
        e.np_random = np.random.default_rng(episode_seed * 1000 + i)

    obs, _ = env.reset()
    n = inner.num_agents

    goals = np.array([inner.envs[i].goal for i in range(n)])
    starts = inner.pos.copy()
    obst_pos = (np.array(inner.obstacles.pos_arr)
                if inner.obstacles is not None else np.zeros((0, 3)))
    grace = getattr(inner, 'collisions_grace_period_steps', 0)

    rnn_states = torch.zeros([n, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    traj = [inner.pos.copy()]

    col_obst = np.zeros(n, dtype=bool)
    col_nbr = np.zeros(n, dtype=bool)
    crashed_floor = np.zeros(n, dtype=bool)
    reached = np.zeros(n, dtype=bool)
    events_obst = [[] for _ in range(n)]
    done_flag = np.zeros(n, dtype=bool)

    with torch.no_grad():
        while True:
            norm_obs = prepare_and_normalize_obs(actor_critic, obs)
            out = actor_critic(norm_obs, rnn_states)
            actions = out['actions']
            if cfg.eval_deterministic:
                actions = argmax_actions(actor_critic.action_distribution())
            if actions.ndim == 1:
                actions = unsqueeze_tensor(actions, dim=-1)
            actions = preprocess_actions(env_info, actions)
            rnn_states = out['new_rnn_states']

            pre_pos = inner.pos.copy()
            was_done = done_flag.copy()
            tick = inner.envs[0].tick

            obs, _, terminated, truncated, _ = env.step(actions)
            dones = make_dones(terminated, truncated).cpu().numpy()

            # Kolizje licza sie dopiero po karencji — tak jak w metryce.
            if tick >= grace:
                for i in inner.curr_quad_col:
                    idx = int(i)
                    if not was_done[idx]:
                        col_obst[idx] = True
                        events_obst[idx].append(tuple(pre_pos[idx, :2]))
                for i in np.atleast_1d(inner.last_step_unique_collisions):
                    idx = int(i)
                    if 0 <= idx < n and not was_done[idx]:
                        col_nbr[idx] = True

            for i, e in enumerate(inner.envs):
                if not was_done[i] and e.dynamics.crashed_floor:
                    crashed_floor[i] = True
            for i in range(n):
                if not was_done[i] and inner.reached_goal[i]:
                    reached[i] = True

            # SF auto-resetuje agenta po done — zamrazamy trajektorie,
            # inaczej dostajemy skok do nowego spawnu.
            curr = inner.pos.copy()
            prev = traj[-1]
            for i in range(n):
                if was_done[i]:
                    curr[i] = prev[i]
                elif dones[i]:
                    curr[i] = pre_pos[i]
                    done_flag[i] = True
            traj.append(curr)

            if bool(np.all(done_flag)):
                break

    collided = col_obst | col_nbr
    return {
        'traj': np.array(traj),
        'obst': obst_pos,
        'goals': goals,
        'starts': starts,
        'col_obst': col_obst,
        'col_nbr': col_nbr,
        'collided': collided,
        'floor': crashed_floor,
        'reached': reached,
        'success': reached & ~collided,
        'deadlock': ~reached & ~collided,
        'events_obst': events_obst,
    }


class ModelSession:
    """Trzyma zaladowane srodowisko i polityke, zeby przeglad wielu ziaren
    nie przeladowywal checkpointu za kazdym razem (to dominujacy koszt)."""

    def __init__(self, model, topology, train_dir, topology_seed):
        self.cfg = build_cfg(model, topology, train_dir, topology_seed)
        self.env = make_env_func_batched(
            self.cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
            render_mode=None)
        self.env_info = extract_env_info(self.env, self.cfg)
        self.actor_critic, self.device = load_policy(self.cfg, self.env)

    def episode(self, seed):
        """Jeden epizod dla zadanego ziarna (topologia + uklad startowy)."""
        inner = self.env.unwrapped
        # Topologia = RandomState(obst_topology_seed + _topology_reset_counter),
        # wiec zerujemy licznik, zeby ziarno jednoznacznie wyznaczalo uklad.
        inner.obst_topology_seed = seed
        inner._topology_reset_counter = 0
        return run_episode(self.env, self.actor_critic, self.device,
                           self.cfg, self.env_info, seed)

    def close(self):
        self.env.close()


# --------------------------------------------------------------------------
# Wybor epizodu i drona
# --------------------------------------------------------------------------
def contrast_score(left, right):
    """Ile dronow pokazuje szukany kontrast: bazowy koliduje z przeszkoda,
    a ten sam dron w drugim modelu NIE koliduje."""
    return int(np.sum(left['col_obst'] & ~right['collided']))


def pick_drone(left, right):
    """Wybiera drona najlepiej ilustrujacego kontrast.

    Priorytet: bazowy koliduje z przeszkoda, a drugi model sie zakleszcza
    (najczystsza ilustracja wymiany kolizji na zatrzymanie). W drugiej
    kolejnosci: bazowy koliduje, drugi dolatuje bez kolizji.
    """
    ideal = np.where(left['col_obst'] & right['deadlock'])[0]
    if len(ideal):
        return int(ideal[0]), 'kolizja → zakleszczenie'
    good = np.where(left['col_obst'] & right['success'])[0]
    if len(good):
        return int(good[0]), 'kolizja → dolot bez kolizji'
    any_col = np.where(left['col_obst'])[0]
    if len(any_col):
        return int(any_col[0]), 'kolizja w modelu bazowym'
    return 0, 'brak wyraźnego kontrastu'


def mode_label(res, i):
    if res['col_obst'][i]:
        return 'kolizja z przeszkodą'
    if res['col_nbr'][i]:
        return 'kolizja z dronem'
    if res['deadlock'][i]:
        return 'zakleszczenie'
    if res['success'][i]:
        return 'dolot bez kolizji'
    return 'nieokreślony'


# --------------------------------------------------------------------------
# Rysowanie
# --------------------------------------------------------------------------
def draw_panel(ax, res, drone, obst_size, panel_title):
    hx = hy = SPAWN_AREA / 2

    ax.add_patch(patches.Rectangle(
        (-hx, -hy), SPAWN_AREA, SPAWN_AREA,
        linewidth=1.0, edgecolor='dimgray', linestyle='--',
        facecolor='#FCFCFC', zorder=0))
    for g in np.arange(-hx, hx + 0.01, 1.0):
        ax.axvline(g, color='#EDEDED', linewidth=0.4, zorder=0.5)
        ax.axhline(g, color='#EDEDED', linewidth=0.4, zorder=0.5)

    for x, y, _ in res['obst']:
        ax.add_patch(patches.Circle(
            (x, y), obst_size / 2, facecolor='#3A3A3A', edgecolor='#111',
            linewidth=0.5, zorder=2))

    traj = res['traj']
    n = traj.shape[1]

    # Pozostale drony — tlo kontekstowe, bez znacznikow
    for i in range(n):
        if i == drone:
            continue
        ax.plot(traj[:, i, 0], traj[:, i, 1], color=OTHERS_COLOR,
                linewidth=0.9, alpha=0.85, zorder=3, solid_capstyle='round')

    # Dron wyrozniony
    xs, ys = traj[:, drone, 0], traj[:, drone, 1]
    ax.plot(xs, ys, color=HIGHLIGHT_COLOR, linewidth=2.1, zorder=5,
            solid_capstyle='round')
    ax.plot(xs[0], ys[0], marker='^', color=HIGHLIGHT_COLOR, markersize=11,
            markeredgecolor='white', markeredgewidth=1.0, zorder=7)
    gx, gy = res['goals'][drone, 0], res['goals'][drone, 1]
    ax.plot(gx, gy, marker='o', markersize=13, markerfacecolor='none',
            markeredgecolor=HIGHLIGHT_COLOR, markeredgewidth=2.0, zorder=7)
    ax.plot(gx, gy, marker='+', markersize=8, color=HIGHLIGHT_COLOR,
            markeredgewidth=1.6, zorder=7)

    # Znaczniki trybu porazki wyroznionego drona
    for cx, cy in res['events_obst'][drone]:
        ax.plot(cx, cy, marker='X', color='#D62728', markersize=13,
                markeredgecolor='white', markeredgewidth=1.1, zorder=8)
    if res['deadlock'][drone]:
        ax.plot(xs[-1], ys[-1], marker='s', color='#7F7F7F', markersize=11,
                markeredgecolor='white', markeredgewidth=1.1, zorder=8)

    ax.set_xlim(-hx - 0.25, hx + 0.25)
    ax.set_ylim(-hy - 0.25, hy + 0.25)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(panel_title, fontsize=12, fontweight='bold', pad=10)


def build_legend(fig):
    from matplotlib.lines import Line2D
    h = [
        Line2D([0], [0], color=HIGHLIGHT_COLOR, lw=2.1,
               label='Trajektoria wyróżnionego drona'),
        Line2D([0], [0], color=OTHERS_COLOR, lw=1.2,
               label='Trajektorie pozostałych dronów'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=HIGHLIGHT_COLOR,
               markersize=10, label='Pozycja startowa'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='none',
               markeredgecolor=HIGHLIGHT_COLOR, markeredgewidth=2, markersize=11,
               label='Cel'),
        Line2D([0], [0], marker='X', color='w', markerfacecolor='#D62728',
               markersize=11, label='Kolizja z przeszkodą'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#7F7F7F',
               markersize=10, label='Zakleszczenie'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#3A3A3A',
               markersize=10, label='Przeszkoda'),
    ]
    fig.legend(handles=h, loc='lower center', bbox_to_anchor=(0.5, -0.045),
               ncol=4, fontsize=9.5, frameon=False,
               columnspacing=2.4, handletextpad=0.7)


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--topology', default='building',
                   choices=['grid', 'poisson', 'cluster', 'building'])
    p.add_argument('--seed', type=int, default=1234,
                   help='Ziarno topologii ORAZ epizodu (jedno dla obu paneli)')
    p.add_argument('--drone', type=int, default=-1,
                   help='Który dron wyróżnić (-1 = wybór automatyczny)')
    p.add_argument('--scan', type=int, default=0,
                   help='Przejrzyj N kolejnych ziaren i wypisz tabelę kontrastu')
    p.add_argument('--models', nargs=2, default=[m for m, _ in DEFAULT_MODELS])
    p.add_argument('--train_dir', default='train_dir')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    register_swarm_components()
    labels = [lbl for _, lbl in DEFAULT_MODELS]

    sess_l = ModelSession(args.models[0], args.topology, args.train_dir, args.seed)
    sess_r = ModelSession(args.models[1], args.topology, args.train_dir, args.seed)

    # ---------------- tryb przegladu ----------------
    if args.scan:
        print(f'\nPrzegląd {args.scan} ziaren, topologia: {args.topology}')
        print('kol/zakl/ok = liczba dronów z kolizją / zakleszczeniem / dolotem\n')
        print(f'{"ziarno":>7s} {"kontrast":>9s} {"bazowy kol/zakl/ok":>20s} '
              f'{"topomix kol/zakl/ok":>21s}')
        print('-' * 60)
        rows = []
        for s in range(args.seed, args.seed + args.scan):
            left = sess_l.episode(s)
            right = sess_r.episode(s)
            sc = contrast_score(left, right)
            rows.append((sc, s))
            print(f'{s:>7d} {sc:>9d} '
                  f'{int(left["collided"].sum()):>8d}/{int(left["deadlock"].sum()):>4d}/'
                  f'{int(left["success"].sum()):>3d} '
                  f'{int(right["collided"].sum()):>10d}/{int(right["deadlock"].sum()):>4d}/'
                  f'{int(right["success"].sum()):>3d}')
        sess_l.close()
        sess_r.close()
        rows.sort(reverse=True)
        print(f'\nNajlepsze ziarna: {[s for _, s in rows[:5]]}')
        print(f'Narysuj: python figures/failure_contrast.py '
              f'--topology {args.topology} --seed {rows[0][1]}')
        return

    # ---------------- tryb rysowania ----------------
    print(f'Symulacja, ziarno={args.seed}, topologia={args.topology}')
    left = sess_l.episode(args.seed)
    right = sess_r.episode(args.seed)
    sess_l.close()
    sess_r.close()

    # Weryfikacja: oba panele MUSZA pokazywac ten sam uklad epizodu
    d_start = float(np.abs(left['starts'] - right['starts']).max())
    d_goal = float(np.abs(left['goals'] - right['goals']).max())
    d_obst = (float(np.abs(left['obst'] - right['obst']).max())
              if left['obst'].shape == right['obst'].shape else float('inf'))
    print(f'  Zgodność warunków: starty Δ={d_start:.4f}  '
          f'cele Δ={d_goal:.4f}  przeszkody Δ={d_obst:.4f}')
    if max(d_start, d_goal, d_obst) > 1e-6:
        print('  [UWAGA] Panele NIE mają identycznych warunków początkowych.')
        print('          Rysunek nadal powstanie, ale nie jest uczciwym porównaniem.')

    drone = args.drone
    if drone < 0:
        drone, why = pick_drone(left, right)
        print(f'  Wyróżniony dron: {drone} ({why})')
    print(f'  Model bazowy → {mode_label(left, drone)}')
    print(f'  Topomix      → {mode_label(right, drone)}')

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.8))
    draw_panel(axes[0], left, drone, 0.6, labels[0])
    draw_panel(axes[1], right, drone, 0.6, labels[1])
    plt.tight_layout()
    build_legend(fig)
    out = args.out or os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        f'failure_contrast_{args.topology}.png')
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'Wrote: {out}')


if __name__ == '__main__':
    main()
