"""
Zrzuty z symulatora: dwa widoki trojwymiarowe obok siebie.

Cel: pokazac czytelnikowi jak faktycznie wyglada srodowisko, zamiast samych
plaskich wykresow punktowych.

Lewy panel  — siatka regularna
Prawy panel — wnetrze budynku

Renderowanie idzie przez wbudowany modul wizualizacji (OpenGL) w trybie
'rgb_array', ktory rysuje do bufora poza ekranem (FBO). Nie otwiera okna,
wiec dziala takze bez interakcji — potrzebny jest jedynie dostepny DISPLAY.

Kamera: widok narozny (`corner0`), obejmujacy cale pomieszczenie.
Klatka jest lapana po `--settle` krokach polityki, zeby drony byly w locie,
a nie zawieszone w pozycjach startowych.

Zgodnie z zamowieniem rysunek NIE zawiera zadnego tekstu — podpisy paneli
i opis ida z LaTeX-a. Flaga --labels wlacza podpisy do celow roboczych.

Uzycie:
    python figures/simulator_views.py
    python figures/simulator_views.py --settle 400 --seed 7
    python figures/simulator_views.py --topologies grid cluster --labels

Output: figures/simulator_views.png
"""

import argparse
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict

from swarm_rl.train import parse_swarm_cfg, register_swarm_components


SPAWN_AREA = 8.0

# Model uzyty tylko po to, zeby drony sensownie latały na zrzucie.
# Nie wplywa na wyglad samego srodowiska.
DEFAULT_MODEL = 'multiranger_r4.0_topomix_8drones_s0_1B'

TOPO_LABELS_PL = {
    'grid': 'Siatka regularna',
    'poisson': 'Rozkład Poissona',
    'cluster': 'Klastry gaussowskie',
    'building': 'Wnętrze budynku',
}


def build_cfg(model, topology, train_dir, seed, view_mode):
    argv = [
        '--algo=APPO', '--env=quadrotor_multi',
        f'--experiment={model}', f'--train_dir={train_dir}',
        '--device=cpu',
        '--max_num_episodes=1', '--eval_deterministic=False',
        '--quads_use_numba=True',
        '--quads_episode_duration=15.0', '--quads_mode=mix',
        '--quads_room_dims', '10', '10', '10',
        '--quads_use_obstacles=True',
        '--quads_obst_spawn_area', str(SPAWN_AREA), str(SPAWN_AREA),
        '--quads_obst_density=0.2', '--quads_obst_size=0.6',
        f'--quads_obst_topology={topology}',
        f'--quads_topology_seed={seed}',
        '--quads_domain_random=False', '--quads_obst_density_random=False',
        '--quads_obst_size_random=False',
        '--replay_buffer_sample_prob=0.0', '--with_wandb=False',
        '--quads_view_mode', view_mode,
        '--quads_render=True',
    ]
    cfg = parse_swarm_cfg(argv=argv, evaluation=True)
    cfg = load_from_checkpoint(cfg)
    cfg.num_envs = 1
    cfg.no_render = False
    cfg.quads_render = True
    cfg.quads_view_mode = [view_mode]
    cfg.quads_obst_topology = topology
    cfg.quads_topology_seed = seed
    cfg.quads_obst_spawn_area = [SPAWN_AREA, SPAWN_AREA]
    cfg.quads_obst_density = 0.2
    cfg.quads_obst_size = 0.6
    return cfg


class FixedCamera:
    """Kamera o stalym polozeniu, podstawiana w miejsce wbudowanych.

    Wbudowane kamery (`corner*`, `global`, `topdown`) stoja wewnatrz albo tuz
    przy pomieszczeniu, przez co kadr wypelniaja slupy przeszkod ciagnace sie
    od podlogi po sufit, a drony sa niewidoczne. Tutaj ustawiamy oko poza
    pomieszczeniem i patrzymy na jego srodek, zeby zlapac calosc obszaru.

    Interfejs musi byc zgodny z wbudowanymi: reset()/step() sa wolane przez
    scene z roznymi argumentami zaleznie od trybu, wiec przyjmuja cokolwiek.
    """

    def __init__(self, eye, center, up=(0.0, 0.0, 1.0)):
        self.eye = np.asarray(eye, dtype=np.float64)
        self.center = np.asarray(center, dtype=np.float64)
        self.up = np.asarray(up, dtype=np.float64)

    def reset(self, *args, **kwargs):
        pass

    def step(self, *args, **kwargs):
        pass

    def look_at(self):
        return self.eye, self.center, self.up


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


def capture(model, topology, train_dir, seed, settle, view_mode,
            eye=None, look_at=None, trace_steps=60, viz_traces=90,
            trail_scale=1.0):
    """Zwraca klatke RGB (H, W, 3).

    Przebieg:
      1. `settle` krokow polityki BEZ renderowania (szybkie) — drony rozlatuja
         sie z pozycji startowych,
      2. `trace_steps` krokow Z renderowaniem — slady lotu (`path_store`)
         narastaja tylko przy wywolaniu render(), a bez nich osmiu malutkich
         dronow praktycznie nie widac w skali pomieszczenia,
      3. klatka koncowa.

    Scena powstaje leniwie, wiec tworzymy ja jawnie (init_scene_multi), zeby
    ustawic dlugosc sladu i kamere ZANIM zbuduje sie geometria.
    """
    cfg = build_cfg(model, topology, train_dir, seed, view_mode)
    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0),
        render_mode='rgb_array')
    env_info = extract_env_info(env, cfg)
    actor_critic, device = load_policy(cfg, env)

    inner = env.unwrapped
    np.random.seed(seed)
    for i, e in enumerate(inner.envs):
        e.np_random = np.random.default_rng(seed * 1000 + i)

    obs, _ = env.reset()
    inner.render_mode = 'rgb_array'
    n = inner.num_agents
    rnn_states = torch.zeros([n, get_rnn_size(cfg)], dtype=torch.float32, device=device)

    def policy_step(o, rnn):
        norm_obs = prepare_and_normalize_obs(actor_critic, o)
        out = actor_critic(norm_obs, rnn)
        a = out['actions']
        if a.ndim == 1:
            a = unsqueeze_tensor(a, dim=-1)
        a = preprocess_actions(env_info, a)
        return env.step(a)[0], out['new_rnn_states']

    with torch.no_grad():
        for _ in range(settle):
            obs, rnn_states = policy_step(obs, rnn_states)

    def grab():
        # Renderer pomija klatki, gdy nie nadaza za czasem rzeczywistym
        # (render_skip_frames), i wtedy render() zwraca None. Przy zrzucie
        # do pliku tempo nie ma znaczenia, wiec zerujemy licznik.
        inner.render_skip_frames = 0
        inner.render_every_nth_frame = 1
        inner.simulation_start_time = 0
        return inner.render()

    # Scena jawnie, ZANIM powstanie geometria — inaczej dlugosci sladu
    # nie da sie juz zmienic (bufor tworzy sie w _make_scene).
    if not inner.scenes:
        inner.init_scene_multi()
    for sc in inner.scenes:
        sc.viz_traces = viz_traces
        if eye is not None:
            sc.chase_cam = FixedCamera(eye, look_at)
        if trail_scale != 1.0:
            # UWAGA co do zakresu dzialania: korpus drona buduje
            # quadrotor_3dmodel(model) z parametrow modelu fizycznego i NIE
            # korzysta ze scene.diameter — samego drona nie da sie wiec
            # powiekszyc bez ruszania modelu dynamiki, czego tu nie robimy.
            # scene.diameter steruje natomiast kulami sladu lotu
            # (path_sphere = sphere(0.15 * diameter)), wiec podbicie go
            # pogrubia SLAD, dzieki czemu widac gdzie leca drony.
            # Crazyflie ma ~0.1 m rozpietosci przy pomieszczeniu 10 m, wiec
            # w kadrze na caly obszar sam korpus zostaje sub-pikselowy.
            _orig = sc.update_goal_diameter

            def _patched(_sc=sc, _o=_orig):
                _o()
                _sc.diameter *= trail_scale

            sc.update_goal_diameter = _patched

    # Slady narastaja wylacznie przy render(), wiec ostatnie kroki
    # wykonujemy z renderowaniem.
    frame = grab()
    with torch.no_grad():
        for _ in range(trace_steps):
            obs, rnn_states = policy_step(obs, rnn_states)
            frame = grab()

    if isinstance(frame, (list, tuple)):
        frame = frame[0]
    frame = None if frame is None else np.asarray(frame).astype(np.uint8)
    env.close()
    return frame


def trim_uniform_border(img, tol=6):
    """Obcina jednolita ramke wokol kadru (renderer zostawia marginesy tla)."""
    if img is None:
        return None
    bg = img[0, 0].astype(np.int16)
    diff = np.abs(img.astype(np.int16) - bg).sum(axis=2)
    mask = diff > tol
    if not mask.any():
        return img
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    pad = 8
    r0 = max(0, rows[0] - pad)
    r1 = min(img.shape[0], rows[-1] + 1 + pad)
    c0 = max(0, cols[0] - pad)
    c1 = min(img.shape[1], cols[-1] + 1 + pad)
    return img[r0:r1, c0:c1]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--topologies', nargs='+', default=['grid', 'building'])
    p.add_argument('--model', default=DEFAULT_MODEL)
    p.add_argument('--seed', type=int, default=1234)
    p.add_argument('--settle', type=int, default=200,
                   help='Kroki polityki bez renderowania (drony rozlatują się)')
    p.add_argument('--trace_steps', type=int, default=60,
                   help='Kroki z renderowaniem, budujące ślady lotu')
    p.add_argument('--viz_traces', type=int, default=90,
                   help='Długość śladu lotu w punktach')
    p.add_argument('--trail_scale', type=float, default=9.0,
                   help='Pogrubienie śladu lotu. Korpusy dronów zostają w skali '
                        'rzeczywistej (renderer buduje je z modelu fizycznego), '
                        'więc to ślad pokazuje gdzie lecą drony')
    p.add_argument('--view', default='corner0',
                   help='Tryb kamery bazowej (podmieniany przez --eye)')
    p.add_argument('--eye', nargs=3, type=float, default=[11.5, -11.5, 15.0],
                   help='Położenie kamery [x y z]; puste = kamera wbudowana')
    p.add_argument('--look_at', nargs=3, type=float, default=[0.0, 0.0, 1.5],
                   help='Punkt, na który patrzy kamera [x y z]')
    p.add_argument('--builtin_cam', action='store_true',
                   help='Użyj kamery wbudowanej zamiast stałej (--eye ignorowane)')
    p.add_argument('--labels', action='store_true',
                   help='Dodaj podpisy paneli (domyślnie brak, podpis idzie z LaTeX-a)')
    p.add_argument('--no_trim', action='store_true',
                   help='Nie obcinaj jednolitej ramki wokół kadru')
    p.add_argument('--train_dir', default='train_dir')
    p.add_argument('--out', default=None)
    args = p.parse_args()

    register_swarm_components()

    eye = None if args.builtin_cam else args.eye
    cam_desc = (f'kamera wbudowana {args.view}' if eye is None
                else f'kamera {eye} → {args.look_at}')

    frames = []
    for topo in args.topologies:
        print(f'Renderowanie: {topo} ({cam_desc}, {args.settle} kroków)...')
        f = capture(args.model, topo, args.train_dir, args.seed,
                    args.settle, args.view, eye=eye, look_at=args.look_at,
                    trace_steps=args.trace_steps, viz_traces=args.viz_traces,
                    trail_scale=args.trail_scale)
        if f is None:
            print(f'  [BŁĄD] Renderer nie zwrócił klatki dla {topo}.')
            print('         Sprawdź czy DISPLAY jest dostępny (echo $DISPLAY).')
            return
        if not args.no_trim:
            f = trim_uniform_border(f)
        print(f'  klatka {f.shape[1]}×{f.shape[0]}')
        frames.append(f)

    n = len(frames)
    fig, axes = plt.subplots(1, n, figsize=(8.0 * n, 4.6))
    if n == 1:
        axes = [axes]
    for ax, img, topo in zip(axes, frames, args.topologies):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if args.labels:
            ax.set_title(TOPO_LABELS_PL.get(topo, topo),
                         fontsize=12, fontweight='bold', pad=8)

    plt.subplots_adjust(wspace=0.02, left=0, right=1, top=1, bottom=0)
    plt.tight_layout(pad=0.3)

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'simulator_views.png')
    plt.savefig(out, dpi=160, bbox_inches='tight', facecolor='white',
                pad_inches=0.02)
    print(f'Wrote: {out}')


if __name__ == '__main__':
    main()
