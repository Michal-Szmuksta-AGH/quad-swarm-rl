"""
Rozklad wynikow epizodow per model i topologia.

DEFINICJA METRYKI (gym_art/quadrotor_multi/quadrotor_multi.py:768-779):

    sukces  = dotarl do celu  AND  brak kolizji z dronem  AND  brak kolizji z przeszkoda
    deadlock = brak kolizji  AND  NIE dotarl do celu
    kolizja  = kolizja z dronem LUB z przeszkoda

Te trzy kategorie sa ROZLACZNE i sumuja sie DOKLADNIE do 1.0 (zweryfikowane
empirycznie na wszystkich komorkach: suma = 1.0000).

Segment "kolizja" rozbijamy dokladnie przez zasade wlaczen i wylaczen:
    P(oba)       = P(przeszkoda) + P(sasiad) - P(kolizja)
    P(tylko przeszkoda) = P(przeszkoda) - P(oba)
    P(tylko sasiad)     = P(sasiad)     - P(oba)
Rozklad jest scisly (sprawdzone: zadna skladowa nie wychodzi ujemna).

UPADKI NA PODLOGE / UDERZENIA W SCIANE NIE NALEZA DO TEGO PODZIALU.
Kryterium sukcesu ich nie obejmuje — dron moze dotrzec do celu, potem spasc
na podloge i nadal liczyc sie jako sukces (flaga reached_goal jest trwala).
Dlatego rysujemy je jako OSOBNY, waski slupek obok, a nie jako wycinek stosu.

Output: figures/failure_mode_breakdown.png
Run: python figures/failure_mode_breakdown.py                    # 4 topologie side-by-side
     python figures/failure_mode_breakdown.py --topology building # tylko budynek
"""

import argparse
import json
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


TRAIN_DIR = '/home/michal/Desktop/MSc-thesis/3-Quad-Swarm-RL/quad-swarm-rl/train_dir'

MODELS = [
    ('paper_baseline_8drones_s0',              'Model bazowy\n(SDF, r=∞)'),
    ('perception_limited_r1.0_8drones_s0',     'Ograniczona\npercepcja r=1.0'),
    ('perception_limited_r0.2_8drones_s0',     'Ograniczona\npercepcja r=0.2'),
    ('multiranger_r4.0_8drones_s0',            'Multi-ranger\n(tylko siatka)'),
    # Checkpoint 1B (nie 1.5B) — rowny budzet treningowy z pozostalymi 4 modelami.
    ('multiranger_r4.0_topomix_8drones_s0_1B', 'Multi-ranger\n+ mieszanka topologii'),
]

TOPOLOGIES = ['grid', 'poisson', 'cluster', 'building']

TOPO_LABELS_PL = {
    'grid': 'Siatka regularna',
    'poisson': 'Rozkład Poissona',
    'cluster': 'Klastry gaussowskie',
    'building': 'Wnętrze budynku',
}

N_AGENTS = 8.0

# Podzial rozlaczny (stack od dolu do gory) — sumuje sie do 1.0
PARTITION = [
    ('success',   '#2CA02C', 'Sukces (dolot, bez zderzeń z dronem i przeszkodą)'),
    ('obst_only', '#FF7F0E', 'Kolizja z przeszkodą'),
    ('both',      '#8C4A2F', 'Kolizja z przeszkodą i sąsiadem'),
    ('nbr_only',  '#9467BD', 'Kolizja z sąsiadem (dron-dron)'),
    ('deadlock',  '#9E9E9E', 'Zakleszczenie (nie dotarł, bez kolizji)'),
]

# Metryki nakladkowe — POZA podzialem, rysowane osobnym slupkiem
OVERLAY = [
    ('floor', '#D62728', 'Upadek na podłogę'),
    ('wall',  '#5A1010', 'Uderzenie w ścianę'),
]


def load_metric(model, topology, key):
    """Load one metric from eval_metrics_topo_<topology>.json."""
    path = os.path.join(TRAIN_DIR, model, f'eval_metrics_topo_{topology}.json')
    if not os.path.exists(path):
        return None
    d = json.load(open(path))
    return d.get('metrics', {}).get(key, {}).get('mean')


def load_breakdown(model, topology):
    """Zwraca (partition_dict, overlay_dict).

    partition_dict sumuje sie do 1.0 (z dokladnoscia do bledu numerycznego).
    overlay_dict to metryki NIE nalezace do podzialu (upadki/uderzenia w sciane).
    """
    success  = load_metric(model, topology, 'metric/agent_success_rate')
    if success is None:
        return None, None
    deadlock = load_metric(model, topology, 'metric/agent_deadlock_rate') or 0.0
    col      = load_metric(model, topology, 'metric/agent_col_rate') or 0.0
    obst     = load_metric(model, topology, 'metric/agent_obst_col_rate') or 0.0
    nbr      = load_metric(model, topology, 'metric/agent_neighbor_col_rate') or 0.0

    # Wlaczenia-wylaczenia: P(oba) = P(A) + P(B) - P(A lub B)
    both      = max(0.0, obst + nbr - col)
    obst_only = max(0.0, obst - both)
    nbr_only  = max(0.0, nbr - both)

    partition = {
        'success': success,
        'obst_only': obst_only,
        'both': both,
        'nbr_only': nbr_only,
        'deadlock': deadlock,
    }

    # Liczniki zdarzen per epizod (8 dronow) -> frakcja agentow.
    # crashed_floor/_wall odpalaja sie przy PRZEJSCIU w stan kolizji, wiec
    # licznik to liczba zdarzen, nie tikow. Clamp na wypadek wielokrotnych zdarzen.
    n_floor = load_metric(model, topology, 'num_collisions_with_floor') or 0.0
    n_wall  = load_metric(model, topology, 'num_collisions_with_wall') or 0.0
    overlay = {
        'floor': min(n_floor / N_AGENTS, 1.0),
        'wall':  min(n_wall / N_AGENTS, 1.0),
    }
    return partition, overlay


def draw_panel(ax, topology):
    xs = np.arange(len(MODELS))
    main_w, ov_w = 0.46, 0.16
    main_x = xs - 0.13
    ov_x = xs + 0.26

    # --- Slupek glowny: podzial rozlaczny ---
    bottoms = np.zeros(len(MODELS))
    for cat, color, _ in PARTITION:
        vals = []
        for model_dir, _ in MODELS:
            part, _ = load_breakdown(model_dir, topology)
            vals.append(part.get(cat, 0.0) if part else 0.0)
        vals = np.array(vals)
        ax.bar(main_x, vals, bottom=bottoms, color=color, width=main_w,
               edgecolor='white', linewidth=0.6, zorder=3)
        for i, (v, b) in enumerate(zip(vals, bottoms)):
            if v > 0.055:
                ax.text(main_x[i], b + v / 2, f'{100*v:.0f}%',
                        ha='center', va='center', fontsize=7.5,
                        color='white' if cat != 'obst_only' else '#3A2000',
                        fontweight='bold', zorder=4)
        bottoms += vals

    # Kontrola: podzial musi sumowac sie do 1.0
    for i, tot in enumerate(bottoms):
        if abs(tot - 1.0) > 5e-3:
            print(f'  [UWAGA] {topology}/{MODELS[i][0]}: podzial sumuje sie do {tot:.4f}')

    # --- Slupek nakladkowy: upadki / sciany (POZA podzialem) ---
    ov_bottoms = np.zeros(len(MODELS))
    for cat, color, _ in OVERLAY:
        vals = []
        for model_dir, _ in MODELS:
            _, ov = load_breakdown(model_dir, topology)
            vals.append(ov.get(cat, 0.0) if ov else 0.0)
        vals = np.array(vals)
        ax.bar(ov_x, vals, bottom=ov_bottoms, color=color, width=ov_w,
               edgecolor='white', linewidth=0.5, hatch='///', zorder=3)
        ov_bottoms += vals
    for i, tot in enumerate(ov_bottoms):
        if tot > 0.015:
            ax.text(ov_x[i], tot + 0.015, f'{100*tot:.0f}%',
                    ha='center', va='bottom', fontsize=7,
                    color='#7A1010', fontweight='bold', zorder=4)

    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in MODELS], fontsize=8, rotation=30,
                       ha='right', rotation_mode='anchor')
    ax.set_ylabel('Odsetek epizodów na drona', fontsize=10)
    ax.set_title(TOPO_LABELS_PL.get(topology, topology),
                 fontsize=11, fontweight='bold')
    ax.set_ylim(0, 1.06)
    ax.grid(True, axis='y', alpha=0.3, zorder=0)
    ax.set_axisbelow(True)


def build_legend_handles():
    handles = [Patch(facecolor=c, label=l) for _, c, l in PARTITION]
    handles.append(Patch(facecolor='none', edgecolor='none',
                         label='─── poza podziałem ───'))
    handles += [Patch(facecolor=c, hatch='///', label=l) for _, c, l in OVERLAY]
    return handles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--topology', default=None,
                        help='Która topologia (brak = wszystkie 4 obok siebie)')
    args = parser.parse_args()

    topos = [args.topology] if args.topology else TOPOLOGIES
    n = len(topos)

    if n == 1:
        fig, axes = plt.subplots(1, 1, figsize=(7.0, 6.6))
        draw_panel(axes, topos[0])
    else:
        fig, axs = plt.subplots(1, n, figsize=(4.2 * n, 7.0), sharey=True)
        for i, t in enumerate(topos):
            draw_panel(axs[i], t)

    plt.tight_layout(rect=[0, 0.14, 1, 1])

    leg = fig.legend(handles=build_legend_handles(),
                     loc='upper center', bbox_to_anchor=(0.5, 0.135),
                     fontsize=8.5, framealpha=0.95, ncol=4,
                     title='Lewy słupek: podział rozłączny (suma = 100%)   •   '
                           'Prawy słupek kreskowany: metryka nakładkowa',
                     title_fontsize=8)
    leg._legend_box.align = 'center'

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       f'failure_mode_breakdown'
                       f'{"_" + args.topology if args.topology else ""}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Wrote: {out}')


if __name__ == '__main__':
    main()
