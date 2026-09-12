"""
Connectivity stats + wizualizacja rejected vs accepted topologii.

Generuje:
  figures/topology_connectivity_stats.png  — bar chart rejection rate + avg attempts
                                              (density sweep 0.1, 0.2, 0.3, 0.4)
  figures/topology_accepted_vs_rejected.png — sample panels: accepted + rejected
                                              (rejected z komponentami kolorowanymi)
  figures/topology_connectivity_stats.json  — surowe liczby do cytowania w pracy

Sample size N=200 per (topology, density) -> ~3200 generacji ~30s runtime.
Run: python figures/topology_connectivity_stats.py
"""

import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gym_art.quadrotor_multi.topology.generators import (
    TOPOLOGY_GENERATORS, generate_topology, label_components,
    _compute_bare_obst_map,
)


SPAWN = np.array([8.0, 8.0])
OBST_SIZE = 0.6
ROOM_H = 10.0
DENSITIES = [0.1, 0.2, 0.3, 0.4]
N_SAMPLES = 200
MAX_RETRIES = 10

TOPO_ORDER = ['grid', 'poisson', 'cluster', 'building']

TOPO_LABELS_PL = {
    'grid': 'siatka regularna',
    'poisson': 'rozkład Poissona',
    'cluster': 'klastry gaussowskie',
    'building': 'wnętrze budynku',
}


# --------------------------------------------------------------------------
# Stats collection
# --------------------------------------------------------------------------
def collect_stats():
    """Zwraca dict:
      stats[topology][density] = {
          'attempts': list of N attempts-to-success,
          'fallback_count': int,
          'rejected_examples': list of (obst_map, positions, bare_map, n_components),
          'accepted_examples': list of (obst_map, positions, bare_map)
      }
    """
    stats = defaultdict(lambda: defaultdict(dict))

    for topo in TOPO_ORDER:
        for density in DENSITIES:
            attempts_list = []
            fallback_count = 0
            accepted_examples = []
            rejected_examples = []

            for seed in range(N_SAMPLES):
                rng = np.random.RandomState(seed * 7919 + hash(topo) % 100000)
                _, pos_arr, _, s = generate_topology(
                    topo, SPAWN, density, OBST_SIZE, ROOM_H,
                    rng=rng, return_stats=True, keep_rejected=True,
                    max_retries=MAX_RETRIES,
                )
                attempts_list.append(s['attempts'])
                fallback_count += s['fallback']

                # Zbierz accepted example (max 4 per config)
                if s['fallback'] == 0 and len(accepted_examples) < 4:
                    positions = [(p[0], p[1]) for p in pos_arr]
                    bare = _compute_bare_obst_map(positions, SPAWN)
                    from gym_art.quadrotor_multi.topology.generators \
                        import _positions_to_obst_map
                    dilated = _positions_to_obst_map(positions, SPAWN)
                    accepted_examples.append((dilated, positions, bare))

                # Zbierz rejected examples (max fragmentation = most interesting)
                for rej_map, rej_pos, rej_bare in s.get('rejected_maps', []):
                    _, n_comp = label_components(rej_bare)
                    rejected_examples.append((rej_map, rej_pos, rej_bare, n_comp))

            # Sortuj rejected po max fragmentation, weź 4 top
            rejected_examples.sort(key=lambda x: x[3], reverse=True)
            rejected_examples = rejected_examples[:4]

            stats[topo][density] = {
                'attempts': attempts_list,
                'fallback_count': fallback_count,
                'rejected_examples': rejected_examples,
                'accepted_examples': accepted_examples,
            }
            mean_att = np.mean(attempts_list)
            reject_rate = sum(a - 1 for a in attempts_list) / sum(attempts_list) * 100
            fallback_rate = fallback_count / N_SAMPLES * 100
            print(f"  {topo:10s} density={density}: "
                  f"mean_attempts={mean_att:.2f}, reject_rate={reject_rate:.1f}%, "
                  f"fallback_rate={fallback_rate:.1f}%")

    return stats


# --------------------------------------------------------------------------
# Figure 1: bar chart rejection rates (density sweep)
# --------------------------------------------------------------------------
def figure_rejection_rates(stats, out_path='figures/topology_connectivity_stats.png'):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: reject rate per (topology, density)
    x = np.arange(len(TOPO_ORDER))
    width = 0.2
    colors = ['#4E79A7', '#F28E2B', '#E15759', '#76B7B2']

    ax = axes[0]
    for i, density in enumerate(DENSITIES):
        rates = [
            sum(a - 1 for a in stats[t][density]['attempts']) /
            sum(stats[t][density]['attempts']) * 100
            for t in TOPO_ORDER
        ]
        ax.bar(x + i * width - 1.5 * width, rates, width,
               label=f'zagęszczenie={density}', color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([TOPO_LABELS_PL[t] for t in TOPO_ORDER], fontsize=11)
    ax.set_ylabel('Odsetek odrzuceń (%)', fontsize=11)
    ax.set_title('Odsetek odrzuconych generacji\n(niespójne wolne przestrzenie)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, max(60, ax.get_ylim()[1]))

    # Right: mean attempts + fallback rate
    ax = axes[1]
    for i, density in enumerate(DENSITIES):
        mean_atts = [np.mean(stats[t][density]['attempts']) for t in TOPO_ORDER]
        ax.bar(x + i * width - 1.5 * width, mean_atts, width,
               label=f'zagęszczenie={density}', color=colors[i])
    ax.set_xticks(x)
    ax.set_xticklabels([TOPO_LABELS_PL[t] for t in TOPO_ORDER], fontsize=11)
    ax.set_ylabel('Średnia liczba prób generacji', fontsize=11)
    ax.set_title('Średnia liczba prób do sukcesu\n(1.0 = zawsze pierwsze przełożenie OK)',
                 fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=9)
    ax.grid(True, alpha=0.3, axis='y')
    ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'Wrote: {out_path}')


# --------------------------------------------------------------------------
# Figure 2: accepted vs rejected samples
# --------------------------------------------------------------------------
COMPONENT_COLORS = plt.cm.tab10.colors  # discrete 10-color palette


def draw_topology_with_components(ax, positions, bare_map, title):
    """Rysuje topologie z kolorowaniem wolnych komponentow (tab10)."""
    hx, hy = SPAWN[0] / 2, SPAWN[1] / 2

    # Backdrop: bounding rectangle
    ax.add_patch(patches.Rectangle(
        (-hx, -hy), SPAWN[0], SPAWN[1],
        linewidth=1.5, edgecolor='dimgray',
        facecolor='#fafafa', linestyle='--', zorder=0
    ))

    # Colored component overlay
    labeled, n_comp = label_components(bare_map)
    for comp_id in range(1, n_comp + 1):
        color = COMPONENT_COLORS[(comp_id - 1) % len(COMPONENT_COLORS)]
        # Cell-level overlay
        rs, cs = np.where(labeled == comp_id)
        for r, c in zip(rs, cs):
            cell_x = r + 0.5 - SPAWN[0] / 2
            cell_y = c + 0.5 - SPAWN[1] / 2
            ax.add_patch(patches.Rectangle(
                (cell_x - 0.5, cell_y - 0.5), 1.0, 1.0,
                facecolor=color, alpha=0.28, edgecolor='none', zorder=1
            ))

    # Faint grid
    for x in np.arange(-hx, hx + 0.1, 1.0):
        ax.axvline(x, color='lightgray', linewidth=0.3, alpha=0.5, zorder=2)
    for y in np.arange(-hy, hy + 0.1, 1.0):
        ax.axhline(y, color='lightgray', linewidth=0.3, alpha=0.5, zorder=2)

    # Obstacles
    for x, y in positions:
        ax.add_patch(patches.Circle(
            (x, y), OBST_SIZE / 2,
            facecolor='#2E2E2E', edgecolor='black',
            linewidth=0.6, alpha=0.9, zorder=3
        ))

    # Component count badge
    badge_color = '#2E7D32' if n_comp == 1 else '#C62828'
    badge_label = 'OK' if n_comp == 1 else f'{n_comp} komp.'
    ax.text(hx - 0.3, -hy + 0.3, badge_label,
            fontsize=9, ha='right', va='bottom', fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.35',
                     facecolor=badge_color, edgecolor='none'))

    ax.set_xlim(-hx - 0.3, hx + 0.3)
    ax.set_ylim(-hy - 0.3, hy + 0.3)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=10)


def figure_accepted_vs_rejected(stats, density=0.3,
                                out_path='figures/topology_accepted_vs_rejected.png'):
    """Pokaz accepted + rejected examples dla wybranej density (najciekawsza dla rejects)."""
    n_examples_per_row = 4  # 2 accepted + 2 rejected

    fig, axes = plt.subplots(len(TOPO_ORDER), n_examples_per_row,
                              figsize=(3.2 * n_examples_per_row, 3.2 * len(TOPO_ORDER)))

    for i, topo in enumerate(TOPO_ORDER):
        s = stats[topo][density]
        accepted = s['accepted_examples']
        rejected = s['rejected_examples']

        for j in range(n_examples_per_row):
            ax = axes[i, j]
            if j < 2:  # accepted
                if j < len(accepted):
                    _, pos, bare = accepted[j]
                    draw_topology_with_components(
                        ax, pos, bare,
                        f"{TOPO_LABELS_PL[topo]}, przyjęte #{j+1}")
                else:
                    ax.text(0.5, 0.5, 'brak',
                            ha='center', va='center', fontsize=12,
                            color='gray', transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])
            else:  # rejected
                rej_idx = j - 2
                if rej_idx < len(rejected):
                    _, pos, bare, n_comp = rejected[rej_idx]
                    draw_topology_with_components(
                        ax, pos, bare,
                        f"{TOPO_LABELS_PL[topo]}, odrzucone (najw. fragm.)")
                else:
                    ax.text(0.5, 0.5, 'brak rejects\n(100% accept rate)',
                            ha='center', va='center', fontsize=10,
                            color='gray', transform=ax.transAxes)
                    ax.set_xticks([]); ax.set_yticks([])

    plt.tight_layout()
    plt.savefig(out_path, dpi=160, bbox_inches='tight', facecolor='white')
    print(f'Wrote: {out_path}')


# --------------------------------------------------------------------------
# JSON export dla pracy
# --------------------------------------------------------------------------
def export_json(stats, out_path='figures/topology_connectivity_stats.json'):
    export = {}
    for topo in TOPO_ORDER:
        export[topo] = {}
        for density in DENSITIES:
            s = stats[topo][density]
            atts = s['attempts']
            export[topo][str(density)] = {
                'n_samples': N_SAMPLES,
                'max_retries': MAX_RETRIES,
                'mean_attempts': float(np.mean(atts)),
                'std_attempts': float(np.std(atts)),
                'max_attempts': int(np.max(atts)),
                'rejection_rate_pct': float(sum(a - 1 for a in atts) / sum(atts) * 100),
                'fallback_rate_pct': float(s['fallback_count'] / N_SAMPLES * 100),
                'first_try_success_rate_pct': float(
                    sum(1 for a in atts if a == 1) / len(atts) * 100),
            }

    with open(out_path, 'w') as f:
        json.dump(export, f, indent=2)
    print(f'Wrote: {out_path}')


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    print(f"Collecting stats ({N_SAMPLES} samples x {len(TOPO_ORDER)} topos x "
          f"{len(DENSITIES)} densities)...")
    stats = collect_stats()
    print()

    figure_rejection_rates(stats)
    figure_accepted_vs_rejected(stats, density=0.3)
    export_json(stats)
    print("\nDone.")


if __name__ == '__main__':
    main()
