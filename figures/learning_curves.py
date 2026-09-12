"""
Learning curves: success rate vs env_steps dla 5 modeli na jednym wykresie.
Pokazuje kiedy modele zbiegly, gdzie plateau, roznice w konwergencji.

Output: figures/learning_curves.png
Run: python figures/learning_curves.py
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing import event_accumulator


TRAIN_DIR = '/home/michal/Desktop/MSc-thesis/3-Quad-Swarm-RL/quad-swarm-rl/train_dir'

# (experiment_dir, display_label, color, linestyle)
MODELS = [
    ('paper_baseline_8drones_s0',           'Model bazowy (SDF, r=∞)',            '#1F77B4', '-'),
    ('perception_limited_r1.0_8drones_s0',  'Ograniczona percepcja r=1.0',        '#2CA02C', '-'),
    ('perception_limited_r0.2_8drones_s0',  'Ograniczona percepcja r=0.2',        '#D62728', '-'),
    ('multiranger_r4.0_8drones_s0',         'Multi-ranger (tylko siatka)',        '#FF7F0E', '-'),
    ('multiranger_r4.0_topomix_8drones_s0', 'Multi-ranger + mieszanka topologii', '#9467BD', '-'),
]

METRIC = 'metric/agent_success_rate'


def load_curve(experiment_dir, metric_tag):
    """Return (steps, values) numpy arrays from all event files in a run's .summary/."""
    log_dir = os.path.join(TRAIN_DIR, experiment_dir)
    if not os.path.isdir(log_dir):
        return None, None
    ef_pattern = os.path.join(log_dir, '.summary', '*', 'events.out.tfevents*')
    event_files = sorted(glob.glob(ef_pattern))
    if not event_files:
        return None, None
    steps_all, vals_all = [], []
    for ef in event_files:
        ea = event_accumulator.EventAccumulator(ef, size_guidance={'scalars': 0})
        ea.Reload()
        if metric_tag not in ea.Tags()['scalars']:
            continue
        events = ea.Scalars(metric_tag)
        for e in events:
            steps_all.append(e.step)
            vals_all.append(e.value)
    if not steps_all:
        return None, None
    # Sort by step (in case multiple event files w resume)
    idx = np.argsort(steps_all)
    steps = np.array(steps_all)[idx]
    vals = np.array(vals_all)[idx]
    # SF loguje "final tick" ktore ma anomalny reward (drop do ~0).
    # Trim ostatnie 3 events per unikniecia visual glitch.
    if len(vals) > 5:
        steps = steps[:-3]
        vals = vals[:-3]
    return steps, vals


def smooth(x, window=25):
    """Moving average — 'valid' mode zeby uniknac zerowego padding'u na krancach
    (co ciagnie smoothed values ku 0 na koncu tablicy)."""
    if len(x) < window:
        return x
    kernel = np.ones(window) / window
    return np.convolve(x, kernel, mode='valid')


def smoothed_steps(steps, window=25):
    """Skoresponduje ze smooth('valid') — trim n_pad z obu stron."""
    if len(steps) < window:
        return steps
    n_pad = (window - 1) // 2
    n_end = window - 1 - n_pad
    return steps[n_pad:len(steps) - n_end]


def main():
    fig, ax = plt.subplots(1, 1, figsize=(11, 6))

    for exp_dir, label, color, ls in MODELS:
        steps, vals = load_curve(exp_dir, METRIC)
        if steps is None:
            print(f'  [SKIP] {exp_dir}: no data')
            continue

        # Raw trace (light)
        ax.plot(steps / 1e6, vals, color=color, linestyle=ls,
                linewidth=0.5, alpha=0.25, zorder=2)
        # Smoothed (bold) — uzywa 'valid' convolve, wiec smoothed jest krotszy
        SMOOTH_WIN = 25
        vals_smooth = smooth(vals, window=SMOOTH_WIN)
        steps_smooth = smoothed_steps(steps, window=SMOOTH_WIN)
        ax.plot(steps_smooth / 1e6, vals_smooth, color=color, linestyle=ls,
                linewidth=2.0, label=label, zorder=3)

        # Print final value
        print(f'  {label:40s}: {len(vals):4d} points, final smoothed = {vals_smooth[-1]:.3f}')

    # Anneal collision zone
    ax.axvspan(0, 300, alpha=0.1, color='gray', label=None, zorder=1)
    ax.text(150, 0.02, 'fazowanie kary za kolizje\n(0 → 300 mln)', fontsize=9,
            ha='center', color='dimgray', style='italic')

    ax.set_xlabel('Kroki treningu [milionów]', fontsize=11)
    ax.set_ylabel('Skuteczność (dolot bez kolizji)', fontsize=11)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.0)
    ax.set_xlim(0, None)

    plt.tight_layout()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'learning_curves.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'\nWrote: {out}')


if __name__ == '__main__':
    main()
