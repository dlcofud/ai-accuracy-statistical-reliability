"""Reproduce the Chapter 8 extra investigation on AI accuracy uncertainty.

Synthetic examples only: this script does not evaluate a trained AI model.
"""
from pathlib import Path
import argparse
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

CASES = [('A (18/20)', 18, 20), ('B (900/1000)', 900, 1000)]


def analyze_case(k: int, n: int, p0: float = 0.8, alpha: float = 0.05) -> dict:
    if not (isinstance(k, int) and isinstance(n, int) and 0 <= k <= n and n > 0):
        raise ValueError('Require integers n > 0 and 0 <= k <= n')
    if not (0 < p0 < 1 and 0 < alpha < 1):
        raise ValueError('Require 0 < p0 < 1 and 0 < alpha < 1')
    phat = k / n
    # IMPORTANT: construct a default (two-sided) binomial test for the two-sided Wilson interval.
    ci = binomtest(k, n).proportion_ci(confidence_level=1-alpha, method='wilson')
    # Separate, one-sided hypothesis test of H0: p = p0 vs H1: p > p0.
    pvalue = binomtest(k, n, p=p0, alternative='greater').pvalue
    return {'k': k, 'n': n, 'accuracy': phat,
            'estimated_se': math.sqrt(phat*(1-phat)/n),
            'ci_low': ci.low, 'ci_high': ci.high,
            'pvalue': pvalue, 'reject': pvalue < alpha}


def draw_comparison(results: list, output: Path) -> None:
    # No manually chosen colors; use Matplotlib default styles.
    y = np.arange(len(results))
    means = np.array([x['accuracy'] for x in results])
    lower = means - np.array([x['ci_low'] for x in results])
    upper = np.array([x['ci_high'] for x in results]) - means
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.errorbar(means, y, xerr=[lower,upper], fmt='o', capsize=6)
    ax.axvline(0.8, linestyle='--', linewidth=1.4, label='Hypothesized accuracy: 80%')
    ax.set_yticks(y, [f"n={x['n']:,} ({x['k']:,} correct)" for x in results])
    ax.invert_yaxis()
    ax.set_xlim(0.65, 1.01)
    ax.set_xlabel('Accuracy and 95% two-sided Wilson confidence interval')
    ax.set_title('Same observed accuracy, different uncertainty')
    ax.legend(loc='lower right')
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def monte_carlo(n: int, p: float, repeats: int, seed: int):
    if not (0 <= p <= 1 and repeats > 0):
        raise ValueError('Require 0 <= p <= 1 and repeats > 0')
    rng = np.random.default_rng(seed)
    # Draw the number of correct answers in each independent test set.
    rates = rng.binomial(n=n, p=p, size=repeats) / n
    return {'mean': float(rates.mean()), 'sd': float(rates.std(ddof=1)),
            'theoretical_sd': math.sqrt(p*(1-p)/n)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs'))
    parser.add_argument('--seed', type=int, default=29,
                        help='Fixed seed for reproducible simulations (default: last four student-ID digits: 0029)')
    parser.add_argument('--repeats', type=int, default=10000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for label,k,n in CASES:
        r = analyze_case(k,n)
        results.append(r)
        print(f"{label}: accuracy={r['accuracy']:.4%}, SE={r['estimated_se']:.6f}, "
              f"Wilson 95% CI=[{r['ci_low']:.4%}, {r['ci_high']:.4%}], "
              f"one-sided exact p={r['pvalue']:.9g}, reject H0={r['reject']}")
    draw_comparison(results, args.output/'accuracy_intervals.png')
    print('\nMonte Carlo: synthetic independent cases with p=0.9')
    for _,_,n in CASES:
        m = monte_carlo(n, .9, args.repeats, args.seed+n)
        print(f"n={n}: average={m['mean']:.5f}, empirical SD={m['sd']:.5f}, "
              f"theoretical SD={m['theoretical_sd']:.5f}")
    print(f"\nSaved graph: {args.output/'accuracy_intervals.png'}")

if __name__ == '__main__':
    main()
