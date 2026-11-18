#!/usr/bin/env python3
"""Walk-forward HMM demo: out-of-sample scoring and state alignment.

    python examples/python_hmm_walkforward.py

The model is refitted on a trailing window and scored on the next block.  Each
new fit is aligned to the first model's state labels before labels are compared.
"""

from __future__ import annotations

import argparse

import numpy as np

from rflux import regimelab


def simulate(seed: int = 13, days: int = 900) -> np.ndarray:
    rng = np.random.default_rng(seed)
    returns = np.empty(days)
    for t in range(days):
        if t < days * 0.35:
            drift, vol = 0.0008, 0.008
        elif t < days * 0.70:
            drift, vol = 0.0000, 0.017
        else:
            drift, vol = -0.0010, 0.028
        returns[t] = rng.normal(drift, vol)
    return returns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--states", type=int, default=3)
    parser.add_argument("--train-window", type=int, default=360)
    parser.add_argument("--refit-every", type=int, default=60)
    parser.add_argument("--n-init", type=int, default=2)
    args = parser.parse_args()

    returns = simulate(args.seed)
    reference_parameters = None
    scores: list[float] = []
    aligned_labels: list[np.ndarray] = []

    start = args.train_window
    while start + args.refit_every <= returns.size:
        train = returns[start - args.train_window:start]
        holdout = returns[start:start + args.refit_every]

        model = regimelab.fit_hmm(
            train.reshape(-1, 1),
            n_states=args.states,
            max_iter=200,
            n_init=args.n_init,
            seed=args.seed + start,
        )
        try:
            score = model.score(holdout)
            parameters = model.parameters
            if reference_parameters is None:
                reference_parameters = parameters
                order = np.arange(args.states, dtype=np.int64)
            else:
                order = regimelab.match_states(reference_parameters, parameters)
            aligned = regimelab.align_parameters(reference_parameters, parameters)
            labels = np.argmax(model.filtered(holdout), axis=1)
            labels = regimelab.align_labels(labels, order)
            aligned_labels.append(labels)
            scores.append(score)
            print(
                "block %4d | oos loglik %8.2f | alignment %s"
                % (start, score, order.tolist())
            )
            reference_parameters = aligned.parameters
        finally:
            model.close()
        start += args.refit_every

    if not scores:
        raise RuntimeError("not enough observations for the requested windows")
    print("mean out-of-sample log-likelihood:", round(float(np.mean(scores)), 3))
    print(
        "aligned state counts:",
        np.bincount(np.concatenate(aligned_labels), minlength=args.states).tolist(),
    )

    tail = returns[-args.train_window:]
    model = regimelab.fit_hmm(
        tail.reshape(-1, 1),
        n_states=args.states,
        n_init=args.n_init,
        seed=args.seed,
    )
    try:
        probabilities = model.filtered(tail)[-1]
        order = regimelab.match_states(reference_parameters, model.parameters)
        aligned_probabilities = regimelab.align_probabilities(
            probabilities.reshape(1, -1), order
        )[0]
        print(
            "latest aligned filtered probabilities:",
            np.round(aligned_probabilities, 4).tolist(),
        )
        print(
            "latest conditional mean:",
            round(model.conditional_mean(probabilities), 6),
        )
    finally:
        model.close()


if __name__ == "__main__":
    main()
