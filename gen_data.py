#!/usr/bin/env python3

import argparse
import numpy as np
import pandas as pd


def minmax(x):
    x = np.asarray(x, dtype=float)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


def generate_dataset(
    n=1000,
    lambda_1=0.7,
    boost_strength=1.2,
    lambda_2=0.5,
    diversity_penalty_strength=1.0,
    n_categories=8,
    seed=42,
):
    rng = np.random.default_rng(seed)

    # -----------------------------
    # Visible features
    # -----------------------------
    relevance_score = rng.beta(5, 2, size=n)
    popularity = rng.lognormal(mean=1.8, sigma=0.8, size=n)
    freshness = rng.beta(2, 5, size=n)
    predicted_ctr = np.clip(
        0.45 * relevance_score + 0.25 * minmax(popularity) + 0.30 * freshness,
        0, 1
    )
    quality_score = np.clip(
        0.50 * relevance_score + 0.30 * predicted_ctr + 0.20 * rng.random(n),
        0, 1
    )
    price = rng.lognormal(mean=3.2, sigma=0.6, size=n)

    # category-like numeric tag
    diversity_tag = rng.integers(0, n_categories, size=n)

    # -----------------------------
    # Visible score WX
    # -----------------------------
    popularity_norm = minmax(popularity)
    price_norm = minmax(price)

    X = np.column_stack([
        relevance_score,
        popularity_norm,
        freshness,
        predicted_ctr,
        quality_score,
        1.0 - price_norm,
        diversity_tag / max(1, n_categories - 1),
    ])

    W = np.array([0.28, 0.14, 0.12, 0.18, 0.16, 0.08, 0.04], dtype=float)
    W = W / W.sum()

    visible_score = X @ W
    visible_score = minmax(visible_score)

    # -----------------------------
    # Hidden boost signal
    # promotes sponsored/new items
    # make it continuous, not only binary
    # -----------------------------
    sponsored = rng.binomial(1, 0.15, size=n)
    sponsor_strength = sponsored * rng.uniform(0.6, 1.0, size=n)

    boost_signal = (
        0.55 * freshness
        + 0.25 * predicted_ctr
        + 0.20 * sponsor_strength
    )
    boost_signal = minmax(boost_signal)

    # -----------------------------
    # Hidden diversity penalty signal
    # penalize repetitive categories
    # make some categories much more common
    # -----------------------------
    category_popularity = rng.dirichlet(alpha=np.ones(n_categories) * 0.5)
    category_weight = category_popularity[diversity_tag]
    category_weight = minmax(category_weight)

    # more popular categories get more penalty
    # and highly popular items in such categories get penalized more
    diversity_penalty_signal = category_weight * (0.6 + 0.4 * popularity_norm)
    diversity_penalty_signal = minmax(diversity_penalty_signal)

    # -----------------------------
    # Final hidden score
    # all terms are on comparable scale
    # -----------------------------
    hidden_score = (
        visible_score
        + lambda_1 * boost_strength * boost_signal
        - lambda_2 * diversity_penalty_strength * diversity_penalty_signal
    )

    order = np.argsort(-hidden_score)
    rank = np.empty(n, dtype=int)
    rank[order] = np.arange(1, n + 1)

    df = pd.DataFrame({
        "item_id": np.arange(n),
        "relevance_score": relevance_score,
        "popularity": popularity,
        "freshness": freshness,
        "predicted_ctr": predicted_ctr,
        "quality_score": quality_score,
        "price": price,
        "diversity_tag": diversity_tag.astype(float),
        "visible_score": visible_score,
        "boost_signal": boost_signal,
        "diversity_penalty_signal": diversity_penalty_signal,
        "hidden_score": hidden_score,
        "rank": rank,
        "is_sponsored": sponsored,
    })

    return df.sort_values("rank").reset_index(drop=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=1000000)
    parser.add_argument("--lambda_1", type=float, default=1.7)
    parser.add_argument("--boost", type=float, default=1.2)
    parser.add_argument("--lambda_2", type=float, default=0.5)
    parser.add_argument("--diversity_penalty", type=float, default=1.0)
    parser.add_argument("--n_categories", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="synthetic_data.csv")
    return parser.parse_args()


def main():
    args = parse_args()

    df = generate_dataset(
        n=args.n,
        lambda_1=args.lambda_1,
        boost_strength=args.boost,
        lambda_2=args.lambda_2,
        diversity_penalty_strength=args.diversity_penalty,
        n_categories=args.n_categories,
        seed=args.seed,
    )

    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
#python gen_data.py --lambda_1 0.1 --lambda_2 0.1 --output syn.csv
