
import numpy as np
import pandas as pd
from scipy.optimize import linprog


def ahp(matrix):
    """AHP weights and consistency ratio."""
    matrix = np.asarray(matrix, dtype=float)
    n = len(matrix)

    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    index = np.argmax(eigenvalues.real)

    weights = np.abs(eigenvectors[:, index].real)
    weights /= weights.sum()

    lambda_max = eigenvalues[index].real
    ci = (lambda_max - n) / (n - 1)

    ri = {3: 0.58, 4: 0.90, 5: 1.12}
    cr = max(0, ci / ri.get(n, 1))

    return weights, cr


def bwm(best, worst, best_to_others, others_to_worst):
    """Linear Best-Worst Method."""
    bo = np.asarray(best_to_others, dtype=float)
    ow = np.asarray(others_to_worst, dtype=float)
    n = len(bo)

    constraints = []

    for j in range(n):
        row = np.zeros(n + 1)
        row[best] += 1
        row[j] -= bo[j]
        row[-1] = -1
        constraints.append(row)

        row = np.zeros(n + 1)
        row[best] -= 1
        row[j] += bo[j]
        row[-1] = -1
        constraints.append(row)

        row = np.zeros(n + 1)
        row[j] += 1
        row[worst] -= ow[j]
        row[-1] = -1
        constraints.append(row)

        row = np.zeros(n + 1)
        row[j] -= 1
        row[worst] += ow[j]
        row[-1] = -1
        constraints.append(row)

    objective = np.zeros(n + 1)
    objective[-1] = 1

    equality = np.zeros((1, n + 1))
    equality[0, :n] = 1

    result = linprog(
        objective,
        A_ub=np.array(constraints),
        b_ub=np.zeros(len(constraints)),
        A_eq=equality,
        b_eq=[1],
        bounds=[(0, None)] * (n + 1),
        method="highs"
    )

    if not result.success:
        raise ValueError(result.message)

    return result.x[:n], result.x[-1]


def entropy(matrix):
    """Objective weights using Shannon entropy."""
    x = np.asarray(matrix, dtype=float)
    p = x / x.sum(axis=0)

    terms = np.zeros_like(p)
    positive = p > 0
    terms[positive] = p[positive] * np.log(p[positive])

    e = -terms.sum(axis=0) / np.log(len(x))
    information = np.maximum(0, 1 - e)

    if information.sum() < 1e-12:
        raise ValueError("Entropy cannot distinguish these criteria.")

    return information / information.sum(), e


def critic(matrix):
    """CRITIC using absolute correlations."""
    x = np.asarray(matrix, dtype=float)
    spread = np.ptp(x, axis=0)

    normalized = np.divide(
        x - x.min(axis=0),
        spread,
        out=np.zeros_like(x),
        where=spread > 0
    )

    std = normalized.std(axis=0, ddof=1)
    n = x.shape[1]
    correlation = np.eye(n)

    for j in range(n):
        for k in range(j + 1, n):
            if std[j] > 1e-12 and std[k] > 1e-12:
                rho = np.corrcoef(
                    normalized[:, j],
                    normalized[:, k]
                )[0, 1]
                correlation[j, k] = rho
                correlation[k, j] = rho

    information = std * (
        1 - np.abs(correlation)
    ).sum(axis=1)

    if information.sum() < 1e-12:
        raise ValueError("CRITIC cannot distinguish these criteria.")

    return information / information.sum(), correlation


def ranking(matrix, weights, method, lam=0.5):
    """
    All criteria are benefit-oriented for risk priority:
    a higher input score means a higher intervention priority.
    """
    x = np.asarray(matrix, dtype=float)
    w = np.asarray(weights, dtype=float)

    if method in ("WSM", "WPM", "WASPAS"):
        normalized = x / x.max(axis=0)
        wsm = normalized @ w
        wpm = np.prod(normalized ** w, axis=1)

        if method == "WSM":
            return wsm, False, {}

        if method == "WPM":
            return wpm, False, {}

        return lam * wsm + (1 - lam) * wpm, False, {}

    if method == "TOPSIS":
        normalized = x / np.sqrt((x ** 2).sum(axis=0))
        weighted = normalized * w

        ideal = weighted.max(axis=0)
        anti_ideal = weighted.min(axis=0)

        d_plus = np.linalg.norm(weighted - ideal, axis=1)
        d_minus = np.linalg.norm(weighted - anti_ideal, axis=1)

        denominator = d_plus + d_minus
        score = np.divide(
            d_minus,
            denominator,
            out=np.full(len(x), 0.5),
            where=denominator > 0
        )

        return score, False, {
            "Ideal distance": d_plus,
            "Anti-ideal distance": d_minus
        }

    raise ValueError(f"Unknown ranking method: {method}")


def ranking_table(matrix, weights, method, risks, lam=0.5):
    scores, ascending, details = ranking(
        matrix, weights, method, lam
    )

    result = pd.DataFrame({
        "Risk": risks,
        "Score": scores,
        **details
    })

    result = result.sort_values(
        "Score", ascending=ascending, kind="stable"
    ).reset_index(drop=True)

    result.insert(0, "Rank", np.arange(1, len(result) + 1))

    return result
