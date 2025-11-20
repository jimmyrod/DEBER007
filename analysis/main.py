import csv
import math
import os
import random
from typing import List, Tuple


# -------------------------------
# Data generation
# -------------------------------

def generate_dataset(n: int = 600, seed: int = 7) -> Tuple[List[List[float]], List[int], List[str]]:
    random.seed(seed)
    rows: List[List[float]] = []
    labels: List[int] = []
    feature_names = [
        "age",
        "annual_income",
        "debt_ratio",
        "prior_defaults",
        "employment_years",
        "education_level",
    ]

    for _ in range(n):
        age = random.randint(21, 70)
        income = random.randint(15000, 120000)
        debt_ratio = round(random.uniform(0.05, 0.95), 3)
        prior_defaults = random.randint(0, 5)
        employment_years = random.randint(0, 30)
        education_level = random.choice([0, 1])  # 1 = university/technical degree

        # latent score encodes ground-truth behavior
        score = (
            0.03 * (age - 40)
            + 0.00005 * (income - 45000)
            - 1.5 * debt_ratio
            - 0.8 * prior_defaults
            + 0.05 * employment_years
            + 0.4 * education_level
            + random.gauss(0, 0.5)
        )

        label = 1 if score > 0 else 0
        rows.append([
            age,
            income,
            debt_ratio,
            prior_defaults,
            employment_years,
            education_level,
        ])
        labels.append(label)

    return rows, labels, feature_names


def save_csv(rows: List[List[float]], labels: List[int], feature_names: List[str], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([*feature_names, "good_credit"])
        for values, label in zip(rows, labels):
            writer.writerow([*values, label])


# -------------------------------
# Basic math helpers
# -------------------------------


def vector_dot(a: List[float], b: List[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def sigmoid(z: float) -> float:
    # guard against overflow
    if z < -700:
        return 0.0
    return 1 / (1 + math.exp(-z))


def standardize_columns(data: List[List[float]]) -> Tuple[List[List[float]], List[float], List[float]]:
    if not data:
        return [], [], []
    num_features = len(data[0])
    means = [0.0] * num_features
    stds = [0.0] * num_features

    for row in data:
        for i, val in enumerate(row):
            means[i] += val
    means = [m / len(data) for m in means]

    for row in data:
        for i, val in enumerate(row):
            stds[i] += (val - means[i]) ** 2
    stds = [math.sqrt(s / len(data)) or 1.0 for s in stds]

    standardized = []
    for row in data:
        standardized.append([(val - means[i]) / stds[i] for i, val in enumerate(row)])

    return standardized, means, stds


# -------------------------------
# Logistic regression (from scratch)
# -------------------------------


def train_logistic_regression(
    X: List[List[float]],
    y: List[int],
    lr: float = 0.05,
    epochs: int = 4000,
    l2: float = 0.01,
) -> Tuple[List[float], float]:
    weights = [0.0 for _ in range(len(X[0]))]
    bias = 0.0

    for epoch in range(epochs):
        grad_w = [0.0 for _ in weights]
        grad_b = 0.0
        for xi, yi in zip(X, y):
            z = vector_dot(weights, xi) + bias
            pred = sigmoid(z)
            error = pred - yi
            for j in range(len(weights)):
                grad_w[j] += error * xi[j]
            grad_b += error

        # apply L2 regularization
        for j in range(len(weights)):
            grad_w[j] = grad_w[j] / len(X) + l2 * weights[j]
        grad_b = grad_b / len(X)

        for j in range(len(weights)):
            weights[j] -= lr * grad_w[j]
        bias -= lr * grad_b

        if (epoch + 1) % 1000 == 0:
            loss = 0.0
            for xi, yi in zip(X, y):
                pred = sigmoid(vector_dot(weights, xi) + bias)
                # logistic loss
                pred = min(max(pred, 1e-12), 1 - 1e-12)
                loss += -(yi * math.log(pred) + (1 - yi) * math.log(1 - pred))
            loss /= len(X)
            print(f"Epoch {epoch+1}: loss={loss:.4f}")

    return weights, bias


def predict_proba(weights: List[float], bias: float, X: List[List[float]]) -> List[float]:
    return [sigmoid(vector_dot(weights, xi) + bias) for xi in X]


def predict_label(probas: List[float], threshold: float = 0.5) -> List[int]:
    return [1 if p >= threshold else 0 for p in probas]


# -------------------------------
# Evaluation utilities
# -------------------------------


def classification_report(y_true: List[int], y_pred: List[int]) -> dict:
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)

    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def train_test_split(X: List[List[float]], y: List[int], test_size: float = 0.2, seed: int = 21):
    random.seed(seed)
    indices = list(range(len(X)))
    random.shuffle(indices)
    split = int(len(X) * (1 - test_size))
    train_idx, test_idx = indices[:split], indices[split:]
    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]
    return X_train, X_test, y_train, y_test


# -------------------------------
# Explainability utilities
# -------------------------------


def permutation_importance(
    weights: List[float],
    bias: float,
    X: List[List[float]],
    y: List[int],
    feature_names: List[str],
    repetitions: int = 10,
) -> List[Tuple[str, float]]:
    base_preds = predict_label(predict_proba(weights, bias, X))
    base_acc = classification_report(y, base_preds)["accuracy"]

    importances = []
    for idx, name in enumerate(feature_names):
        drops = []
        for r in range(repetitions):
            X_perturbed = [row[:] for row in X]
            shuffled = [row[idx] for row in X_perturbed]
            random.shuffle(shuffled)
            for row, new_val in zip(X_perturbed, shuffled):
                row[idx] = new_val
            preds = predict_label(predict_proba(weights, bias, X_perturbed))
            acc = classification_report(y, preds)["accuracy"]
            drops.append(base_acc - acc)
        mean_drop = sum(drops) / len(drops)
        importances.append((name, mean_drop))
    importances.sort(key=lambda x: x[1], reverse=True)
    return importances


def partial_dependence(
    weights: List[float],
    bias: float,
    X: List[List[float]],
    feature_idx: int,
    values: List[float],
) -> List[float]:
    averages = []
    for v in values:
        X_temp = [row[:] for row in X]
        for row in X_temp:
            row[feature_idx] = v
        probas = predict_proba(weights, bias, X_temp)
        averages.append(sum(probas) / len(probas))
    return averages


def contribution_explanation(weights: List[float], bias: float, row: List[float], feature_names: List[str]):
    contributions = []
    for w, x, name in zip(weights, row, feature_names):
        contributions.append((name, w * x))
    total = sum(val for _, val in contributions) + bias
    probability = sigmoid(total)
    contributions.sort(key=lambda x: abs(x[1]), reverse=True)
    return contributions, probability


# -------------------------------
# Simple SVG helpers (no external dependencies)
# -------------------------------


def _svg_header(width: int, height: int) -> List[str]:
    return [f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}'>"]


def _svg_footer() -> List[str]:
    return ["</svg>"]


def save_bar_chart(data: List[Tuple[str, float]], title: str, path: str) -> None:
    width, height = 800, 480
    margin = 80
    bar_width = 30
    spacing = 25
    max_val = max(val for _, val in data) if data else 1
    scale = (height - 2 * margin) / max_val if max_val else 1

    y_axis = height - margin
    x_start = margin

    lines = _svg_header(width, height)
    lines.append(f"<text x='{width/2}' y='40' text-anchor='middle' font-size='20'>{title}</text>")
    # axes
    lines.append(f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{y_axis}' stroke='black' stroke-width='2'/>")
    lines.append(f"<line x1='{margin}' y1='{y_axis}' x2='{width - margin/2}' y2='{y_axis}' stroke='black' stroke-width='2'/>")

    for idx, (label, val) in enumerate(data):
        bar_height = val * scale
        x = x_start + idx * (bar_width + spacing)
        y = y_axis - bar_height
        lines.append(
            f"<rect x='{x}' y='{y}' width='{bar_width}' height='{bar_height}' fill='#4a90e2'/>"
        )
        lines.append(
            f"<text x='{x + bar_width/2}' y='{y_axis + 18}' text-anchor='middle' font-size='12'>{label}</text>"
        )
        lines.append(
            f"<text x='{x + bar_width/2}' y='{y - 5}' text-anchor='middle' font-size='12'>{val:.3f}</text>"
        )

    lines.extend(_svg_footer())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def save_line_chart(x: List[float], y: List[float], title: str, x_label: str, y_label: str, path: str) -> None:
    width, height = 820, 500
    margin = 80
    plot_width = width - 2 * margin
    plot_height = height - 2 * margin

    min_x, max_x = min(x), max(x)
    min_y, max_y = min(y), max(y)
    x_range = max_x - min_x or 1
    y_range = max_y - min_y or 1

    def scale_x(val: float) -> float:
        return margin + (val - min_x) / x_range * plot_width

    def scale_y(val: float) -> float:
        return height - margin - (val - min_y) / y_range * plot_height

    lines = _svg_header(width, height)
    lines.append(f"<text x='{width/2}' y='40' text-anchor='middle' font-size='20'>{title}</text>")
    lines.append(
        f"<line x1='{margin}' y1='{margin}' x2='{margin}' y2='{height - margin}' stroke='black' stroke-width='2'/>"
    )
    lines.append(
        f"<line x1='{margin}' y1='{height - margin}' x2='{width - margin}' y2='{height - margin}' stroke='black' stroke-width='2'/>"
    )

    points = " ".join(f"{scale_x(xi)},{scale_y(yi)}" for xi, yi in zip(x, y))
    lines.append(f"<polyline fill='none' stroke='#e67e22' stroke-width='3' points='{points}'/>")

    for xi, yi in zip(x, y):
        sx, sy = scale_x(xi), scale_y(yi)
        lines.append(f"<circle cx='{sx}' cy='{sy}' r='4' fill='#e67e22'/>")

    lines.append(f"<text x='{width/2}' y='{height - 20}' text-anchor='middle' font-size='14'>{x_label}</text>")
    lines.append(
        f"<text x='20' y='{height/2}' transform='rotate(-90 20,{height/2})' text-anchor='middle' font-size='14'>{y_label}</text>"
    )

    lines.extend(_svg_footer())
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


# -------------------------------
# Main execution
# -------------------------------


def main():
    rows, labels, feature_names = generate_dataset()
    csv_path = os.path.join("data", "synthetic_credit.csv")
    save_csv(rows, labels, feature_names, csv_path)
    print(f"Dataset saved to {csv_path} with {len(rows)} rows")

    standardized, means, stds = standardize_columns(rows)
    X_train, X_test, y_train, y_test = train_test_split(standardized, labels)

    weights, bias = train_logistic_regression(X_train, y_train)

    # evaluation
    test_probas = predict_proba(weights, bias, X_test)
    test_preds = predict_label(test_probas)
    metrics = classification_report(y_test, test_preds)
    print("Test metrics:", metrics)

    # explainability: coefficient magnitudes
    coef_effects = list(zip(feature_names, weights))
    coef_effects.sort(key=lambda x: abs(x[1]), reverse=True)
    coef_chart_path = os.path.join("outputs", "coef_importance.svg")
    save_bar_chart([(n, abs(w)) for n, w in coef_effects], "Influencia por coeficiente", coef_chart_path)
    print(f"Saved coefficient chart to {coef_chart_path}")

    # permutation importance
    perm = permutation_importance(weights, bias, X_test, y_test, feature_names)
    perm_chart_path = os.path.join("outputs", "permutation_importance.svg")
    save_bar_chart(perm, "Permutation Feature Importance", perm_chart_path)
    print(f"Saved permutation chart to {perm_chart_path}")

    # PDP for income and debt ratio
    income_index = feature_names.index("annual_income")
    debt_index = feature_names.index("debt_ratio")
    income_values = [x for x in range(-2, 3)]  # standardized scale
    debt_values = [round(-1.5 + i * 0.5, 2) for i in range(7)]

    income_pdp = partial_dependence(weights, bias, X_test, income_index, income_values)
    debt_pdp = partial_dependence(weights, bias, X_test, debt_index, debt_values)

    income_chart_path = os.path.join("outputs", "pdp_income.svg")
    save_line_chart(
        income_values,
        income_pdp,
        "PDP: ingreso anual (escalado)",
        "Valor estandarizado",
        "Probabilidad de buen crédito",
        income_chart_path,
    )

    debt_chart_path = os.path.join("outputs", "pdp_debt_ratio.svg")
    save_line_chart(
        debt_values,
        debt_pdp,
        "PDP: razón de deuda (escalado)",
        "Valor estandarizado",
        "Probabilidad de buen crédito",
        debt_chart_path,
    )

    print(f"Saved PDP charts to {income_chart_path} and {debt_chart_path}")

    # individual explanations for two samples
    if X_test:
        sample_one = X_test[0]
        contribs, prob = contribution_explanation(weights, bias, sample_one, feature_names)
        print("Ejemplo 1 - probabilidad", round(prob, 3))
        for name, val in contribs:
            print(f"  {name}: {val:.3f}")

        if len(X_test) > 1:
            sample_two = X_test[1]
            contribs2, prob2 = contribution_explanation(weights, bias, sample_two, feature_names)
            print("Ejemplo 2 - probabilidad", round(prob2, 3))
            for name, val in contribs2:
                print(f"  {name}: {val:.3f}")


if __name__ == "__main__":
    main()
