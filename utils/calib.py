import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

from out_file import write_to_file

from betacal import BetaCalibration
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np


def calib_rep(prob, y_oot, prob_valid, y_valid, prod_id):
    # --- 1. Platt Scaling ---
    lr = LogisticRegression()
    lr.fit(prob_valid.reshape(-1, 1), y_valid)
    prob_platt = lr.predict_proba(prob.reshape(-1, 1))[:, 1]

    # --- 2. Isotonic Regression ---
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob_valid, y_valid)
    prob_iso = iso.transform(prob)

    # --- 3. Beta Calibration ---
    beta = BetaCalibration()
    beta.fit(prob_valid, y_valid)
    prob_beta = beta.predict(prob)

    score(prob, y_oot, prod_id, "raw_score")
    score(prob_platt, y_oot, prod_id, "platt")
    score(prob_iso, y_oot, prod_id, "isotonic")
    score(prob_beta, y_oot, prod_id, "beta")
    return prob_platt, prob_iso, prob_beta


def score(prob, y_oot, prod_id, method):
    print("="*50)
    write_to_file(f"calibrate_info_{prod_id}", "="*50)
    print(method)
    write_to_file(f"calibrate_info_{prod_id}", method)
    print(f"Brier score - {brier_score_loss(y_oot, prob)}")
    write_to_file(f"calibrate_info_{prod_id}", f"Brier score - {brier_score_loss(y_oot, prob)}")
    print(f"Log los - {log_loss(y_oot, prob)}")
    write_to_file(f"calibrate_info_{prod_id}", f"Log los - {log_loss(y_oot, prob)}")
    plot_calibration(prob, y_oot, method)

def plot_calibration(prob, y, method, bins=10):
    prob = np.asarray(prob)
    y = np.asarray(y)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # --- 1. Диаграмма калибровки ---
    frac_pos, mean_pred = calibration_curve(y, prob, n_bins=bins, strategy='uniform')
    ax[0].plot(mean_pred, frac_pos, "o-", label="Модель")
    ax[0].plot([0, 1], [0, 1], "--", color="gray", label="Медиана")

    # --- 1a. Среднее истинных меток в каждом бине ---
    bin_edges = np.linspace(0, 1, bins + 1)
    bin_means = []
    for i in range(bins):
        mask = (prob >= bin_edges[i]) & (prob < bin_edges[i+1])
        if mask.sum() > 0:
            bin_means.append(y[mask].mean())
        else:
            bin_means.append(np.nan)
    ax[0].plot((bin_edges[:-1] + bin_edges[1:]) / 2, bin_means, "x-", color="black", label="Среднее истинное значение")

    ax[0].set_title("Диаграмма калибровки " + method)
    ax[0].set_xlabel("Предсказанная вероятность")
    ax[0].set_ylabel("Фактическая частота")
    ax[0].legend()
    ax[0].grid(True)

    # --- 2. Гистограмма вероятностей (шаг 0.1) ---
    bins_hist = np.arange(0, 1.1, 0.1)
    ax[1].hist(prob, bins=bins_hist, edgecolor="black")
    ax[1].set_title("Распределение предсказанных вероятностей " + method)
    ax[1].set_xlabel("Вероятность (шаг 0.1)")
    ax[1].set_ylabel("Количество объектов")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()
