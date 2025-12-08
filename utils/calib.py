import numpy as np
from sklearn.metrics import brier_score_loss, log_loss
import matplotlib.pyplot as plt

from out_file import write_to_file

from betacal import BetaCalibration
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
import numpy as np
def log_scling(preds):
    train_preds_mean = -4
    train_preds_std = 1
    scale_mean = 674
    scale_std = 88
    standard_predict = -(np.log(preds) - train_preds_mean) / train_preds_std
    scaled_predict = np.clip(standard_predict * scale_std + scale_mean, 0, 1000)
    return scaled_predict

def calib_rep(prob, y_oot, prob_valid, y_valid, prod_id):
    # --- 1. Platt Scaling ---
    prob_s = prob#log_scling(prob)
    prob_valid_s = prob_valid#log_scling(prob_valid)
    lr = LogisticRegression()
    lr.fit(prob_valid_s.reshape(-1, 1), y_valid)
    prob_platt = lr.predict_proba(prob_s.reshape(-1, 1))[:, 1]

    # --- 2. Isotonic Regression ---
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(prob_valid_s, y_valid)
    prob_iso = iso.transform(prob_s)

    # --- 3. Beta Calibration ---
    beta = BetaCalibration()
    beta.fit(prob_valid_s, y_valid)
    prob_beta = beta.predict(prob_s)

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

def plot_calibration(prob, y, method, bins=5):
    prob = np.asarray(prob)
    y = np.asarray(y)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))

    # --- 1. Диаграмма калибровки по квантилям ---
    quantile_edges = np.quantile(prob, np.linspace(0, 1, bins + 1))
    bin_means = []
    bin_centers = []

    for i in range(bins):
        mask = (prob >= quantile_edges[i]) & (prob < quantile_edges[i+1]) if i < bins-1 else (prob >= quantile_edges[i]) & (prob <= quantile_edges[i+1])
        if mask.sum() > 0:
            bin_means.append(y[mask].mean())
            bin_centers.append(prob[mask].mean())
        else:
            bin_means.append(np.nan)
            bin_centers.append((quantile_edges[i]+quantile_edges[i+1])/2)

    ax[0].plot(bin_centers, bin_means, "o-", color="blue", label="Квантильная калибровка")
    ax[0].plot([0, 1], [0, 1], "--", color="gray", label="Идеальная калибровка")

    ax[0].set_title("Диаграмма калибровки по квантилям " + method)
    ax[0].set_xlabel("Средняя предсказанная вероятность в квантиле")
    ax[0].set_ylabel("Фактическая частота")
    ax[0].legend()
    ax[0].grid(True)

    # --- 2. Гистограмма средних истинных меток по квантилям ---
    ax[1].bar(range(bins), bin_means, width=0.8, edgecolor='black')
    ax[1].set_xticks(range(bins))
    ax[1].set_xticklabels([f"{int(i*100/bins)}-{int((i+1)*100/bins)}%" for i in range(bins)])
    ax[1].set_title("Доля истинных меток по квантилям " + method)
    ax[1].set_xlabel("Квантиль")
    ax[1].set_ylabel("Доля единиц")
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()