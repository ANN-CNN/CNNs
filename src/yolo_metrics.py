"""
Módulo de Métricas e Curvas de Avaliação idênticas ao YOLOv5.
Permite calcular e plotar:
- Matriz de Confusão (com classe Background)
- Curva Precision-Confiança (P_curve)
- Curva Recall-Confiança (R_curve)
- Curva F1-Confiança (F1_curve)
- Curva Precision-Recall com mAP@0.5 (PR_curve)
"""

import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch

# Compatibilidade NumPy 1.x e 2.0+ (trapezoid substituiu trapz)
trapezoid = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def box_iou(box1, box2):
    """
    Calcula a interseção sobre união (IoU) entre duas listas de caixas no formato [x1, y1, x2, y2].
    Args:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): matriz de IoU
    """
    if box1.shape[0] == 0 or box2.shape[0] == 0:
        return torch.zeros((box1.shape[0], box2.shape[0]), dtype=torch.float32, device=box1.device)

    # Interseção
    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N, M, 2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]

    union = area1[:, None] + area2 - inter
    return inter / (union + 1e-16)


def process_batch(detections, labels, iouv):
    """
    Determina quais detecções são True Positives para cada limiar de IoU (padrão YOLOv5).
    Args:
        detections (Tensor[N, 6]): [x1, y1, x2, y2, conf, class]
        labels (Tensor[M, 5]): [class, x1, y1, x2, y2]
        iouv (Tensor[10]): limiares de IoU (ex: linspace(0.5, 0.95, 10))
    Returns:
        correct (Tensor[N, 10]): booleano indicando TP para cada limiar
    """
    correct = np.zeros((detections.shape[0], iouv.shape[0]), dtype=bool)
    if detections.shape[0] == 0 or labels.shape[0] == 0:
        return torch.tensor(correct, dtype=torch.bool, device=iouv.device)

    iou = box_iou(labels[:, 1:], detections[:, :4])
    correct_class = labels[:, 0:1] == detections[:, 5]

    for i in range(len(iouv)):
        matches_idx = torch.where((iou >= iouv[i]) & correct_class)
        if matches_idx[0].shape[0]:
            matches = torch.cat(
                (torch.stack(matches_idx, 1), iou[matches_idx[0], matches_idx[1]][:, None]),
                1,
            ).cpu().numpy()
            if matches_idx[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True

    return torch.tensor(correct, dtype=torch.bool, device=iouv.device)


def smooth(y, f=0.05):
    """Filtro de suavização estilo caixa (YOLOv5)."""
    nf = round(len(y) * f * 2) // 2 + 1  # ímpar
    p = np.ones(nf // 2)
    yp = np.concatenate((p * y[0], y, p * y[-1]), 0)
    return np.convolve(yp, np.ones(nf) / nf, mode="valid")


def compute_ap(recall, precision):
    """
    Calcula Average Precision (AP) pelo método de 101 pontos do COCO.
    """
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))

    # Envelope de precisão (máximo acumulado reverso)
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))

    x = np.linspace(0, 1, 101)  # interpolação de 101 pontos
    ap = trapezoid(np.interp(x, mrec, mpre), x)
    return ap, mpre, mrec


def plot_pr_curve(px, py, ap, save_dir="pr_curve.png", names=None):
    """Plota a curva Precision-Recall padrão YOLOv5."""
    names = names if names is not None else {}
    save_path = Path(save_dir)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    py_stack = np.stack(py, axis=1) if len(py) > 0 else np.zeros((len(px), len(names)))

    if 0 < len(names) < 21:
        for i, y in enumerate(py_stack.T):
            class_name = names.get(i, f"class_{i}")
            ax.plot(px, y, linewidth=1, label=f"{class_name} {ap[i, 0]:.3f}")
    else:
        ax.plot(px, py_stack, linewidth=1, color="gray")

    mean_ap = ap[:, 0].mean() if len(ap) > 0 else 0.0
    ax.plot(px, py_stack.mean(1), linewidth=3, color="blue", label=f"all classes {mean_ap:.3f} mAP@0.5")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_mc_curve(px, py, save_dir="mc_curve.png", names=None, xlabel="Confidence", ylabel="Metric"):
    """Plota curvas de métrica vs confiança (P, R, F1) padrão YOLOv5."""
    names = names if names is not None else {}
    save_path = Path(save_dir)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)

    if 0 < len(names) < 21:
        for i, y in enumerate(py):
            class_name = names.get(i, f"class_{i}")
            ax.plot(px, y, linewidth=1, label=f"{class_name}")
    else:
        ax.plot(px, py.T, linewidth=1, color="gray")

    y_mean = smooth(py.mean(0), 0.05) if len(py) > 0 else np.zeros_like(px)
    best_idx = y_mean.argmax() if len(y_mean) > 0 else 0
    best_val = y_mean[best_idx] if len(y_mean) > 0 else 0.0
    best_conf = px[best_idx] if len(px) > 0 else 0.0

    ax.plot(px, y_mean, linewidth=3, color="blue", label=f"all classes {best_val:.2f} at {best_conf:.3f}")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(bbox_to_anchor=(1.04, 1), loc="upper left", frameon=True)
    ax.set_title(f"{ylabel}-Confidence Curve", fontsize=14, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.6)
    fig.savefig(save_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def ap_per_class(tp, conf, pred_cls, target_cls, plot=False, save_dir=".", names=(), eps=1e-16, prefix=""):
    """
    Computa a precisão média (AP) e gera as curvas P, R, F1 e PR idênticas ao YOLOv5.
    Args:
        tp (ndarray): [N, 10]
        conf (ndarray): [N]
        pred_cls (ndarray): [N]
        target_cls (ndarray): [M]
        plot (bool): se True, salva as 4 curvas
        save_dir (str/Path): diretório onde salvar os gráficos
        names (dict): dicionário {class_id: class_name}
    """
    # Ordenar por confiança decrescente
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    unique_classes, nt = np.unique(target_cls, return_counts=True)
    nc = len(names) if len(names) > 0 else len(unique_classes)

    px, py = np.linspace(0, 1, 1000), []
    ap = np.zeros((nc, tp.shape[1] if tp.ndim > 1 else 1))
    p = np.zeros((nc, 1000))
    r = np.zeros((nc, 1000))

    for ci, c in enumerate(range(nc)):
        idx_pred = pred_cls == c
        n_l = nt[unique_classes == c][0] if c in unique_classes else 0
        n_p = idx_pred.sum()

        if n_p == 0 or n_l == 0:
            continue

        fpc = (1 - tp[idx_pred]).cumsum(0)
        tpc = tp[idx_pred].cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)
        r[ci] = np.interp(-px, -conf[idx_pred], recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)
        p[ci] = np.interp(-px, -conf[idx_pred], precision[:, 0], left=1)

        # AP para cada threshold de IoU
        for j in range(tp.shape[1]):
            ap[ci, j], mpre, mrec = compute_ap(recall[:, j], precision[:, j])
            if plot and j == 0:
                py.append(np.interp(px, mrec, mpre))

    f1 = 2 * p * r / (p + r + eps)

    if plot:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        plot_pr_curve(px, py, ap, save_path / f"{prefix}PR_curve.png", names)
        plot_mc_curve(px, f1, save_path / f"{prefix}F1_curve.png", names, ylabel="F1")
        plot_mc_curve(px, p, save_path / f"{prefix}P_curve.png", names, ylabel="Precision")
        plot_mc_curve(px, r, save_path / f"{prefix}R_curve.png", names, ylabel="Recall")

    i = smooth(f1.mean(0), 0.1).argmax() if f1.shape[1] > 0 else 0
    p_opt, r_opt, f1_opt = p[:, i], r[:, i], f1[:, i]
    tp_count = (r_opt * (np.array([nt[unique_classes == c][0] if c in unique_classes else 0 for c in range(nc)]))).round()
    fp_count = (tp_count / (p_opt + eps) - tp_count).round()

    return tp_count, fp_count, p_opt, r_opt, f1_opt, ap, unique_classes.astype(int)


class ConfusionMatrix:
    """
    Matriz de Confusão para detecção de objetos com suporte a background (YOLOv5 style).
    Dimensão: (nc + 1) x (nc + 1).
    """

    def __init__(self, nc, conf=0.25, iou_thres=0.45):
        self.nc = nc
        self.conf = conf
        self.iou_thres = iou_thres
        self.matrix = np.zeros((nc + 1, nc + 1))

    def process_batch(self, detections, labels):
        """
        Atualiza a matriz de confusão.
        Args:
            detections (Tensor[N, 6]): [x1, y1, x2, y2, conf, class]
            labels (Tensor[M, 5]): [class, x1, y1, x2, y2]
        """
        if detections is None or detections.shape[0] == 0:
            if labels is not None and labels.shape[0] > 0:
                gt_classes = labels[:, 0].int()
                for gc in gt_classes:
                    self.matrix[self.nc, gc] += 1  # background FN
            return

        if labels is None or labels.shape[0] == 0:
            detections = detections[detections[:, 4] > self.conf]
            detection_classes = detections[:, 5].int()
            for dc in detection_classes:
                self.matrix[dc, self.nc] += 1  # predicted background FP
            return

        detections = detections[detections[:, 4] > self.conf]
        if detections.shape[0] == 0:
            gt_classes = labels[:, 0].int()
            for gc in gt_classes:
                self.matrix[self.nc, gc] += 1
            return

        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        matches_idx = torch.where(iou > self.iou_thres)
        if matches_idx[0].shape[0]:
            matches = torch.cat(
                (torch.stack(matches_idx, 1), iou[matches_idx[0], matches_idx[1]][:, None]),
                1,
            ).cpu().numpy()
            if matches_idx[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(int)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # True Positive
            else:
                self.matrix[self.nc, gc] += 1  # False Negative (virou background)

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # False Positive (era background)

    def plot(self, normalize=True, save_dir="", names=()):
        """Plota a matriz usando seaborn com estilo do YOLOv5."""
        import seaborn as sn

        save_path = Path(save_dir) / "confusion_matrix.png"
        save_path.parent.mkdir(parents=True, exist_ok=True)

        array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1e-9) if normalize else 1)
        array[array < 0.005] = np.nan

        fig, ax = plt.subplots(1, 1, figsize=(10, 8), tight_layout=True)
        nc, nn = self.nc, len(names)

        labels = (0 < nn < 99) and (nn == nc)
        class_list = [names[i] for i in range(nc)] if isinstance(names, dict) else list(names)
        ticklabels = ([*class_list, "background"]) if labels else "auto"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sn.heatmap(
                array,
                ax=ax,
                annot=nc < 30,
                annot_kws={"size": 10},
                cmap="Blues",
                fmt=".2f",
                square=True,
                vmin=0.0,
                vmax=1.0,
                xticklabels=ticklabels,
                yticklabels=ticklabels,
            ).set_facecolor((1, 1, 1))

        ax.set_xlabel("True (Gabarito)", fontsize=12, fontweight="bold")
        ax.set_ylabel("Predicted (Predição)", fontsize=12, fontweight="bold")
        ax.set_title("Confusion Matrix (Normalizada)", fontsize=14, fontweight="bold")
        fig.savefig(save_path, dpi=250, bbox_inches="tight")
        plt.close(fig)
