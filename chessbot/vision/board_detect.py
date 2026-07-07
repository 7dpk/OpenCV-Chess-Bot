import cv2
import numpy as np

Region = tuple[int, int, int, int]


def _get_all_sequences(seq, min_seq_len=7, err_px=5):
    """All subsequences with common spacing (within err_px) of length >= min_seq_len."""
    if len(seq) < min_seq_len:
        return []
    seqs = []
    for i in range(len(seq) - 1):
        for j in range(i + 1, len(seq)):
            duplicate = False
            for prev in seqs:
                for k in range(len(prev) - 1):
                    if seq[i] == prev[k] and seq[j] == prev[k + 1]:
                        duplicate = True
            if duplicate:
                continue
            d = seq[j] - seq[i]
            if d < err_px:
                continue
            s = [seq[i], seq[j]]
            n = s[-1] + d
            while np.abs(seq - n).min() < err_px:
                n = seq[np.abs(seq - n).argmin()]
                s.append(n)
                n = s[-1] + d
            if len(s) >= min_seq_len:
                seqs.append(np.array(s))
    return seqs


def _nonmax_suppress_1d(arr, winsize=5):
    out = arr.copy()
    for i in range(out.size):
        left = arr[max(0, i - winsize):i] if i > 0 else np.zeros(1)
        right = arr[i + 1:min(arr.size - 1, i + winsize)] if i < out.size - 2 else np.zeros(1)
        if left.size and arr[i] < left.max():
            out[i] = 0
        elif right.size and arr[i] <= right.max():
            out[i] = 0
    return out


def _trim_sequence(seq, vals):
    while len(seq) > 9:
        if vals[0] > vals[-1]:
            seq, vals = seq[:-1], vals[:-1]
        else:
            seq, vals = seq[1:], vals[1:]
    return seq, vals


def _crop_padded(img, x0, y0, x1, y1):
    h, w = img.shape
    out = np.zeros((y1 - y0, x1 - x0), img.dtype)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def detect_chessboard_corners(gray: np.ndarray, noise_threshold: float = 8000) -> np.ndarray | None:
    grad_rows, grad_cols = np.gradient(gray.astype(float))
    r_pos, r_neg = np.clip(grad_rows, 0, None), np.clip(-grad_rows, 0, None)
    c_pos, c_neg = np.clip(grad_cols, 0, None), np.clip(-grad_cols, 0, None)

    hough_rows = r_pos.sum(axis=1) * r_neg.sum(axis=1)
    hough_cols = c_pos.sum(axis=0) * c_neg.sum(axis=0)

    if min(hough_rows.std() / hough_rows.size, hough_cols.std() / hough_cols.size) < noise_threshold:
        return None

    hough_rows = _nonmax_suppress_1d(hough_rows) / hough_rows.max()
    hough_cols = _nonmax_suppress_1d(hough_cols) / hough_cols.max()
    hough_rows[hough_rows < 0.2] = 0
    hough_cols[hough_cols < 0.2] = 0

    lines_y = np.where(hough_rows)[0]
    lines_x = np.where(hough_cols)[0]
    vals_y = hough_rows[lines_y]
    vals_x = hough_cols[lines_x]

    seqs_y = _get_all_sequences(lines_y)
    seqs_x = _get_all_sequences(lines_x)
    if not seqs_y or not seqs_x:
        return None

    seqs_y_vals = [vals_y[[v in seq for v in lines_y]] for seq in seqs_y]
    seqs_x_vals = [vals_x[[v in seq for v in lines_x]] for seq in seqs_x]
    for i in range(len(seqs_y)):
        seqs_y[i], seqs_y_vals[i] = _trim_sequence(seqs_y[i], seqs_y_vals[i])
    for i in range(len(seqs_x)):
        seqs_x[i], seqs_x_vals[i] = _trim_sequence(seqs_x[i], seqs_x_vals[i])

    best_y = seqs_y[int(np.argmax([np.mean(v) for v in seqs_y_vals]))]
    best_x = seqs_x[int(np.argmax([np.mean(v) for v in seqs_x_vals]))]

    sub_y = [best_y[k:k + 7] for k in range(len(best_y) - 6)]
    sub_x = [best_x[k:k + 7] for k in range(len(best_x) - 6)]

    dy = int(np.median(np.diff(best_y)))
    dx = int(np.median(np.diff(best_x)))
    x0, y0 = int(best_x[0] - dx), int(best_y[0] - dy)
    x1, y1 = int(best_x[-1] + dx), int(best_y[-1] + dy)
    crop = _crop_padded(gray, x0, y0, x1, y1)

    quad = np.ones([8, 8])
    kernel = np.vstack([np.hstack([quad, -quad]), np.hstack([-quad, quad])])
    kernel = np.tile(kernel, (4, 4))
    kernel = kernel / np.linalg.norm(kernel)

    best_score = None
    final = None
    for sx in sub_x:
        for sy in sub_y:
            rx0 = int(sx[0] - dx - x0)
            ry0 = int(sy[0] - dy - y0)
            rx1 = int(sx[-1] + dx - x0)
            ry1 = int(sy[-1] + dy - y0)
            sub = crop[max(0, ry0):ry1, max(0, rx0):rx1]
            if sub.size == 0:
                continue
            sub = cv2.resize(sub, (64, 64), interpolation=cv2.INTER_NEAREST)
            score = abs(float(np.sum(kernel * sub)))
            if best_score is None or score > best_score:
                best_score = score
                final = np.array([rx0 + x0, ry0 + y0, rx1 + x0, ry1 + y0])
    return final


def locate_board(img_bgr: np.ndarray) -> Region | None:
    _, thresh = cv2.threshold(img_bgr, 169, 255, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
    corners = detect_chessboard_corners(gray)
    if corners is None:
        return None
    x0, y0, x1, y1 = (int(v) for v in corners)
    w, h = x1 - x0, y1 - y0
    if h <= 0 or w <= 0 or abs(1 - w / h) > 0.05:
        return None
    height, width = gray.shape
    return (max(0, x0), max(0, y0), min(width, x1), min(height, y1))
