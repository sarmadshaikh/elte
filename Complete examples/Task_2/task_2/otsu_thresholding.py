import numpy as np


def P_k(nk, n):
    return nk/n


def get_histogram(image: np.ndarray):
    histogram = np.zeros(256)
    n = image.shape[0] * image.shape[1]
    for x in range(image.shape[0]):
        for y in range(image.shape[1]):
            histogram[image[x, y]] += 1
    histogram = [P_k(nk, n) for nk in histogram]
    print(histogram)
    return histogram


def q_t(histogram: np.ndarray, q_type: int, t: np.uint8):
    if q_type == 1:
        return sum(histogram[:t+1])
    elif q_type == 2:
        return sum(histogram[t+1:])
    else:
        print("invalid type of q_t")
        return None


def mean(histogram: np.ndarray, mu_type: np.uint8, t: np.uint8) -> np.float32:
    mu = 0
    if mu_type == 1:
        for i in range(t+1):
            mu += i * histogram[i]
    elif mu_type == 2:
        for i in range(t+1, 256):
            mu += i * histogram[i]
    else:
        for i in range(256):
            mu += i * histogram[i]
    return mu


def variance(histogram: np.ndarray, mean: np.float32, sigma_type: np.uint8, t: np.uint8):
    sigma = 0
    if sigma_type == 1:
        for i in range(t+1):
            sigma += (i - mean)**2 * histogram[i]
    elif sigma_type == 2:
        for i in range(t+1, 256):
            sigma += (i - mean)**2 * histogram[i]
    else:
        for i in range(256):
            sigma += (i - mean)**2 * histogram[i]
    return sigma


def variance_between_classes(q1_t: np.float32, mu1_t: np.float32, mu2_t: np.float32) -> np.float32:
    return q1_t * (1 - q1_t) * (mu1_t - mu2_t)**2


def recursive_q1_t(histogram: np.ndarray, t: np.uint8) -> np.float32:
    if t == 0:
        return histogram[0]

    return recursive_q1_t(histogram, t - 1) + histogram[t]


def recursive_mu1_t(histogram: np.ndarray, t: np.uint8) -> np.float32:
    if t == 0:
        return 0
    q_t_minus_1 = recursive_q1_t(histogram, t-1)

    return (q_t_minus_1 * recursive_mu1_t(histogram, t-1) + t * histogram[t]) / (q_t_minus_1 + histogram[t]) if (q_t_minus_1 + histogram[t]) != 0.0 else 0.0


def recursive_mu2_t(histogram: np.ndarray, t: np.uint8):
    mu = mean(histogram, 0, 0)
    q1_t = recursive_q1_t(histogram, t)

    return (mu - q1_t * recursive_mu1_t(histogram, t)) / (1 - q1_t) if (1 - q1_t) != 0.0 else 0.0
