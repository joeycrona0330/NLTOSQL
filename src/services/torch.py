from sklearn.metrics import accuracy_score
import numpy as np


def get_numpy(arr, gpu=False):
    if gpu:
        return arr.data.cuda().cpu()
    else:
        return arr.data.cpu().numpy()


def accuracy(true_output, logits, gpu=False):
    logits = get_numpy(logits, gpu)
    predicted_output = np.argmax(logits, 1)
    true_output = get_numpy(true_output, gpu)
    return accuracy_score(true_output, predicted_output)


def append(a, b, gpu=False):
    b = get_numpy(b, gpu)
    return np.append(a, b)
