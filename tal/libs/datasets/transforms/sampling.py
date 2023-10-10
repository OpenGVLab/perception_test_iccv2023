import random

import torch


def sparse_sampling_sec(start_sec, end_sec, num_samples, shift=0.5, test_mode=True):
    """
    :param start_sec: start time in seconds
    :param end_sec: end time in seconds
    :param num_samples: number of samples
    :param shift: shift the sampling points by this amount. choices: [0, 0.5, 1]
    :return: a list of time points in seconds
    """
    assert shift in [0, 0.5, 1]
    assert start_sec < end_sec
    assert num_samples > 0
    if test_mode:
        return [start_sec + (end_sec - start_sec) * (i + shift) / num_samples for i in range(num_samples)]
    else:
        return [start_sec + (end_sec - start_sec) * (i + random.random()) / num_samples for i in range(num_samples)]


def batch_sparse_sampling_sec(units, num_samples, shift=0.5, test_mode=True):
    """
    :param units: (start time in seconds,end time in seconds)
    :param num_samples: number of samples
    :param shift: shift the sampling points by this amount. choices: [0, 0.5, 1]
    :return: a list of time points in seconds
    """
    assert shift in [0, 0.5, 1]
    assert num_samples > 0
    if test_mode:
        return torch.stack(
            [units[:, 0] + (units[:, 1] - units[:, 0]) * (i + shift) / num_samples for i in range(num_samples)], dim=1)
    else:
        return torch.stack(
            [units[:, 0] + (units[:, 1] - units[:, 0]) * (i + random.random()) / num_samples for i in
             range(num_samples)], dim=1)


if __name__ == '__main__':
    a1 = sparse_sampling_sec(1, 2, 4, 0.5, True)
    a = sparse_sampling_sec(1, 2, 4, 0.5, False)
    print(a1)
    print(a)
