import numpy as np


def test_union():
    arr = [1.2, 100.3, 0.5, 0, 2.3]
    arr = np.array(arr, dtype=np.float32)

    min_val = np.min(arr)
    max_val = np.max(arr)

    res = (arr-min_val) / (max_val-min_val+1e-8)
    print(res.tolist())

    # [(x-min_val)/(max_val-min_val) for x in arr]


if __name__ == "__main__":
    test_union()