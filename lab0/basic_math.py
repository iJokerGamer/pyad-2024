import numpy as np
import scipy as sc


def matrix_multiplication(matrix_a, matrix_b):
    shape_a, shape_b = (
        (len(matrix_a), len(matrix_a[0])),
        (len(matrix_b), len(matrix_b[0])),
    )

    if shape_a[1] != shape_b[0]:
        raise ValueError(f"invalid matrix shape: {shape_a=}, {shape_b=}")

    matrix_c = [
        [
            sum(matrix_a[i][k] * matrix_b[k][j] for k in range(shape_a[1]))
            for j in range(shape_b[1])
        ]
        for i in range(shape_a[0])
    ]

    return matrix_c
    
    


def functions(a_1, a_2):
    coefs_1 = np.array(list(map(float, a_1.split())))
    coefs_2 = np.array(list(map(float, a_2.split())))

    if np.array_equal(coefs_1, coefs_2):
        return None

    F = lambda x: np.polyval(coefs_1, x)

    A, B, C = coefs_1 - coefs_2

    if A == 0:
        if B == 0:
            return [] if C != 0 else None
        root = -C / B
        return [(root, F(root))]

    D = B ** 2 - 4 * A * C

    if D < 0:
        return []

    if D == 0:
        root = -B / (2 * A)
        return [(root, F(root))]

    root1 = (-B + np.sqrt(D)) / (2 * A)
    root2 = (-B - np.sqrt(D)) / (2 * A)
    return sorted([(root1, F(root1)), (root2, F(root2))])
    
def calculate_moments(x, n):
    return np.mean((x - np.mean(x)) ** n)

def skew(x):
    return round(calculate_moments(x, 3) / calculate_moments(x, 2) ** (3/2), 2)

def kurtosis(x):
    
    return round(calculate_moments(x, 4) / calculate_moments(x, 2) ** 2 - 3, 2)
    

