# -*- coding: utf-8 -*-
# Author : xiequanbin
# Date:  18-07-01
# Email:  xquanbin072095@gmail.com


import numpy as np


def cholcov(sigma, flag=1):
    n, m = sigma.shape
    tol = 10 * np.spacing(np.max(np.abs(np.diag(sigma))))
    if (n == m) & np.all(np.all(np.abs(sigma - sigma.T) < tol)):
        try:
            T = np.linalg.cholesky(sigma).T
        except np.linalg.LinAlgError:
            if flag:
                eigen, u = np.linalg.eigh(sigma)
                for i, j in enumerate(np.argmax(np.abs(u), 0)):
                    if u[j][i] < 0:
                        u[:, i] = - u[:, i]

                tol = np.spacing(np.max(eigen)) * len(eigen)
                t = np.abs(eigen) > tol
                eigen = eigen[t]
                p = np.sum(eigen < 0)

                if not p:
                    T = np.dot(np.diag(np.sqrt(eigen)), u[:, t].T)

    return T


def mvnrnd(mean, sigma, size=1):
    # mean: 1-D array_like, of length N
    # sigma: 2-D array_like, of shape (N, N)
    # size: int
    c = cholcov(np.array(sigma))
    r = np.dot(np.random.randn(size, c.shape[0]), c) + np.array(mean)
    if size == 1:
        r = r[0]

    return r