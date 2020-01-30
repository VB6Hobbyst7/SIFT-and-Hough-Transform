import numpy as np

np.seterr(over='ignore', divide='ignore')
import cv2
from scipy import ndimage
from scipy.stats import multivariate_normal
import itertools
import matplotlib.pyplot as plt


def Detect(imgName, Th):
    img = cv2.imread(imgName, 0)

    s = 3
    k = 2 ** (1.0 / s)

    GValue1 = np.array([1.3, 1.6, 1.6 * k, 1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4)])
    GValue2 = np.array([1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4), 1.6 * (k ** 5), 1.6 * (k ** 6), 1.6 * (k ** 7)])
    GValue3 = np.array(
        [1.6 * (k ** 5), 1.6 * (k ** 6), 1.6 * (k ** 7), 1.6 * (k ** 8), 1.6 * (k ** 9), 1.6 * (k ** 10)])
    GValue4 = np.array(
        [1.6 * (k ** 8), 1.6 * (k ** 9), 1.6 * (k ** 10), 1.6 * (k ** 11), 1.6 * (k ** 12), 1.6 * (k ** 13)])
    GValuetotal = np.array(
        [1.6, 1.6 * k, 1.6 * (k ** 2), 1.6 * (k ** 3), 1.6 * (k ** 4), 1.6 * (k ** 5), 1.6 * (k ** 6), 1.6 * (k ** 7),
         1.6 * (k ** 8), 1.6 * (k ** 9), 1.6 * (k ** 10), 1.6 * (k ** 11)])

    doubled = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    normal = cv2.resize(doubled, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    halved = cv2.resize(normal, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
    quartered = cv2.resize(halved, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

    pyr0 = np.zeros((doubled.shape[0], doubled.shape[1], 6))
    pyr1 = np.zeros((normal.shape[0], normal.shape[1], 6))
    pyr2 = np.zeros((halved.shape[0], halved.shape[1], 6))
    pyr3 = np.zeros((quartered.shape[0], quartered.shape[1], 6))

    for i in range(0, 6):
        pyr0[:, :, i] = ndimage.filters.gaussian_filter(doubled, GValue1[i])
        pyr1[:, :, i] = cv2.resize(ndimage.filters.gaussian_filter(doubled, GValue2[i]), None, fx=0.5, fy=0.5,
                                   interpolation=cv2.INTER_LINEAR)
        pyr2[:, :, i] = cv2.resize(ndimage.filters.gaussian_filter(doubled, GValue3[i]), None, fx=0.25, fy=0.25,
                                   interpolation=cv2.INTER_LINEAR)
        pyr3[:, :, i] = cv2.resize(ndimage.filters.gaussian_filter(doubled, GValue4[i]), None, fx=1.0 / 8.0,
                                   fy=1.0 / 8.0, interpolation=cv2.INTER_LINEAR)
    Dpyr0 = np.zeros((doubled.shape[0], doubled.shape[1], 5))
    Dpyr1 = np.zeros((normal.shape[0], normal.shape[1], 5))
    Dpyr2 = np.zeros((halved.shape[0], halved.shape[1], 5))
    Dpyr3 = np.zeros((quartered.shape[0], quartered.shape[1], 5))
    for i in range(0, 5):
        Dpyr0[:, :, i] = pyr0[:, :, i + 1] - pyr0[:, :, i]
        Dpyr1[:, :, i] = pyr1[:, :, i + 1] - pyr1[:, :, i]
        Dpyr2[:, :, i] = pyr2[:, :, i + 1] - pyr2[:, :, i]
        Dpyr3[:, :, i] = pyr3[:, :, i + 1] - pyr3[:, :, i]
    Epyr0 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    Epyr1 = np.zeros((normal.shape[0], normal.shape[1], 3))
    Epyr2 = np.zeros((halved.shape[0], halved.shape[1], 3))
    Epyr3 = np.zeros((quartered.shape[0], quartered.shape[1], 3))

    # octave 1
    for i in range(1, 4):
        for j in range(80, doubled.shape[0] - 80):
            for k in range(80, doubled.shape[1] - 80):
                if np.absolute(Dpyr0[j, k, i]) < Th:
                    continue
                maxbool = (Dpyr0[j, k, i] > 0)
                minbool = (Dpyr0[j, k, i] < 0)
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            maxbool = maxbool and (Dpyr0[j, k, i] > Dpyr0[j + dj, k + dk, i + di])
                            minbool = minbool and (Dpyr0[j, k, i] < Dpyr0[j + dj, k + dk, i + di])
                            if not maxbool and not minbool:
                                break
                        if not maxbool and not minbool:
                            break
                    if not maxbool and not minbool:
                        break
                if maxbool or minbool:
                    dx = (Dpyr0[j, k + 1, i] - Dpyr0[j, k - 1, i]) * 0.5 / 255
                    dy = (Dpyr0[j + 1, k, i] - Dpyr0[j - 1, k, i]) * 0.5 / 255
                    ds = (Dpyr0[j, k, i + 1] - Dpyr0[j, k, i - 1]) * 0.5 / 255
                    dxx = (Dpyr0[j, k + 1, i] + Dpyr0[j, k - 1, i] - 2 * Dpyr0[j, k, i]) * 1.0 / 255
                    dyy = (Dpyr0[j + 1, k, i] + Dpyr0[j - 1, k, i] - 2 * Dpyr0[j, k, i]) * 1.0 / 255
                    dss = (Dpyr0[j, k, i + 1] + Dpyr0[j, k, i - 1] - 2 * Dpyr0[j, k, i]) * 1.0 / 255
                    dxy = (Dpyr0[j + 1, k + 1, i] - Dpyr0[j + 1, k - 1, i] - Dpyr0[j - 1, k + 1, i] + Dpyr0[
                        j - 1, k - 1, i]) * 0.25 / 255
                    dxs = (Dpyr0[j, k + 1, i + 1] - Dpyr0[j, k - 1, i + 1] - Dpyr0[j, k + 1, i - 1] + Dpyr0[
                        j, k - 1, i - 1]) * 0.25 / 255
                    dys = (Dpyr0[j + 1, k, i + 1] - Dpyr0[j - 1, k, i + 1] - Dpyr0[j + 1, k, i - 1] + Dpyr0[
                        j - 1, k, i - 1]) * 0.25 / 255
                    dD = np.matrix([[dx], [dy], [ds]])
                    H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
                    x_hat = np.linalg.lstsq(H, dD, rcond=None)[0]
                    D_x_hat = Dpyr0[j, k, i] + 0.5 * np.dot(dD.transpose(), x_hat)
                    r = 10.0
                    if ((((dxx + dyy) ** 2) * r) < (dxx + dyy - (dxy ** 2)) * (((r + 1) ** 2))) and (
                            np.absolute(x_hat[0]) < 0.5) and (np.absolute(x_hat[1]) < 0.5) and (
                            np.absolute(x_hat[2]) < 0.5) and (np.absolute(D_x_hat) > 0.33):
                        Epyr0[j, k, i - 1] = 1
    # octave 2
    for i in range(1, 4):
        for j in range(40, normal.shape[0] - 40):
            for k in range(40, normal.shape[1] - 40):
                if np.absolute(Dpyr1[j, k, i]) < Th:
                    continue
                maxbool = (Dpyr1[j, k, i] > 0)
                minbool = (Dpyr1[j, k, i] < 0)
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            maxbool = maxbool and (Dpyr1[j, k, i] > Dpyr1[j + dj, k + dk, i + di])
                            minbool = minbool and (Dpyr1[j, k, i] < Dpyr1[j + dj, k + dk, i + di])
                            if not maxbool and not minbool:
                                break
                        if not maxbool and not minbool:
                            break
                    if not maxbool and not minbool:
                        break
                if maxbool or minbool:
                    dx = (Dpyr1[j, k + 1, i] - Dpyr1[j, k - 1, i]) * 0.5 / 255
                    dy = (Dpyr1[j + 1, k, i] - Dpyr1[j - 1, k, i]) * 0.5 / 255
                    ds = (Dpyr1[j, k, i + 1] - Dpyr1[j, k, i - 1]) * 0.5 / 255
                    dxx = (Dpyr1[j, k + 1, i] + Dpyr1[j, k - 1, i] - 2 * Dpyr1[j, k, i]) * 1.0 / 255
                    dyy = (Dpyr1[j + 1, k, i] + Dpyr1[j - 1, k, i] - 2 * Dpyr1[j, k, i]) * 1.0 / 255
                    dss = (Dpyr1[j, k, i + 1] + Dpyr1[j, k, i - 1] - 2 * Dpyr1[j, k, i]) * 1.0 / 255
                    dxy = (Dpyr1[j + 1, k + 1, i] - Dpyr1[j + 1, k - 1, i] - Dpyr1[j - 1, k + 1, i] + Dpyr1[
                        j - 1, k - 1, i]) * 0.25 / 255
                    dxs = (Dpyr1[j, k + 1, i + 1] - Dpyr1[j, k - 1, i + 1] - Dpyr1[j, k + 1, i - 1] + Dpyr1[
                        j, k - 1, i - 1]) * 0.25 / 255
                    dys = (Dpyr1[j + 1, k, i + 1] - Dpyr1[j - 1, k, i + 1] - Dpyr1[j + 1, k, i - 1] + Dpyr1[
                        j - 1, k, i - 1]) * 0.25 / 255
                    dD = np.matrix([[dx], [dy], [ds]])
                    H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
                    x_hat = np.linalg.lstsq(H, dD, rcond=None)[0]
                    D_x_hat = Dpyr1[j, k, i] + 0.5 * np.dot(dD.transpose(), x_hat)
                    r = 10.0
                    if ((((dxx + dyy) ** 2) * r) < (dxx + dyy - (dxy ** 2)) * (((r + 1) ** 2))) and (
                            np.absolute(x_hat[0]) < 0.5) and (np.absolute(x_hat[1]) < 0.5) and (
                            np.absolute(x_hat[2]) < 0.5) and (np.absolute(D_x_hat) > 0.33):
                        Epyr1[j, k, i - 1] = 1
    # octave 3
    for i in range(1, 4):
        for j in range(20, halved.shape[0] - 20):
            for k in range(20, halved.shape[1] - 20):
                if np.absolute(Dpyr2[j, k, i]) < Th:
                    continue
                maxbool = (Dpyr2[j, k, i] > 0)
                minbool = (Dpyr2[j, k, i] < 0)
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            maxbool = maxbool and (Dpyr2[j, k, i] > Dpyr2[j + dj, k + dk, i + di])
                            minbool = minbool and (Dpyr2[j, k, i] < Dpyr2[j + dj, k + dk, i + di])
                            if not maxbool and not minbool:
                                break
                        if not maxbool and not minbool:
                            break
                    if not maxbool and not minbool:
                        break
                if maxbool or minbool:
                    dx = (Dpyr2[j, k + 1, i] - Dpyr2[j, k - 1, i]) * 0.5 / 255
                    dy = (Dpyr2[j + 1, k, i] - Dpyr2[j - 1, k, i]) * 0.5 / 255
                    ds = (Dpyr2[j, k, i + 1] - Dpyr2[j, k, i - 1]) * 0.5 / 255
                    dxx = (Dpyr2[j, k + 1, i] + Dpyr2[j, k - 1, i] - 2 * Dpyr2[j, k, i]) * 1.0 / 255
                    dyy = (Dpyr2[j + 1, k, i] + Dpyr2[j - 1, k, i] - 2 * Dpyr2[j, k, i]) * 1.0 / 255
                    dss = (Dpyr2[j, k, i + 1] + Dpyr2[j, k, i - 1] - 2 * Dpyr2[j, k, i]) * 1.0 / 255
                    dxy = (Dpyr2[j + 1, k + 1, i] - Dpyr2[j + 1, k - 1, i] - Dpyr2[j - 1, k + 1, i] + Dpyr2[
                        j - 1, k - 1, i]) * 0.25 / 255
                    dxs = (Dpyr2[j, k + 1, i + 1] - Dpyr2[j, k - 1, i + 1] - Dpyr2[j, k + 1, i - 1] + Dpyr2[
                        j, k - 1, i - 1]) * 0.25 / 255
                    dys = (Dpyr2[j + 1, k, i + 1] - Dpyr2[j - 1, k, i + 1] - Dpyr2[j + 1, k, i - 1] + Dpyr2[
                        j - 1, k, i - 1]) * 0.25 / 255
                    dD = np.matrix([[dx], [dy], [ds]])
                    H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
                    x_hat = np.linalg.lstsq(H, dD, rcond=None)[0]
                    D_x_hat = Dpyr2[j, k, i] + 0.5 * np.dot(dD.transpose(), x_hat)
                    r = 10.0
                    if ((((dxx + dyy) ** 2) * r) < (dxx + dyy - (dxy ** 2)) * (((r + 1) ** 2))) and (
                            np.absolute(x_hat[0]) < 0.5) and (np.absolute(x_hat[1]) < 0.5) and (
                            np.absolute(x_hat[2]) < 0.5) and (np.absolute(D_x_hat) > 0.33):
                        Epyr2[j, k, i - 1] = 1
    # octave 4
    for i in range(1, 4):
        for j in range(10, quartered.shape[0] - 10):
            for k in range(10, quartered.shape[1] - 10):
                if np.absolute(Dpyr3[j, k, i]) < Th:
                    continue
                maxbool = (Dpyr3[j, k, i] > 0)
                minbool = (Dpyr3[j, k, i] < 0)
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        for dk in range(-1, 2):
                            if di == 0 and dj == 0 and dk == 0:
                                continue
                            maxbool = maxbool and (Dpyr3[j, k, i] > Dpyr3[j + dj, k + dk, i + di])
                            minbool = minbool and (Dpyr3[j, k, i] < Dpyr3[j + dj, k + dk, i + di])
                            if not maxbool and not minbool:
                                break
                        if not maxbool and not minbool:
                            break
                    if not maxbool and not minbool:
                        break
                if maxbool or minbool:
                    dx = (Dpyr3[j, k + 1, i] - Dpyr3[j, k - 1, i]) * 0.5 / 255
                    dy = (Dpyr3[j + 1, k, i] - Dpyr3[j - 1, k, i]) * 0.5 / 255
                    ds = (Dpyr3[j, k, i + 1] - Dpyr3[j, k, i - 1]) * 0.5 / 255
                    dxx = (Dpyr3[j, k + 1, i] + Dpyr3[j, k - 1, i] - 2 * Dpyr3[j, k, i]) * 1.0 / 255
                    dyy = (Dpyr3[j + 1, k, i] + Dpyr3[j - 1, k, i] - 2 * Dpyr3[j, k, i]) * 1.0 / 255
                    dss = (Dpyr3[j, k, i + 1] + Dpyr3[j, k, i - 1] - 2 * Dpyr3[j, k, i]) * 1.0 / 255
                    dxy = (Dpyr3[j + 1, k + 1, i] - Dpyr3[j + 1, k - 1, i] - Dpyr3[j - 1, k + 1, i] + Dpyr3[
                        j - 1, k - 1, i]) * 0.25 / 255
                    dxs = (Dpyr3[j, k + 1, i + 1] - Dpyr3[j, k - 1, i + 1] - Dpyr3[j, k + 1, i - 1] + Dpyr3[
                        j, k - 1, i - 1]) * 0.25 / 255
                    dys = (Dpyr3[j + 1, k, i + 1] - Dpyr3[j - 1, k, i + 1] - Dpyr3[j + 1, k, i - 1] + Dpyr3[
                        j - 1, k, i - 1]) * 0.25 / 255
                    dD = np.matrix([[dx], [dy], [ds]])
                    H = np.matrix([[dxx, dxy, dxs], [dxy, dyy, dys], [dxs, dys, dss]])
                    x_hat = np.linalg.lstsq(H, dD, rcond=None)[0]
                    D_x_hat = Dpyr3[j, k, i] + 0.5 * np.dot(dD.transpose(), x_hat)
                    r = 10.0
                    if ((((dxx + dyy) ** 2) * r) < (dxx + dyy - (dxy ** 2)) * (((r + 1) ** 2))) and (
                            np.absolute(x_hat[0]) < 0.5) and (np.absolute(x_hat[1]) < 0.5) and (
                            np.absolute(x_hat[2]) < 0.5) and (np.absolute(D_x_hat) > 0.33):
                        Epyr3[j, k, i - 1] = 1
    Magpyr0 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    Magpyr1 = np.zeros((normal.shape[0], normal.shape[1], 3))
    Magpyr2 = np.zeros((halved.shape[0], halved.shape[1], 3))
    Magpyr3 = np.zeros((quartered.shape[0], quartered.shape[1], 3))

    Orpyr0 = np.zeros((doubled.shape[0], doubled.shape[1], 3))
    Orpyr1 = np.zeros((normal.shape[0], normal.shape[1], 3))
    Orpyr2 = np.zeros((halved.shape[0], halved.shape[1], 3))
    Orpyr3 = np.zeros((quartered.shape[0], quartered.shape[1], 3))

    for i in range(0, 3):
        for j in range(1, doubled.shape[0] - 1):
            for k in range(1, doubled.shape[1] - 1):
                Magpyr0[j, k, i] = (((doubled[j + 1, k] - doubled[j - 1, k]) ** 2) + (
                            (doubled[j, k + 1] - doubled[j, k - 1]) ** 2)) ** 0.5
                Orpyr0[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((doubled[j, k + 1] - doubled[j, k - 1]),
                                                                           (doubled[j + 1, k] - doubled[j - 1, k])))
    for i in range(0, 3):
        for j in range(1, normal.shape[0] - 1):
            for k in range(1, normal.shape[1] - 1):
                Magpyr1[j, k, i] = (((normal[j + 1, k] - normal[j - 1, k]) ** 2) + (
                            (normal[j, k + 1] - normal[j, k - 1]) ** 2)) ** 0.5
                Orpyr1[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((normal[j, k + 1] - normal[j, k - 1]),
                                                                           (normal[j + 1, k] - normal[j - 1, k])))
    for i in range(0, 3):
        for j in range(1, halved.shape[0] - 1):
            for k in range(1, halved.shape[1] - 1):
                Magpyr2[j, k, i] = (((halved[j + 1, k] - halved[j - 1, k]) ** 2) + (
                            (halved[j, k + 1] - halved[j, k - 1]) ** 2)) ** 0.5
                Orpyr2[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((halved[j, k + 1] - halved[j, k - 1]),
                                                                           (halved[j + 1, k] - halved[j - 1, k])))
    for i in range(0, 3):
        for j in range(1, quartered.shape[0] - 1):
            for k in range(1, quartered.shape[1] - 1):
                Magpyr3[j, k, i] = (((quartered[j + 1, k] - quartered[j - 1, k]) ** 2) + (
                            (quartered[j, k + 1] - quartered[j, k - 1]) ** 2)) ** 0.5
                Orpyr3[j, k, i] = (36 / (2 * np.pi)) * (np.pi + np.arctan2((quartered[j, k + 1] - quartered[j, k - 1]),
                                                                           (quartered[j + 1, k] - quartered[j - 1, k])))
    Extreme_sum = int(np.sum(Epyr0) + np.sum(Epyr1) + np.sum(Epyr2) + np.sum(Epyr3))
    keypoints = np.zeros((Extreme_sum, 4))
    count = 0
    for i in range(0, 3):
        for j in range(80, doubled.shape[0] - 80):
            for k in range(80, doubled.shape[1] - 80):
                if Epyr0[j, k, i] == 1:
                    G_window = multivariate_normal(mean=[j, k], cov=((1.5 * GValuetotal[i]) ** 2))
                    two_sd = np.floor(2 * 1.5 * GValuetotal[i])
                    or_hist = np.zeros([36, 1])
                    for x in range(int(-1 * two_sd * 2), int(two_sd * 2) + 1):
                        ylimit = int((((two_sd * 2) ** 2) - (np.absolute(x) ** 2)) ** 0.5)
                        for y in range(-1 * ylimit, ylimit + 1):
                            if j + x < 0 or j + x > doubled.shape[0] - 1 or k + y < 0 or k + y > doubled.shape[1] - 1:
                                continue
                            w = Magpyr0[j + x, k + y, i] * G_window.pdf([j + x, k + y])
                            bin_ind = np.clip(np.floor(Orpyr0[j + x, k + y, i]), 0, 35)
                            or_hist[int(np.floor(bin_ind))] += w
                    maxval = np.amax(or_hist)
                    maxidx = np.argmax(or_hist)
                    keypoints[count, :] = np.array([int(j * 0.5), int(k * 0.5), GValuetotal[i], maxidx])
                    count += 1
                    or_hist[maxidx] = 0
                    new_maxval = np.amax(or_hist)
                    while new_maxval > 0.8 * maxval:
                        new_maxidx = np.argmax(or_hist)
                        np.append(keypoints, np.array([[int(j * 0.5), int(k * 0.5), GValuetotal[i], new_maxidx]]),
                                  axis=0)
                        or_hist[new_maxidx] = 0
                        new_maxval = np.amax(or_hist)
    for i in range(0, 3):
        for j in range(40, normal.shape[0] - 40):
            for k in range(40, normal.shape[1] - 40):
                if Epyr1[j, k, i] == 1:
                    G_window = multivariate_normal(mean=[j, k], cov=((1.5 * GValuetotal[i]) ** 2))
                    two_sd = np.floor(2 * 1.5 * GValuetotal[i])
                    or_hist = np.zeros([36, 1])
                    for x in range(int(-1 * two_sd * 2), int(two_sd * 2) + 1):
                        ylimit = int((((two_sd * 2) ** 2) - (np.absolute(x) ** 2)) ** 0.5)
                        for y in range(-1 * ylimit, ylimit + 1):
                            if j + x < 0 or j + x > normal.shape[0] - 1 or k + y < 0 or k + y > normal.shape[1] - 1:
                                continue
                            w = Magpyr1[j + x, k + y, i] * G_window.pdf([j + x, k + y])
                            bin_ind = np.clip(np.floor(Orpyr1[j + x, k + y, i]), 0, 35)
                            or_hist[int(np.floor(bin_ind))] += w
                    maxval = np.amax(or_hist)
                    maxidx = np.argmax(or_hist)
                    keypoints[count, :] = np.array([int(j * 0.5), int(k * 0.5), GValuetotal[i], maxidx])
                    count += 1
                    or_hist[maxidx] = 0
                    new_maxval = np.amax(or_hist)
                    while new_maxval > 0.8 * maxval:
                        new_maxidx = np.argmax(or_hist)
                        np.append(keypoints, np.array([[int(j * 0.5), int(k * 0.5), GValuetotal[i], new_maxidx]]),
                                  axis=0)
                        or_hist[new_maxidx] = 0
                        new_maxval = np.amax(or_hist)
    for i in range(0, 3):
        for j in range(20, halved.shape[0] - 20):
            for k in range(20, halved.shape[1] - 20):
                if Epyr2[j, k, i] == 1:
                    G_window = multivariate_normal(mean=[j, k], cov=((1.5 * GValuetotal[i]) ** 2))
                    two_sd = np.floor(2 * 1.5 * GValuetotal[i])
                    or_hist = np.zeros([36, 1])
                    for x in range(int(-1 * two_sd * 2), int(two_sd * 2) + 1):
                        ylimit = int((((two_sd * 2) ** 2) - (np.absolute(x) ** 2)) ** 0.5)
                        for y in range(-1 * ylimit, ylimit + 1):
                            if j + x < 0 or j + x > halved.shape[0] - 1 or k + y < 0 or k + y > halved.shape[1] - 1:
                                continue
                            w = Magpyr2[j + x, k + y, i] * G_window.pdf([j + x, k + y])
                            bin_ind = np.clip(np.floor(Orpyr2[j + x, k + y, i]), 0, 35)
                            or_hist[int(np.floor(bin_ind))] += w
                    maxval = np.amax(or_hist)
                    maxidx = np.argmax(or_hist)
                    keypoints[count, :] = np.array([int(j * 0.5), int(k * 0.5), GValuetotal[i], maxidx])
                    count += 1
                    or_hist[maxidx] = 0
                    new_maxval = np.amax(or_hist)
                    while new_maxval > 0.8 * maxval:
                        new_maxidx = np.argmax(or_hist)
                        np.append(keypoints, np.array([[int(j * 0.5), int(k * 0.5), GValuetotal[i], new_maxidx]]),
                                  axis=0)
                        or_hist[new_maxidx] = 0
                        new_maxval = np.amax(or_hist)
    for i in range(0, 3):
        for j in range(10, quartered.shape[0] - 10):
            for k in range(10, quartered.shape[1] - 10):
                if Epyr3[j, k, i] == 1:
                    G_window = multivariate_normal(mean=[j, k], cov=((1.5 * GValuetotal[i]) ** 2))
                    two_sd = np.floor(2 * 1.5 * GValuetotal[i])
                    or_hist = np.zeros([36, 1])
                    for x in range(int(-1 * two_sd * 2), int(two_sd * 2) + 1):
                        ylimit = int((((two_sd * 2) ** 2) - (np.absolute(x) ** 2)) ** 0.5)
                        for y in range(-1 * ylimit, ylimit + 1):
                            if j + x < 0 or j + x > quartered.shape[0] - 1 or k + y < 0 or k + y > quartered.shape[1] - 1:
                                continue
                            w = Magpyr3[j + x, k + y, i] * G_window.pdf([j + x, k + y])
                            bin_ind = np.clip(np.floor(Orpyr3[j + x, k + y, i]), 0, 35)
                            or_hist[int(np.floor(bin_ind))] += w
                    maxval = np.amax(or_hist)
                    maxidx = np.argmax(or_hist)
                    keypoints[count, :] = np.array([int(j * 0.5), int(k * 0.5), GValuetotal[i], maxidx])
                    count += 1
                    or_hist[maxidx] = 0
                    new_maxval = np.amax(or_hist)
                    while new_maxval > 0.8 * maxval:
                        new_maxidx = np.argmax(or_hist)
                        np.append(keypoints, np.array([[int(j * 0.5), int(k * 0.5), GValuetotal[i], new_maxidx]]),
                                  axis=0)
                        or_hist[new_maxidx] = 0
                        new_maxval = np.amax(or_hist)
    Magpyr = np.zeros((normal.shape[0], normal.shape[1], 12))
    Orpyr = np.zeros((normal.shape[0], normal.shape[1], 12))

    for i in range(0, 3):
        magmax = np.amax(Magpyr0[:, :, i])
        Magpyr[:, :, i] = cv2.resize(Magpyr0[:, :, i], (normal.shape[1], normal.shape[0]),
                                     interpolation=cv2.INTER_LINEAR)
        Magpyr[:, :, i] = (magmax / np.amax(Magpyr[:, :, i])) * Magpyr[:, :, i]
        Orpyr[:, :, i] = cv2.resize(Orpyr0[:, :, i], (normal.shape[1], normal.shape[0]), interpolation=cv2.INTER_LINEAR)
        Orpyr[:, :, i] = ((36.0 / np.amax(Orpyr[:, :, i])) * Orpyr[:, :, i]).astype(float)
    for i in range(0, 3):
        Magpyr[:, :, i + 3] = Magpyr1[:, :, i].astype(float)
        Orpyr[:, :, i + 3] = Orpyr1[:, :, i].astype(int)
    for i in range(0, 3):
        Magpyr[:, :, i + 6] = cv2.resize(Magpyr2[:, :, i], (normal.shape[1], normal.shape[0])).astype(int)
        Orpyr[:, :, i + 6] = cv2.resize(Orpyr2[:, :, i], (normal.shape[1], normal.shape[0])).astype(int)
    for i in range(0, 3):
        Magpyr[:, :, i + 9] = cv2.resize(Magpyr3[:, :, i], (normal.shape[1], normal.shape[0])).astype(int)
        Orpyr[:, :, i + 9] = cv2.resize(Orpyr3[:, :, i], (normal.shape[1], normal.shape[0])).astype(int)

    desc = np.zeros([keypoints.shape[0], 128])

    for i in range(0, keypoints.shape[0]):
        for x in range(-8, 8):
            for y in range(-8, 8):
                theta = 10 * keypoints[i, 3] * np.pi / 180.0
                xrot = np.round((np.cos(theta) * x) - (np.sin(theta) * y))
                yrot = np.round((np.sin(theta) * x) - (np.cos(theta) * y))
                scale_idx = np.argwhere(GValuetotal == keypoints[i, 2])[0][0]
                x0 = keypoints[i, 0]
                y0 = keypoints[i, 1]
                G_window = multivariate_normal(mean=[x0, y0], cov=8)
                w = Magpyr[int(x0 + xrot), int(y0 + yrot), int(scale_idx)] * G_window.pdf([x0 + xrot, y + yrot])
                angle = Orpyr[int(x0 + xrot), int(y0 + yrot), int(scale_idx)] - keypoints[i, 3]
                if angle < 0:
                    angle = 36 + angle
                bin_ind = np.clip(np.floor((8.0 / 36) * angle), 0, 7).astype(int)
                desc[i, 32 * int((x + 8) / 4) + 8 * int((y + 8) / 4) + bin_ind] += w
        desc[i, :] = desc[i, :] / np.linalg.norm(desc[i, :])
        desc[i, :] = np.clip(desc[i, :], 0, 0.2)
        desc[i, :] = desc[i, :] / np.linalg.norm(desc[i, :])
    return [keypoints, desc]


def Match(imgName, TempName, Th, cut):
    img = cv2.imread(imgName)
    template = cv2.imread(TempName)
    [kpi, di] = Detect(imgName, Th)
    print(kpi.shape)
    [kpt, dt] = Detect(TempName, Th)
    print(kpt.shape)

    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(np.asarray(di, np.float32), flann_params)
    idx, dist = flann.knnSearch(np.asarray(dt, np.float32), 1, params={})
    del flann

    dist = dist[:, 0] / 2500.0
    dist = dist.reshape(-1, ).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    sorted(indices,key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]

    kpi_cut = []
    for i, dis in zip(idx, dist):
        if dis < cut:
            kpi_cut.append(kpi[i])
        else:
            break

    kpt_cut = []
    for i, dis in zip(indices, dist):
        if dis < cut:
            kpt_cut.append(kpt[i])
        else:
            break

    h1, w1 = img.shape[:2]
    h2, w2 = template.shape[:2]
    nWidth = w1 + w2
    nHeight = max(h1, h2)
    hdif = (h1 - h2) / 2
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[:int(h2), :int(w2)] = template
    newimg[:int(h1), int(w2):int(w1 + w2)] = img

    for i in range(min(len(kpi), len(kpt))):
        pt_a = (int(kpt[i, 1]), int(kpt[i, 0]))
        pt_b = (int(kpi[i, 1] + w2), int(kpi[i, 0]))
        cv2.line(newimg, pt_a, pt_b, (255, 0, 0))
    for i in range(kpi.shape[0]):
        cv2.circle(img,(int(kpi[i, 1]), int(kpi[i, 0])), 5, (0, 255, 0))
    #cv2.imwrite('Images/EX2/ImageKeypoint.jpg', img)
    cv2.imshow("ImageKeypoint",img)
    for i in range(kpt.shape[0]):
        cv2.circle(template, (int(kpt[i, 1]), int(kpt[i, 0])), 5, (0, 255, 0))
   # cv2.imwrite('Images/EX2/TempKeypoint.jpg', template)
    cv2.imshow("TempKeypoint", template)
    #cv2.imwrite('Images/EX2/matches.jpg', newimg)
    cv2.imshow("match", newimg)
    cv2.waitKey(0)


Match("image2.jpg", "image3.jpg", 1, 5)
