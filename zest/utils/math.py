"""Vectorized rotations. Quaternions are always wxyz; vectors are column vectors."""

import numpy as np


def conjugate(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def multiply(a, b):
    w = a[..., :1] * b[..., :1] - np.sum(a[..., 1:] * b[..., 1:], axis=-1, keepdims=True)
    v = a[..., :1] * b[..., 1:] + b[..., :1] * a[..., 1:] + np.cross(a[..., 1:], b[..., 1:])
    return np.concatenate((w, v), axis=-1)


def rotate(q, v):
    t = 2 * np.cross(q[..., 1:], v)
    return v + q[..., :1] * t + np.cross(q[..., 1:], t)


def inverse_rotate(q, v):
    return rotate(conjugate(q), v)


def rotvec(q):
    q = q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
    q = np.where(q[..., :1] < 0, -q, q)
    n = np.linalg.norm(q[..., 1:], axis=-1, keepdims=True)
    return q[..., 1:] * (2 * np.arctan2(n, q[..., :1]) / np.maximum(n, 1e-12))


def difference(target, actual):
    """Shortest rotation from actual to target, expressed in the parent frame."""
    return rotvec(multiply(target, conjugate(actual)))


def slerp(a, b, fraction):
    dot = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(dot < 0, -b, b)
    dot = np.clip(np.abs(dot), 0, 1)
    theta = np.arccos(dot)
    denom = np.maximum(np.sin(theta), 1e-12)
    q = np.sin((1 - fraction) * theta) / denom * a + np.sin(fraction * theta) / denom * b
    q = np.where(dot > 0.9995, (1 - fraction) * a + fraction * b, q)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1e-12)
