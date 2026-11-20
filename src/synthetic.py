"""Generador de series sintéticas multivariadas con anomalías.

Copia fiel del generador oficial del benchmark de Lai et al. (2021),
``tods/benchmark/synthetic/Generator/multivariate_generator.py`` (rama
``benchmark`` de github.com/datamllab/tods), que es el que usa el artículo.
Sólo se cambió: (1) configuraciones copiadas por dataset para evitar efectos
colaterales entre llamadas y (2) una función ``make_dataset`` que arma los
15 datasets del artículo con los parámetros del ``__main__`` original.

Tipos de anomalía (Lai et al., 2021; Fig. 1 del artículo):
    0 = puntual global, 1 = puntual contextual, 2 = shapelet (colectiva global),
    3 = estacional, 4 = de tendencia.
El tipo t se inyecta en el canal t, como en el ``__main__`` original.
"""
from __future__ import annotations

import copy

import numpy as np


def sine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    timestamp = np.arange(length)
    value = np.sin(2 * np.pi * freq * timestamp)
    if noise_amp != 0:
        value = value + noise_amp * np.random.normal(0, 1, length)
    return coef * value + offset


def cosine(length, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    timestamp = np.arange(length)
    value = np.cos(2 * np.pi * freq * timestamp)
    if noise_amp != 0:
        value = value + noise_amp * np.random.normal(0, 1, length)
    return coef * value + offset


def square_sine(level=5, length=500, freq=0.04, coef=1.5, offset=0.0, noise_amp=0.05):
    value = np.zeros(length)
    for i in range(level):
        value += 1 / (2 * i + 1) * sine(length=length, freq=freq * (2 * i + 1), coef=coef,
                                        offset=offset, noise_amp=noise_amp)
    return value


class MultivariateDataGenerator:
    def __init__(self, dim, stream_length, behavior, behavior_config):
        self.dim = dim
        self.STREAM_LENGTH = stream_length
        self.behavior = behavior
        self.behavior_config = copy.deepcopy(behavior_config)
        self.data = np.empty(shape=[0, stream_length], dtype=float)
        self.generate_timeseries()

    def generate_timeseries(self):
        for i in range(self.dim):
            self.behavior_config[i]["length"] = self.STREAM_LENGTH
            self.data = np.append(self.data, [self.behavior[i](**self.behavior_config[i])], axis=0)
        self.data_origin = self.data.copy()
        self.label = np.zeros(self.STREAM_LENGTH, dtype=int)

    def point_global_outliers(self, dim_no, ratio, factor, radius):
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio)) * self.STREAM_LENGTH).astype(int)
        maximum, minimum = max(self.data[dim_no]), min(self.data[dim_no])
        for i in position:
            local_std = self.data_origin[dim_no][max(0, i - radius):min(i + radius, self.STREAM_LENGTH)].std()
            self.data[dim_no][i] = self.data_origin[dim_no][i] * factor * local_std
            if 0 <= self.data[dim_no][i] < maximum:
                self.data[dim_no][i] = maximum
            if 0 > self.data[dim_no][i] > minimum:
                self.data[dim_no][i] = minimum
            self.label[i] = 1

    def point_contextual_outliers(self, dim_no, ratio, factor, radius):
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio)) * self.STREAM_LENGTH).astype(int)
        maximum, minimum = max(self.data[dim_no]), min(self.data[dim_no])
        for i in position:
            local_std = self.data_origin[dim_no][max(0, i - radius):min(i + radius, self.STREAM_LENGTH)].std()
            self.data[dim_no][i] = self.data_origin[dim_no][i] * factor * local_std
            if self.data[dim_no][i] > maximum:
                self.data[dim_no][i] = maximum * min(0.95, abs(np.random.normal(0, 1)))
            if self.data[dim_no][i] < minimum:
                self.data[dim_no][i] = minimum * min(0.95, abs(np.random.normal(0, 1)))
            self.label[i] = 1

    def collective_global_outliers(self, dim_no, ratio, radius, coef=3.0, noise_amp=0.0,
                                   level=5, freq=0.04, offset=0.0):
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio / (2 * radius))) * self.STREAM_LENGTH).astype(int)
        sub_data = square_sine(level=level, length=self.STREAM_LENGTH, freq=freq,
                               coef=coef, offset=offset, noise_amp=noise_amp)
        for i in position:
            start, end = max(0, i - radius), min(self.STREAM_LENGTH, i + radius)
            self.data[dim_no][start:end] = sub_data[start:end]
            self.label[start:end] = 1

    def collective_trend_outliers(self, dim_no, ratio, factor, radius):
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio / (2 * radius))) * self.STREAM_LENGTH).astype(int)
        for i in position:
            start, end = max(0, i - radius), min(self.STREAM_LENGTH, i + radius)
            slope = np.random.choice([-1, 1]) * factor * np.arange(end - start)
            self.data[dim_no][start:end] = self.data_origin[dim_no][start:end] + slope
            self.data[dim_no][end:] = self.data[dim_no][end:] + slope[-1]
            self.label[start:end] = 1

    def collective_seasonal_outliers(self, dim_no, ratio, factor, radius):
        position = (np.random.rand(round(self.STREAM_LENGTH * ratio / (2 * radius))) * self.STREAM_LENGTH).astype(int)
        seasonal_config = self.behavior_config[dim_no]
        seasonal_config["freq"] = factor * self.behavior_config[dim_no]["freq"]
        for i in position:
            start, end = max(0, i - radius), min(self.STREAM_LENGTH, i + radius)
            self.data[dim_no][start:end] = self.behavior[dim_no](**seasonal_config)[start:end]
            self.label[start:end] = 1


BEHAVIOR = [sine, cosine, sine, cosine, sine]
BEHAVIOR_CONFIG = [
    {"freq": 0.04, "coef": 1.5, "offset": 0.0, "noise_amp": 0.05},
    {"freq": 0.04, "coef": 2.5, "offset": 0.0, "noise_amp": 0.05},
    {"freq": 0.04, "coef": 1.5, "offset": 0.0, "noise_amp": 0.05},
    {"freq": 0.04, "coef": 2.5, "offset": 2.0, "noise_amp": 0.05},
    {"freq": 0.04, "coef": 1.5, "offset": -2.0, "noise_amp": 0.05},
]

# Los 15 datasets del artículo (Tabla 2).
DATASETS = ["0", "1", "2", "3", "4", "01", "12", "23", "34",
            "012", "123", "234", "0123", "1234", "01234"]


def make_dataset(types: str, seed: int, length: int = 400, ratio: float = 0.05):
    """Genera la serie ``types`` (p. ej. "012") con la semilla ``seed``.
    ``ratio`` es la proporción por tipo (0.05 en el benchmark original)."""
    np.random.seed(seed)
    g = MultivariateDataGenerator(dim=5, stream_length=length, behavior=BEHAVIOR,
                                  behavior_config=BEHAVIOR_CONFIG)
    # Mismo orden de inyección que en el __main__ original.
    if "0" in types:
        g.point_global_outliers(dim_no=0, ratio=ratio, factor=3.5, radius=5)
    if "1" in types:
        g.point_contextual_outliers(dim_no=1, ratio=ratio, factor=2.5, radius=5)
    if "2" in types:
        g.collective_global_outliers(dim_no=2, ratio=ratio, radius=5, coef=1.5, noise_amp=0.03,
                                     level=20, freq=0.04, offset=0.0)
    if "3" in types:
        g.collective_seasonal_outliers(dim_no=3, ratio=ratio, factor=3, radius=5)
    if "4" in types:
        g.collective_trend_outliers(dim_no=4, ratio=ratio, factor=0.5, radius=5)
    return g.data.T.astype(np.float32), g.label.astype(np.int8)
