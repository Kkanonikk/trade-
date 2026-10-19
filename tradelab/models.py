"""Реестр моделей. Каждая: fit(X, y), predict(X) -> ожидаемая доходность следующего бара.
Чтобы добавить свою модель, достаточно написать класс и внести его в MODELS."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class Momentum:
    """Продолжение 20-барного тренда."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["ret20"].to_numpy()


class MeanReversion:
    """Возврат к скользящей средней."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return -X["dist_ma20"].to_numpy()


class Sklearn:
    def __init__(self, factory):
        self.factory = factory
        self.model = None

    def fit(self, X, y):
        self.model = self.factory().fit(X.to_numpy(), y.to_numpy())
        return self

    def predict(self, X):
        return self.model.predict(X.to_numpy())


MODELS = {
    "Моментум (20)": Momentum,
    "Возврат к средней": MeanReversion,
    "Ridge-регрессия": lambda: Sklearn(lambda: make_pipeline(StandardScaler(), Ridge(alpha=10.0))),
    "Градиентный бустинг": lambda: Sklearn(lambda: HistGradientBoostingRegressor(
        max_depth=3, learning_rate=0.05, max_iter=150, random_state=0)),
}


def create(name):
    return MODELS[name]()
