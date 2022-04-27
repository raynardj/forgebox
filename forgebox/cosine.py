import numpy as np
import pandas as pd
from category import Category


class CosineSearch:
    """
    Build a index search on cosine distance

    cos = CosineSearch(base_array)
    idx_order = cos(vec)
    """

    def __init__(self, base):
        assert len(base.shape) == 2,\
            f"Base array has to be 2 dimentional, input is {len(base.shape)}"
        self.base = base
        self.base_norm = self.calc_base_norm(self.base)
        self.normed_base = self.base/self.base_norm[:, None]
        self.dim = self.base.shape[1]

    def __len__(self): return self.base.shape[0]

    @staticmethod
    def calc_base_norm(base: np.ndarray) -> np.ndarray:
        return np.sqrt(np.power(base, 2).sum(1))

    def search(self, vec: np.ndarray, return_similarity: bool = False):
        if return_similarity:
            similarity = (vec * self.normed_base /
                          (np.power(vec, 2).sum())).sum(1)
            order = similarity.argsort()[::-1]
            return order, similarity[order]
        return self(vec)

    def __call__(self, vec: np.ndarray) -> np.ndarray:
        """
        Return the order index of the closest vector to the furthest
        vec: an 1 dimentional vector
        """
        return (vec * self.normed_base).sum(1).argsort()[::-1]


class CosineSearchWithCategory(CosineSearch):
    """
    Combine with the category manager
    The class can return a dataframe with category information

    search_dataframe
    """

    def __init__(self, base: np.ndarray, category: Category):
        super().__init__(base)
        self.category = category
        assert len(self.category) >= len(self), "category number too small"

    def search_dataframe(
        self, vec, return_similarity=True
    ) -> pd.DataFrame:
        """
        return a dataframe from the closest
            category to the furthest
        """
        if return_similarity:
            idx, similarity = self.search(vec, return_similarity)
            return pd.DataFrame({
                "category": self.category.i2c[idx],
                "idx": idx,
                "similarity": similarity})
        idx = self.search(vec, return_similarity)
        return pd.DataFrame({
            "category": self.category.i2c[idx],
            "idx": idx})
