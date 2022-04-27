import math
from typing import Iterable, List, Any


class Batcher:
    """
    To cut iterable to batch
    # example
    >>> for batch in Batcher([1,2,3,4,5],batch_size=2):
    >>>     print(batch)
    [1, 2]
    [3, 4]
    [5]
    """

    def __init__(self, iterable: Iterable, batch_size: int = 512):
        self.iterable = iterable
        self.batch_size = batch_size

    def __repr__(self) -> str:
        return f"Batched:{len(self)} batches, with batch size: {self.batch_size}"

    def __len__(self,) -> int:
        idx = float(len(self.iterable))/float(self.batch_size)
        return math.ceil(idx)

    def __getitem__(self, idx) -> List[Any]:
        if idx >= len(self):
            raise KeyError(
                f"We have {len(self)} batch, your index is {idx}")
        return self.iterable[idx*self.batch_size:(idx+1)*self.batch_size]

    def __iter__(self) -> Iterable:
        for i in range(len(self)):
            yield self.iterable[i*self.batch_size:(i+1)*self.batch_size]

    def one_batch(self) -> List[Any]:
        return self[0]
