import numpy as np

from typing import (
    Tuple,
    List,
)


class ToNumpy(object):
    def __call__(self, data):
        data["signals"] = np.array(data["signals"])
        data["labels"] = np.array(data["labels"])
        return data

    def __repr__(self):
        return self.__class__.__name__ + "()"


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data
