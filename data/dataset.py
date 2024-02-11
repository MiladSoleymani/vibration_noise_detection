import os
import numpy as np
import pandas as pd
import scipy.signal as ss

from typing import (
    Tuple,
    List,
    Dict,
)


class VibrationNoiseDataset(object):
    """ """

    def __init__(
        self,
        data_path: str,
        time_points: List,
        cutoff: int = 1,
        transforms=None,
        type_use: str = "train",
    ) -> None:

        self.data_path = data_path
        self.time_points = time_points
        self.cutoff = cutoff
        self.transforms = transforms

        self.meta_data = {}
        for root, dirs, files in os.walk(data_path):
            for file in files:
                if os.path.split(root)[-1] not in self.meta_data.keys():
                    self.meta_data[os.path.split(root)[-1]] = []
                self.meta_data[os.path.split(root)[-1]].append(os.path.join(root, file))

        self.type_use = type_use

    def get_engine_names(self) -> np.array:
        return "\n\t\t\t".join(list(np.reshape(list(self.meta_data.values()), (-1,))))

    def get_sampling_rate(self) -> int:
        return 1000

    def chunk_signal(
        self,
        signal: np.array,
        label: int,
        engine_name: str,
        sensor_number: int,
        time_points: List,
        data: Dict = None,
    ) -> Tuple[List, List]:

        if data == None:
            data = {
                "engine_names": [],
                "sensor_number": [],
                "labels": [],
                "part_numbers": [],
                "signals": [],
            }

        for idx, time_point in enumerate(zip(time_points[:-1], time_points[1:])):

            data["engine_names"].append(engine_name)
            data["sensor_number"].append(sensor_number)
            data["labels"].append(label)
            data["part_numbers"].append(idx)
            data["signals"].append(signal[time_point[0] : time_point[1]])

        return data

    def remove_DC(self, signal: np.array, cutoff: int, order: int = 20) -> np.array:
        sos = ss.butter(order, cutoff, "hp", fs=self.get_sampling_rate(), output="sos")
        return ss.sosfilt(sos, signal)

    def __len__(self) -> int:
        return len(list(np.reshape(list(self.meta_data.values()), (-1,))))

    def __getitem__(self):

        data = None

        for _ in self.meta_data.keys():

            for csv_data_path in self.meta_data[_]:

                engine_data = pd.read_csv(csv_data_path)
                engine_data["sensor_1"] = self.remove_DC(
                    np.array(engine_data["sensor_1"]), self.cutoff
                )
                engine_data["sensor_2"] = self.remove_DC(
                    np.array(engine_data["sensor_2"]), self.cutoff
                )

                data = self.chunk_signal(
                    signal=np.array(engine_data["sensor_1"]),
                    label=0 if _ == "OK_clean" else 1,
                    engine_name=os.path.split(csv_data_path)[-1],
                    sensor_number=1,
                    time_points=self.time_points,
                    data=data,
                )

                data = self.chunk_signal(
                    signal=np.array(engine_data["sensor_2"]),
                    label=0 if _ == "OK_clean" else 1,
                    engine_name=os.path.split(csv_data_path)[-1],
                    sensor_number=2,
                    time_points=self.time_points,
                    data=data,
                )

        if self.transforms is not None:
            data = self.transforms(data)

        return data

    def __repr__(self) -> str:
        return (
            "#############################################################\n"
            + f"Dataset              : {self.__class__.__name__}\n"
            + f"# Engine Names       : {self.get_engine_names()}\n"
            + f"# Sampling Rate      : {self.get_sampling_rate()}\n"
            + "#############################################################\n"
        )
