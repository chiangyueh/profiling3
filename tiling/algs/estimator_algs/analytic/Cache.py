from collections import OrderedDict
from tiling.limits import MatmulLimits
from .data import DataInfoBlock, TilingInfo, Traffic, HardwareInfo


class Cache:
    def __init__(self, name: str, max_size: int):
        self.name = name
        self.max_size = max_size
        self.current_size = 0
        self.storage: OrderedDict[str, DataInfoBlock] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def add_data(self, data: DataInfoBlock) -> None:
        if data.name in self.storage:
            self.storage.move_to_end(data.name)
            return
        if data.size > self.max_size:
            raise ValueError(
                f"{data.name}: {data.size} Б не влезает в {self.name} "
                f"({self.max_size} Б) — невалидный тайлинг"
            )
        while self.max_size - self.current_size < data.size and self.storage:
            _, evicted = self.storage.popitem(last=False)
            self.current_size -= evicted.size
        self.storage[data.name] = data
        self.current_size += data.size

    def get_data(self, data_name: str) -> DataInfoBlock | None:
        block = self.storage.get(data_name)
        if block is None:
            self.misses += 1
            return None
        self.storage.move_to_end(data_name)
        self.hits += 1
        return block

    def peek(self, data_name: str) -> DataInfoBlock | None:
        return self.storage.get(data_name)

    def clear(self) -> None:
        self.storage.clear()
        self.current_size = 0

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0

