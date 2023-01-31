from typing import NamedTuple, Tuple, Type, Union, IO, Iterator, Any

API_VERSION: str

PATH_OR_FILE = Union[str, IO]

def hashindex_variant(fn: str) -> str: ...

class IndexBase:
    value_size: int
    MAX_VALUE: int
    MAX_LOAD_FACTOR: int
    def __init__(
        self, capacity: int = ..., path: PATH_OR_FILE = ..., permit_compact: bool = ..., usable: Union[int, float] = ...
    ): ...
    @classmethod
    def read(cls, path: PATH_OR_FILE, permit_compact: bool = False): ...
    def write(self, path: PATH_OR_FILE) -> None: ...
    def clear(self) -> None: ...
    def setdefault(self, key: bytes, value: bytes) -> None: ...
    def __delitem__(self, key: bytes) -> None: ...
    def get(self, key: bytes, default: Any = ...) -> Any: ...
    def pop(self, key: bytes, default: Any = ...) -> Any: ...
    def __len__(self) -> int: ...
    def size(self) -> int: ...
    def compact(self) -> Any: ...

class ChunkIndexEntry(NamedTuple):
    refcount: int
    size: int

CIE = Union[Tuple[int, int], Type[ChunkIndexEntry]]

class ChunkKeyIterator:
    def __init__(self, keysize: int) -> None: ...
    def __iter__(self) -> Iterator: ...
    def __next__(self) -> Tuple[bytes, Type[ChunkIndexEntry]]: ...

class ChunkIndex(IndexBase):
    def add(self, key: bytes, refs: int, size: int) -> None: ...
    def decref(self, key: bytes) -> CIE: ...
    def incref(self, key: bytes) -> CIE: ...
    def iteritems(self, marker: bytes = ...) -> Iterator: ...
    def merge(self, other_index) -> None: ...
    def stats_against(self, master_index) -> Tuple: ...
    def summarize(self) -> Tuple: ...
    def zero_csize_ids(self) -> int: ...
    def __contains__(self, key: bytes) -> bool: ...
    def __getitem__(self, key: bytes) -> Type[ChunkIndexEntry]: ...
    def __setitem__(self, key: bytes, value: CIE) -> None: ...

class NSIndexEntry(NamedTuple):
    segment: int
    offset: int
    size: int

class NSKeyIterator:
    def __init__(self, keysize: int) -> None: ...
    def __iter__(self) -> Iterator: ...
    def __next__(self) -> Tuple[bytes, Type[Any]]: ...

class NSIndex(IndexBase):
    def iteritems(self, *args, **kwargs) -> Iterator: ...
    def __contains__(self, key: bytes) -> bool: ...
    def __getitem__(self, key: bytes) -> Any: ...
    def __setitem__(self, key: bytes, value: Any) -> None: ...
    def flags(self, key: bytes, mask: int, value: int = None) -> int: ...

class NSIndex1(IndexBase):  # legacy
    def iteritems(self, *args, **kwargs) -> Iterator: ...
    def __contains__(self, key: bytes) -> bool: ...
    def __getitem__(self, key: bytes) -> Any: ...
    def __setitem__(self, key: bytes, value: Any) -> None: ...
    def flags(self, key: bytes, mask: int, value: int = None) -> int: ...

class FuseVersionsIndex(IndexBase):
    def __contains__(self, key: bytes) -> bool: ...
    def __getitem__(self, key: bytes) -> Any: ...
    def __setitem__(self, key: bytes, value: Any) -> None: ...

class CacheSynchronizer:
    size_totals: int
    num_files_totals: int
    def __init__(self, chunks_index: Any) -> None: ...
    def feed(self, chunk: bytes) -> None: ...
