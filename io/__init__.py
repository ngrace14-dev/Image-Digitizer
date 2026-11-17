"""
Embroidery file I/O package (exporters and vector parsers).
Also re-exports standard library `_io` symbols so top-level `io/` directory
does not break stdlib modules that `import io`.
"""

import _io
from _io import (  # noqa: F401
     DEFAULT_BUFFER_SIZE,
    BlockingIOError,
    BufferedRandom,
    BufferedReader,
    BufferedRWPair,
    BufferedWriter,
    BytesIO,
    FileIO,
    IncrementalNewlineDecoder,
     RawIOBase,
    StringIO,
    TextIOWrapper,
    UnsupportedOperation,
    open,
    open_code,
)
import abc

SEEK_SET = 0
SEEK_CUR = 1
SEEK_END = 2


class IOBase(_io._IOBase, metaclass=abc.ABCMeta):
    __doc__ = _io._IOBase.__doc__


class RawIOBase(_io._RawIOBase, IOBase):
    __doc__ = _io._RawIOBase.__doc__


class BufferedIOBase(_io._BufferedIOBase, IOBase):
    __doc__ = _io._BufferedIOBase.__doc__


class TextIOBase(_io._TextIOBase, IOBase):
    __doc__ = _io._TextIOBase.__doc__


RawIOBase.register(FileIO)
for _klass in (BytesIO, BufferedReader, BufferedWriter, BufferedRandom, BufferedRWPair):
    BufferedIOBase.register(_klass)
for _klass in (StringIO, TextIOWrapper):
    TextIOBase.register(_klass)
del _klass

try:
    from _io import text_encoding  # type: ignore # noqa: F401
except ImportError:
    pass

