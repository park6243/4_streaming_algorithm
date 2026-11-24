"""Python 3.10+. Run: python streaming_experiment.py [ml-1m.zip]"""
import hashlib
import json
import math
import platform
import random
import statistics
import sys
import time
import urllib.request
import zipfile
from array import array
from collections import Counter
from pathlib import Path

URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
REPEATS = 3
PROBES = 100_000


def events(archive):
    with zipfile.ZipFile(archive) as zipped, zipped.open("ml-1m/ratings.dat") as rows:
        for row in rows:
            yield row.split(b"::", 2)[1]


def hashes(key):
    digest = hashlib.blake2b(key, digest_size=16).digest()
    return int.from_bytes(digest[:8], "little"), int.from_bytes(digest[8:], "little") | 1


class Bloom:
    def __init__(self, bits, count):
        self.bits = bits
        self.count = count
        self.data = bytearray(bits // 8)

    def indices(self, key):
        start, step = hashes(key)
        return ((start + row * step) % self.bits for row in range(self.count))

    def add(self, key):
        for bit in self.indices(key):
            self.data[bit >> 3] |= 1 << (bit & 7)

    def contains(self, key):
        return all(self.data[bit >> 3] & (1 << (bit & 7)) for bit in self.indices(key))


class Sketch:
    def __init__(self, width, depth):
        self.width = width
        self.depth = depth
        self.data = array("I", [0]) * (width * depth)
        if self.data.itemsize != 4:
            raise RuntimeError("This benchmark requires 32-bit unsigned counters.")

    def indices(self, key):
        digest = hashlib.blake2b(key, digest_size=self.depth * 8).digest()
        for row in range(self.depth):
            offset = row * 8
            value = int.from_bytes(digest[offset:offset + 8], "little")
            yield row * self.width + value % self.width

    def add(self, key):
        for index in self.indices(key):
            self.data[index] += 1

    def estimate(self, key):
        return min(self.data[index] for index in self.indices(key))


def measure(factory, archive, count):
    times = []
    for _ in range(REPEATS):
        state = factory()
        start = time.perf_counter()
        for key in events(archive):
            state.add(key)
        times.append(time.perf_counter() - start)
    elapsed = statistics.median(times)
    return state, dict(seconds=elapsed, runs=times, rate=count / elapsed,
                       buffer_bytes=sys.getsizeof(state.data))


def main():
    archive = Path(sys.argv[1] if len(sys.argv) > 1 else "ml-1m.zip")
    if not archive.exists():
        urllib.request.urlretrieve(URL, archive)
    with archive.open("rb") as source:
        checksum = hashlib.file_digest(source, "sha256").hexdigest() if sys.version_info >= (3, 11) else None
    start = time.perf_counter()
    truth = Counter(events(archive))
    exact_seconds = time.perf_counter() - start
    count = sum(truth.values())
    assert count == 1_000_209 and len(truth) == 3706
    observed = set(truth)
    randomizer = random.Random(2026)
    negatives = tuple(str(x).encode() for x in randomizer.sample(range(10_000, 10_010_000), PROBES))
    assert observed.isdisjoint(negatives)
    baseline = dict(events=count, distinct=len(truth), sha256=checksum, counter_seconds=exact_seconds,
                    set_bytes=sys.getsizeof(observed) + sum(map(sys.getsizeof, observed)),
                    counter_bytes=sys.getsizeof(truth) + sum(sys.getsizeof(k) + sys.getsizeof(v)
                                                             for k, v in truth.items()))
    result = dict(python=platform.python_version(), platform=platform.platform(),
                  baseline=baseline, bloom=[], sketch=[])
    for bits, hashes_count in [(16384, 3), (32768, 3), (32768, 5), (32768, 9), (65536, 5)]:
        state, metrics = measure(lambda: Bloom(bits, hashes_count), archive, count)
        fn = sum(not state.contains(key) for key in observed)
        fp = sum(state.contains(key) for key in negatives)
        assert fn == 0
        result["bloom"].append(dict(bits=bits, hashes=hashes_count, fp=fp, fn=fn,
                                    fpr=fp / PROBES, theory=(1-math.exp(-hashes_count*len(truth)/bits))**hashes_count,
                                    payload=len(state.data), **metrics))
    for width, depth in [(512, 3), (1024, 3), (1024, 5), (2048, 5), (4096, 5)]:
        state, metrics = measure(lambda: Sketch(width, depth), archive, count)
        errors = [state.estimate(key) - frequency for key, frequency in truth.items()]
        assert min(errors) == 0
        relative = [error / frequency for error, frequency in zip(errors, truth.values())]
        result["sketch"].append(dict(width=width, depth=depth, mae=statistics.mean(errors),
                                     mre=statistics.mean(relative), exact=errors.count(0)/len(errors),
                                     p95=sorted(errors)[math.ceil(.95*len(errors))-1], max_error=max(errors),
                                     payload=len(state.data)*state.data.itemsize, **metrics))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
