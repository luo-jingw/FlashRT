"""Versioned ImageWAM LIBERO regression-gate fixture: format and storage.

A fixture is one ``fixture.npz`` holding real LIBERO observations already
preprocessed for the served frontend, the conditioning and noise every
run must share, and reference outputs:

======================== =============== ==========================================
array                    shape           meaning
======================== =============== ==========================================
``view1``, ``view2``     (N,224,224,3)   uint8 camera views after the official
                                         center-crop resize (third-person, wrist)
``state``                (N,P) f32       raw proprio (the frontend normalizes it)
``task_index``           (N,) i64        row of ``prompts``/``context_*``
``episode``, ``frame``   (N,) i64        LIBERO source position
``gt_actions``           (N,H,A) f32     ground-truth action chunk, NaN past
                                         ``gt_len``
``gt_len``               (N,) i64        valid ground-truth rows
``prompts``              (T,) str        task instruction per task
``context_bf16_bits``    (T,L,D) u16     official Qwen3 context, bfloat16 bit
                                         pattern (``torch.view(torch.bfloat16)``)
``context_mask``         (T,L) bool      official context mask
``seeds``                (S,) i64        sampler seeds
``noise``                (N,S,H,A) f32   initial action noise, drawn exactly as
                                         the official sampler draws it per seed
``official_actions``     (N,S,H,A) f32   official model output, normalized space
``fp16_reference_actions`` (N,S,H,A) f32 FlashRT ``fp16`` served output,
                                         normalized space
======================== =============== ==========================================

One flag sits beside the arrays: ``text_trim`` records whether
``fp16_reference_actions`` was produced by a frontend running
``text_trim=True`` (the flag removes the padded text keys, issues.md
ISSUE-020), so a run is compared against a reference recorded the same
way. It is not an array and is not stored in the ``.npz``: the manifest
carries it (``FixtureManifest.text_trim``), ``GateFixtureStore.save``
copies the fixture's own value into the manifest, and
``GateFixtureStore.load`` copies the manifest's value back into the
fixture.

``text_trim`` is additive to format 1, which stays the format version: a
manifest written before the field means untrimmed, and that is what
fixture v1 is (``imagewam_libero_gate_v1``, an untrimmed ``fp16``
reference).

A ``FixtureManifest`` records the SHA-256 of the file and of every
array (bytes, shape, dtype), the ``text_trim`` flag, plus generation
metadata. The manifest is committed to git; the ``.npz`` is not.
``GateFixtureStore.load`` refuses a fixture whose file or arrays do not
match the manifest.

Manifest ``metadata`` written by ``benchmarks/imagewam_gate_fixture_generate.py``:
``generator`` (script path), ``generator_sha256`` (SHA-256 of the
generator file that ran), ``git`` (HEAD at the start of generation;
``tracked_changes`` for modified tracked files, ``untracked_files``,
``untracked_count`` and ``clean``, so an uncommitted generator shows),
``source``, ``sampler``, ``checkpoint`` (path, bytes, SHA-256),
``dataset_stats`` (path, SHA-256), ``official``, ``fp16_reference``
(precision, ``text_trim``, profile, frontend call, dims), ``device``,
``torch``, ``peak_gib`` and ``reference_summary``.

``imagewam_libero_gate_v1`` predates ``generator_sha256`` and the
untracked-file record. It was produced by the generator content
committed in ``d03b073`` (file SHA-256 ``d16399e0e6afc3bf...``), with one
difference that does not touch the data: that run took its git snapshot
at the end instead of the start. Its ``git.commit`` is ``ae3a358`` with
``tracked_changes: false`` because the generator was still an untracked
file at that commit.

numpy only; no torch, no GPU.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

FIXTURE_FORMAT_VERSION = 1
FIXTURE_FILE = "fixture.npz"
MANIFEST_FILE = "manifest.json"
_HASH_CHUNK = 1 << 20


@dataclass
class ImageWAMGateFixture:
    """In-memory fixture; see the module docstring for every array.

    ``text_trim`` is not an array: it is the trimming switch
    ``fp16_reference_actions`` was recorded with, and the value
    ``GateFixtureStore`` writes into and reads from the manifest.
    """

    text_trim: bool
    view1: np.ndarray
    view2: np.ndarray
    state: np.ndarray
    task_index: np.ndarray
    episode: np.ndarray
    frame: np.ndarray
    gt_actions: np.ndarray
    gt_len: np.ndarray
    prompts: np.ndarray
    context_bf16_bits: np.ndarray
    context_mask: np.ndarray
    seeds: np.ndarray
    noise: np.ndarray
    official_actions: np.ndarray
    fp16_reference_actions: np.ndarray

    @classmethod
    def array_names(cls) -> tuple[str, ...]:
        """The fixture's array fields, in field order (every field but ``text_trim``)."""
        return ("view1", "view2", "state", "task_index", "episode", "frame", "gt_actions",
                "gt_len", "prompts", "context_bf16_bits", "context_mask", "seeds", "noise",
                "official_actions", "fp16_reference_actions")

    @property
    def num_observations(self) -> int:
        return int(self.view1.shape[0])

    @property
    def num_seeds(self) -> int:
        return int(self.seeds.shape[0])

    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in self.array_names()}

    def validate(self) -> None:
        """Raise ``ValueError`` on any dtype or shape inconsistency, or on a
        ``text_trim`` that is not a bool."""
        n, s = self.num_observations, self.num_seeds
        t = int(self.prompts.shape[0])
        problems = [] if isinstance(self.text_trim, bool) else [
            f"text_trim: {self.text_trim!r} is not a bool"]
        expect_dtype = {
            "view1": np.uint8, "view2": np.uint8, "state": np.float32, "task_index": np.int64,
            "episode": np.int64, "frame": np.int64, "gt_actions": np.float32, "gt_len": np.int64,
            "context_bf16_bits": np.uint16, "context_mask": np.bool_, "seeds": np.int64,
            "noise": np.float32, "official_actions": np.float32, "fp16_reference_actions": np.float32,
        }
        problems += [f"{name}: dtype {getattr(self, name).dtype} != {np.dtype(dtype)}"
                     for name, dtype in expect_dtype.items() if getattr(self, name).dtype != dtype]
        if self.prompts.dtype.kind != "U":
            problems.append(f"prompts: dtype {self.prompts.dtype} is not a unicode string array")
        if self.view1.ndim != 4 or self.view1.shape[-1] != 3 or self.view2.shape != self.view1.shape:
            problems.append(f"views: shapes {self.view1.shape} / {self.view2.shape}")
        per_observation = ("state", "task_index", "episode", "frame", "gt_actions", "gt_len",
                           "noise", "official_actions", "fp16_reference_actions")
        problems += [f"{name}: leading dim {getattr(self, name).shape[0]} != {n}"
                     for name in per_observation if getattr(self, name).shape[0] != n]
        chunk = self.noise.shape[2:]
        for name in ("noise", "official_actions", "fp16_reference_actions"):
            shape = getattr(self, name).shape
            if len(shape) != 4 or shape[1] != s or shape[2:] != chunk:
                problems.append(f"{name}: shape {shape}, expected (N={n}, S={s}, H, A)={chunk}")
        if self.gt_actions.shape[1:] != chunk:
            problems.append(f"gt_actions: shape {self.gt_actions.shape}, chunk {chunk}")
        if self.context_bf16_bits.shape[0] != t or self.context_mask.shape != self.context_bf16_bits.shape[:2]:
            problems.append(f"context: bits {self.context_bf16_bits.shape}, mask {self.context_mask.shape}, prompts {t}")
        if n and (self.task_index.min() < 0 or self.task_index.max() >= t):
            problems.append(f"task_index outside [0, {t})")
        if problems:
            raise ValueError("invalid ImageWAM gate fixture:\n  " + "\n  ".join(problems))


@dataclass(frozen=True)
class ArrayRecord:
    shape: tuple[int, ...]
    dtype: str
    sha256: str

    @classmethod
    def of(cls, array: np.ndarray) -> ArrayRecord:
        contiguous = np.ascontiguousarray(array)
        return cls(shape=tuple(int(x) for x in contiguous.shape), dtype=contiguous.dtype.str,
                   sha256=hashlib.sha256(contiguous.tobytes()).hexdigest())


@dataclass(frozen=True)
class FileRecord:
    sha256: str
    bytes: int

    @classmethod
    def of(cls, path: Path) -> FileRecord:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(_HASH_CHUNK), b""):
                digest.update(chunk)
        return cls(sha256=digest.hexdigest(), bytes=Path(path).stat().st_size)


@dataclass(frozen=True)
class FixtureManifest:
    """Identity of one generated fixture.

    ``text_trim`` is the manifest's record of the flag the fixture's
    ``fp16_reference_actions`` were recorded with. A manifest written
    before the field exists means untrimmed: ``from_json`` applies that
    rule. The field has no Python default, so a caller that builds a
    manifest states the value, and ``GateFixtureStore.save`` always writes
    the fixture's own.
    """

    name: str
    format_version: int
    files: dict[str, FileRecord]
    arrays: dict[str, ArrayRecord]
    metadata: dict[str, object]
    text_trim: bool

    def to_json(self) -> str:
        record = {
            "name": self.name,
            "format_version": self.format_version,
            "text_trim": self.text_trim,
            "files": {k: asdict(v) for k, v in self.files.items()},
            "arrays": {k: dict(asdict(v), shape=list(v.shape)) for k, v in self.arrays.items()},
            "metadata": self.metadata,
        }
        return json.dumps(record, indent=2, sort_keys=True) + "\n"

    @classmethod
    def from_json(cls, text: str) -> FixtureManifest:
        record = json.loads(text)
        return cls(
            name=str(record["name"]), format_version=int(record["format_version"]),
            files={k: FileRecord(sha256=v["sha256"], bytes=int(v["bytes"])) for k, v in record["files"].items()},
            arrays={k: ArrayRecord(shape=tuple(v["shape"]), dtype=v["dtype"], sha256=v["sha256"])
                    for k, v in record["arrays"].items()},
            metadata=dict(record["metadata"]), text_trim=bool(record.get("text_trim", False)))

    @classmethod
    def read(cls, path: Path) -> FixtureManifest:
        return cls.from_json(Path(path).read_text())

    def write(self, path: Path) -> None:
        Path(path).write_text(self.to_json())


class GateFixtureStore:
    """Writes and verifies fixtures in one directory."""

    def __init__(self, directory: Path) -> None:
        self.directory = Path(directory)

    @property
    def fixture_path(self) -> Path:
        return self.directory / FIXTURE_FILE

    def save(self, fixture: ImageWAMGateFixture, name: str, metadata: dict[str, object]) -> FixtureManifest:
        """Write ``fixture.npz`` and ``manifest.json``; return the manifest.

        The manifest's ``text_trim`` is the fixture's own value, so the
        recorded flag and the data it describes cannot disagree.
        """
        fixture.validate()
        self.directory.mkdir(parents=True, exist_ok=True)
        arrays = fixture.arrays()
        np.savez(self.fixture_path, **arrays)
        manifest = FixtureManifest(
            name=name, format_version=FIXTURE_FORMAT_VERSION, text_trim=fixture.text_trim,
            files={FIXTURE_FILE: FileRecord.of(self.fixture_path)},
            arrays={k: ArrayRecord.of(v) for k, v in arrays.items()}, metadata=metadata)
        manifest.write(self.directory / MANIFEST_FILE)
        return manifest

    def load(self, manifest: FixtureManifest) -> ImageWAMGateFixture:
        """Load the fixture after checking it against ``manifest``.

        ``text_trim`` is read from the manifest (the ``.npz`` does not hold
        it), so a loaded fixture reports the flag its reference was recorded
        with and a caller can compare it with the configuration it runs.
        """
        if manifest.format_version != FIXTURE_FORMAT_VERSION:
            raise ValueError(f"fixture format {manifest.format_version} != supported {FIXTURE_FORMAT_VERSION}")
        if not self.fixture_path.is_file():
            raise FileNotFoundError(f"fixture file missing: {self.fixture_path}")
        expected_file = manifest.files[FIXTURE_FILE]
        actual_file = FileRecord.of(self.fixture_path)
        if actual_file != expected_file:
            raise ValueError(f"{self.fixture_path}: {actual_file} does not match manifest {expected_file}")
        with np.load(self.fixture_path, allow_pickle=False) as data:
            arrays = {name: data[name] for name in data.files}
        names = set(ImageWAMGateFixture.array_names())
        if set(arrays) != names or set(manifest.arrays) != names:
            raise ValueError(f"fixture arrays {sorted(arrays)} / manifest {sorted(manifest.arrays)} "
                             f"!= format {sorted(names)}")
        mismatched = [name for name, array in arrays.items() if ArrayRecord.of(array) != manifest.arrays[name]]
        if mismatched:
            raise ValueError(f"fixture arrays differ from the manifest: {mismatched}")
        fixture = ImageWAMGateFixture(text_trim=manifest.text_trim, **arrays)
        fixture.validate()
        return fixture
