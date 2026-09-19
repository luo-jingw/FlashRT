"""The bounded per-length graph cache and its startup fill.

CPU only: no CUDA device, no compiled extension, no frontend construction
(the constructor allocates CUDA tensors, so `__init__` is mocked the way
`tests/test_imagewam_public_entry.py` does it). What is pinned here:

* `evict_lru` (`imagewam_thor.py`) is the whole eviction decision: a pure
  function of a mapping in least-recently-used order, a bound and the active
  length; it returns the keys to drop, in order, never the active one, and
  nothing while the cache is within its bound;
* `text_trim_cache_size` is a resolver option (`ImageWAMOptions`,
  `EXPERT_KEYS`), `>= 1` (rule V1), and it reaches the constructor through
  `frontend_kwargs_from_config`;
* `from_config` and `load_imagewam` run the frontend's own
  `precapture_text_lengths` once, with the lengths they were given, and not
  at all for `None`.

What a capture does, and what the bound does to a real cache, is only
observable on a captured graph: `tests/test_imagewam_text_trim.py` covers
that on a GPU. The resolver's option-set pin lives in
`tests/test_imagewam_frontend_from_config.py`.
"""
from __future__ import annotations

import inspect
from dataclasses import fields
from unittest import mock

import pytest

from flash_rt.frontends.torch.imagewam_thor import (
    ImageWAMTorchFrontendThor,
    evict_lru,
    frontend_kwargs_from_config,
    load_imagewam,
)
from flash_rt.models.imagewam.config_resolver import (
    EXPERT_KEYS,
    ConfigError,
    ImageWAMOptions,
    resolve_config,
)
from flash_rt.models.imagewam.structure import ImageWAMStructure
from flash_rt.models.imagewam.workload import ImageWAMWorkload

WORKLOAD = ImageWAMWorkload.libero()
STRUCTURE = ImageWAMStructure.libero()
CKPT = "/models/imagewam/model.pt"     # never read: `structure` is passed
LENGTHS = [6, 13]


def _cache(*lengths: int) -> dict[int, object]:
    """A capture cache in least-recently-used to most-recently-used order.

    `evict_lru` reads the keys and the mapping's order only (`dict` keeps its
    insertion order, which is the frontend's recency order), so opaque
    sentinels stand in for the `TextLengthCapture` values here.
    """
    return {x0: object() for x0 in lengths}


# -- evict_lru: the whole eviction decision ----------------------------------


def test_evict_lru_within_the_bound_drops_nothing():
    captures = _cache(3, 5, 7)
    assert evict_lru(captures, active_x0=7, limit=3) == ()
    assert evict_lru(captures, active_x0=7, limit=32) == ()
    assert evict_lru({}, active_x0=None, limit=1) == ()
    assert tuple(captures) == (3, 5, 7)     # a reader: nothing was dropped here


def test_evict_lru_drops_the_least_recently_used_first():
    """The number of keys returned is the excess over the bound, oldest
    first, so a caller deleting them in that order ends at the bound."""
    captures = _cache(3, 5, 7, 9)
    assert evict_lru(captures, active_x0=9, limit=3) == (3,)
    assert evict_lru(captures, active_x0=9, limit=2) == (3, 5)
    assert evict_lru(captures, active_x0=9, limit=1) == (3, 5, 7)


def test_evict_lru_limit_one_with_two_entries_drops_the_older():
    captures = _cache(6, 13)
    assert evict_lru(captures, active_x0=13, limit=1) == (6,)
    assert evict_lru(_cache(6, 13), active_x0=6, limit=1) == (13,)


def test_evict_lru_never_returns_the_active_length():
    """The active capture is the graph `infer()` replays: it is skipped, and
    the keys after it are dropped in its place."""
    assert evict_lru(_cache(3, 5, 7), active_x0=5, limit=1) == (3, 7)
    assert evict_lru(_cache(3, 5, 7), active_x0=7, limit=1) == (3, 5)
    assert evict_lru(_cache(3, 5, 7, 9), active_x0=5, limit=1) == (3, 7, 9)


def test_evict_lru_leaves_a_cache_of_only_the_active_capture_alone():
    """Nothing is droppable, so nothing is returned -- `()` -- and the
    caller's cache still holds the active capture (`len(captures) == 1`,
    above a bound of zero), which keeps its state valid: `infer()` still has
    a graph to replay."""
    captures = _cache(5)
    assert evict_lru(captures, active_x0=5, limit=1) == ()   # within the bound
    assert evict_lru(captures, active_x0=5, limit=0) == ()   # above it: still nothing to drop
    assert tuple(captures) == (5,)


def test_evict_lru_without_an_active_length_may_drop_any():
    """`active_x0=None` is the frontend with no captured graph: no length is
    protected, so the oldest keys go."""
    assert evict_lru(_cache(3, 5, 7), active_x0=None, limit=2) == (3,)
    assert evict_lru(_cache(3, 5, 7), active_x0=None, limit=1) == (3, 5)
    # an active length that is not cached protects nothing either
    assert evict_lru(_cache(3, 5, 7), active_x0=99, limit=2) == (3,)


# -- the resolver's bound -----------------------------------------------------


def test_the_bound_is_a_resolver_option():
    assert "text_trim_cache_size" in EXPERT_KEYS
    assert "text_trim_cache_size" in {f.name for f in fields(ImageWAMOptions)}
    assert resolve_config(WORKLOAD, STRUCTURE).options.text_trim_cache_size == 32


def test_a_legal_bound_resolves_and_reaches_the_frontend():
    """The resolver's option, the mapping and `from_config`'s constructor
    call agree on one value, without building a frontend."""
    resolved = resolve_config(WORKLOAD, STRUCTURE, text_trim=True, text_trim_cache_size=4)
    assert resolved.options.text_trim_cache_size == 4
    assert frontend_kwargs_from_config(resolved)["text_trim_cache_size"] == 4
    parameters = inspect.signature(ImageWAMTorchFrontendThor.__init__).parameters
    assert parameters["text_trim_cache_size"].default == 32

    calls: list[dict] = []

    def record(self, **kwargs) -> None:
        calls.append(dict(kwargs))

    with mock.patch.object(ImageWAMTorchFrontendThor, "__init__", record):
        ImageWAMTorchFrontendThor.from_config(resolved)
    assert calls[0]["text_trim_cache_size"] == 4

    # a length set larger than the bound still resolves: the bound is applied
    # by the frontend, not by the resolver
    assert resolve_config(WORKLOAD, STRUCTURE, text_trim_cache_size=64).options.text_trim_cache_size == 64


@pytest.mark.parametrize("value", [0, -1, "32", 2.5, True, None])
def test_an_illegal_bound_raises_V1(value):
    with pytest.raises(ConfigError) as excinfo:
        resolve_config(WORKLOAD, STRUCTURE, text_trim_cache_size=value)
    assert excinfo.value.rule == "V1"
    assert str(excinfo.value).startswith("V1: ")
    assert "text_trim_cache_size" in str(excinfo.value)


@pytest.mark.parametrize("value", [0, -1, "32", 2.5, True])
def test_the_constructor_refuses_the_same_values(value):
    """The constructor's own domain check (rule V1's counterpart for a
    caller that does not go through the resolver). It runs before any buffer
    is allocated, so it raises here without a CUDA device: no frontend is
    constructed."""
    with pytest.raises(ValueError, match="text_trim_cache_size"):
        ImageWAMTorchFrontendThor(text_trim_cache_size=value)


# -- startup precapture -------------------------------------------------------


def test_from_config_precaptures_the_given_lengths_once():
    """`precapture_text_lengths` is not a constructor argument: `from_config`
    runs the frontend's own method once, after construction, with the
    sequence it was given."""
    resolved = resolve_config(WORKLOAD, STRUCTURE, text_trim=True)
    with mock.patch.object(ImageWAMTorchFrontendThor, "__init__", lambda self, **kw: None), \
            mock.patch.object(ImageWAMTorchFrontendThor, "precapture_text_lengths") as precapture:
        frontend = ImageWAMTorchFrontendThor.from_config(
            resolved, workload=WORKLOAD, precapture_text_lengths=LENGTHS)
    precapture.assert_called_once_with(LENGTHS)
    assert frontend.resolved_config is resolved and frontend._workload is WORKLOAD


def test_from_config_without_lengths_precaptures_nothing():
    resolved = resolve_config(WORKLOAD, STRUCTURE, text_trim=True)
    with mock.patch.object(ImageWAMTorchFrontendThor, "__init__", lambda self, **kw: None), \
            mock.patch.object(ImageWAMTorchFrontendThor, "precapture_text_lengths") as precapture:
        ImageWAMTorchFrontendThor.from_config(resolved, workload=WORKLOAD)
        ImageWAMTorchFrontendThor.from_config(resolved, workload=WORKLOAD, precapture_text_lengths=None)
    precapture.assert_not_called()


def test_load_imagewam_passes_the_lengths_to_from_config():
    with mock.patch.object(ImageWAMTorchFrontendThor, "from_config",
                           return_value="frontend") as from_config:
        assert load_imagewam(CKPT, WORKLOAD, structure=STRUCTURE,
                             precapture_text_lengths=(6, 13)) == "frontend"
        assert load_imagewam(CKPT, WORKLOAD, structure=STRUCTURE) == "frontend"
    forwarded = [call.kwargs["precapture_text_lengths"] for call in from_config.call_args_list]
    assert forwarded == [(6, 13), None]


def test_load_imagewam_precaptures_the_given_lengths_once():
    """The deployment entry, end to end up to the capture itself: the real
    `from_config` runs, the constructor and the capture are mocked."""
    with mock.patch.object(ImageWAMTorchFrontendThor, "__init__", lambda self, **kw: None), \
            mock.patch.object(ImageWAMTorchFrontendThor, "precapture_text_lengths") as precapture:
        load_imagewam(CKPT, WORKLOAD, structure=STRUCTURE, text_trim=True,
                      precapture_text_lengths=LENGTHS)
    precapture.assert_called_once_with(LENGTHS)


def test_load_imagewam_without_lengths_precaptures_nothing():
    with mock.patch.object(ImageWAMTorchFrontendThor, "__init__", lambda self, **kw: None), \
            mock.patch.object(ImageWAMTorchFrontendThor, "precapture_text_lengths") as precapture:
        load_imagewam(CKPT, WORKLOAD, structure=STRUCTURE, text_trim=True)
    precapture.assert_not_called()
