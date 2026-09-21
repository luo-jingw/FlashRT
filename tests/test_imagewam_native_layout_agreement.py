"""The C++ config structs, their ctypes mirrors and the field-name sets agree.

`flash_rt/models/imagewam/native_library.py` mirrors `frt_imagewam_io_config`
and `frt_imagewam_pipeline_config`
(cpp/models/imagewam/include/flashrt/cpp/models/imagewam/c_api.h) as ctypes
structures, and `ImageWAMNativeLibrary._check_layout` compares only the four
`sizeof` values the library reports (`frt_imagewam_native_abi_sizes`). A
size-neutral reorder of the members — on either side — is invisible to that
check, and a device pointer written into the wrong slot of a struct the C++
side reads by name (`img_raw` / `context` / `action_latent`,
`state_scale` / `state_offset`, the four `frt_imagewam_*_linear` members of
`frt_imagewam_single_layer`) is not a size difference. The header's member
names and their order are therefore pinned here against the mirrors, as C
source text: no C compiler and no new dependency, and the header stays the
only copy of its own field lists.

Two more name sets are coupled to the header by convention only:
`PIPELINE_DIM_FIELDS` and `PIPELINE_BUFFER_FIELDS` (`native_library.py`) are
the names `build_pipeline_config` copies out of `PipelineDims` /
`PipelineBuffers` (`pipeline_resources.py`), so a field a dataclass gains but
a list does not is zero-initialised in the struct the C++ pipeline reads, and
a field a list gains in the wrong order routes the wrong number into the
pipeline's dimensions.

CPU only: the header is parsed as text and the mirrors are classes
(`ctypes.sizeof` needs no device), so nothing here is skipped on a machine
without a GPU. What is checked:

- the io config mirror: the header's member names, in order, against
  `ImageWAMIoConfig._fields_`;
- the pipeline config mirror: the same against
  `ImageWAMPipelineConfig._fields_`, each nested struct member (by value or
  through a pointer) expanded into the subfields the header lists for that
  struct type;
- `PIPELINE_DIM_FIELDS` against the header's dim block and
  `PIPELINE_BUFFER_FIELDS` against its buffer block, both in header order;
- both lists against the fields of the dataclasses they are copied from.
"""
from __future__ import annotations

import ctypes
import dataclasses
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from flash_rt.models.imagewam.native_library import (
    PIPELINE_BUFFER_FIELDS,
    PIPELINE_DIM_FIELDS,
    ImageWAMIoConfig,
    ImageWAMPipelineConfig,
)
from flash_rt.models.imagewam.pipeline_resources import PipelineBuffers, PipelineDims

HEADER = Path(__file__).resolve().parents[1] / (
    "cpp/models/imagewam/include/flashrt/cpp/models/imagewam/c_api.h")

# The two config structs the library's `_check_layout` sizes, and the ctypes
# mirror of each.
MIRRORS = (
    ("frt_imagewam_io_config", ImageWAMIoConfig),
    ("frt_imagewam_pipeline_config", ImageWAMPipelineConfig),
)

# The pipeline config's blocks, each read as the header's members between two
# member names of the header itself (`struct_size` and `eps` are the last
# scalar before the dims and the last one before the buffers; `q_o` is the
# first member of the attention block that follows them).
DIM_BLOCK = ("struct_size", "eps")
BUFFER_BLOCK = ("eps", "q_o")

_COMMENT = re.compile(r"/\*.*?\*/|//[^\n]*", re.S)
_TYPE_DEFINITION = re.compile(
    r"typedef\s+struct\s+(?P<name>[A-Za-z_]\w*)\s*\{(?P<body>[^{}]*)\}\s*(?P<alias>[A-Za-z_]\w*)\s*;",
    re.S)
_MEMBER = re.compile(
    r"^(?P<type>(?:const\s+)?(?:(?:struct|union|enum)\s+)?[A-Za-z_]\w*)\s*(?P<declarators>.+)$", re.S)
_IDENTIFIER = re.compile(r"[A-Za-z_]\w*")


@dataclass(frozen=True)
class _HeaderMember:
    """One member a struct in the header declares: its name, the type it is
    declared with (qualifiers and `struct`/`union`/`enum` tags dropped),
    whether its declarator is a pointer, and — when the header declares that
    type as a struct — that struct's own members, so a nested member expands
    into the subfields the header lists for it."""

    name: str
    type_name: str
    pointer: bool
    fields: tuple["_HeaderMember", ...] | None


def _strip_comments(text: str) -> str:
    """The header without its comments: every field is followed by a comment
    that carries `,` and `;` of its own."""
    return _COMMENT.sub(" ", text)


def _struct_bodies(text: str) -> dict[str, str]:
    """`{struct name: body}` for every `typedef struct NAME { ... } NAME;` in
    `text`, comments stripped."""
    return {match.group("name"): match.group("body")
            for match in _TYPE_DEFINITION.finditer(_strip_comments(text))}


def _tagged_type(type_text: str) -> tuple[str, bool]:
    """The type name one declaration starts with and whether it was written
    with a tag: `const uint32_t` -> `("uint32_t", False)`,
    `struct frt_imagewam_linear` -> `("frt_imagewam_linear", True)`."""
    tagged = any(f" {tag} " in f" {type_text} " for tag in ("struct", "union", "enum"))
    name = type_text
    for qualifier in ("const ", "struct ", "union ", "enum "):
        name = name.removeprefix(qualifier)
    return name.strip(), tagged


def _raw_members(body: str) -> tuple[tuple[str, str, bool], ...]:
    """`(name, type text, is a pointer declarator)` per member of one struct
    body, in declaration order: a statement declares one type and one or more
    declarators (`void *context, *img_raw, ...` declares all of them)."""
    members: list[tuple[str, str, bool]] = []
    for statement in body.split(";"):
        statement = statement.strip()
        if not statement:
            continue
        match = _MEMBER.match(statement)
        if match is None:
            raise AssertionError(f"cannot read the header member {statement!r}")
        type_text = match.group("type").strip()
        for declarator in match.group("declarators").split(","):
            declarator = declarator.strip()
            names = _IDENTIFIER.findall(declarator)
            if not names:
                raise AssertionError(f"cannot read the declarator {declarator!r} of {statement!r}")
            members.append((names[-1], type_text, "*" in declarator))
    return tuple(members)


def _members_of(struct_name: str, raw: dict[str, tuple[tuple[str, str, bool], ...]],
                stack: tuple[str, ...] = ()) -> tuple[_HeaderMember, ...]:
    """The members of one header struct; a member whose type name is another
    struct of the header carries that struct's members, resolved the same way,
    so the header stays the source of truth for the nested layout too. A
    member declared with a `struct` / `union` / `enum` tag the header declares
    no body for is reported instead of being left unexpanded."""
    if struct_name in stack:
        raise AssertionError(f"the header nests {struct_name} inside itself: "
                             f"{' -> '.join((*stack, struct_name))}")
    members: list[_HeaderMember] = []
    for name, type_text, pointer in raw[struct_name]:
        type_name, tagged = _tagged_type(type_text)
        if type_name in raw:
            fields = _members_of(type_name, raw, (*stack, struct_name))
        else:
            if tagged:
                raise AssertionError(f"{struct_name}.{name} is declared with the tag {type_text!r}, "
                                     f"but the header declares no struct {type_name!r} to expand")
            fields = None
        members.append(_HeaderMember(name, type_name, pointer, fields))
    return tuple(members)


def _header_structs(text: str) -> dict[str, tuple[_HeaderMember, ...]]:
    """Every struct the header declares, each member expanded as
    `_members_of` describes."""
    raw = {name: _raw_members(body) for name, body in _struct_bodies(text).items()}
    return {name: _members_of(name, raw) for name in raw}


def _nested_struct_names(members: Sequence[_HeaderMember]) -> set[str]:
    """Every struct type the header nests inside `members`, at any depth: the
    set of expansions a full comparison of `members` has to walk."""
    nested: set[str] = set()
    for member in members:
        if member.fields is None:
            continue
        nested.add(member.type_name)
        nested |= _nested_struct_names(member.fields)
    return nested


def _structure_type(field_type: object) -> type | None:
    """The ctypes struct behind one mirror field: the field type itself when
    the member is by value, its pointee when the member is a pointer
    (`frt_imagewam_linear txt_in` and `const frt_imagewam_single_layer*
    single_layers` both expand into the struct the header declares)."""
    if isinstance(field_type, type) and issubclass(field_type, ctypes.Structure):
        return field_type
    pointee = getattr(field_type, "_type_", None)
    if isinstance(pointee, type) and issubclass(pointee, ctypes.Structure):
        return pointee
    return None


def _difference(where: str, header: Sequence[str], mirror: Sequence[str]) -> str:
    """The first position where two name lists differ, as one line ("" when
    they are equal): a reorder on either side names its own position."""
    for position, (header_name, mirror_name) in enumerate(zip(header, mirror)):
        if header_name != mirror_name:
            return (f"{where}: field {position} is the header's {header_name!r} and the mirror's "
                    f"{mirror_name!r}")
    if len(header) != len(mirror):
        position = min(len(header), len(mirror))
        side, extra = (("header", header[position]) if len(header) > len(mirror)
                       else ("mirror", mirror[position]))
        return (f"{where}: field {position} is {extra!r} in the {side} alone ({len(header)} header "
                f"names, {len(mirror)} mirror names)")
    return ""


def _compare(where: str, header: Sequence[_HeaderMember], mirror: Sequence[tuple],
             compared: set[str]) -> None:
    """Assert the header's members and one mirror's `_fields_` carry the same
    names in the same order, recursing into every nested struct member and
    recording the struct types it expanded."""
    difference = _difference(where, [member.name for member in header],
                            [field[0] for field in mirror])
    assert not difference, difference
    for member, field in zip(header, mirror):
        if member.fields is None:
            continue
        nested = _structure_type(field[1])
        assert nested is not None, (f"{where}.{member.name} is a {member.type_name} member in the "
                                    f"header, but the mirror field is {field[1]!r}")
        compared.add(member.type_name)
        _compare(f"{where}.{member.name}", member.fields, nested._fields_, compared)


def test_the_ctypes_mirrors_declare_the_header_members_in_the_header_order():
    """Each config struct's members are the mirror's `_fields_` names, in
    order, including the members declared through the header's nested struct
    types — by value (`frt_imagewam_linear txt_in, img_in, ...`: four members,
    not one) and through a pointer (`frt_imagewam_single_layer*`, whose own
    four `frt_imagewam_linear` members are a size-neutral swap)."""
    structs = _header_structs(HEADER.read_text())
    compared: set[str] = set()
    expected: set[str] = set()
    for struct_name, mirror in MIRRORS:
        header = structs[struct_name]
        print(f"{struct_name}: {len(header)} header members == mirror fields "
              f"{[field[0] for field in mirror._fields_]}")
        _compare(struct_name, header, mirror._fields_, compared)
        expected |= _nested_struct_names(header)
    print(f"nested structs expanded through the mirrors: {sorted(compared)}")
    assert compared == expected, ("the header nests " + str(sorted(expected)) +
                                 ", but the mirrors were expanded through " + str(sorted(compared)))


def test_the_pipeline_field_lists_are_the_header_blocks_in_header_order():
    """`PIPELINE_DIM_FIELDS` is the header's dim block (`struct_size` to `eps`)
    and `PIPELINE_BUFFER_FIELDS` its buffer block (`eps` to `q_o`), in the
    order the header declares them: `build_pipeline_config` copies both lists
    positionally onto the struct, so the order is the pipeline's dimensions."""
    names = [member.name
             for member in _header_structs(HEADER.read_text())["frt_imagewam_pipeline_config"]]
    dims = names[names.index(DIM_BLOCK[0]) + 1:names.index(DIM_BLOCK[1])]
    buffers = names[names.index(BUFFER_BLOCK[0]) + 1:names.index(BUFFER_BLOCK[1])]
    print(f"header dim block ({len(dims)}): {dims}")
    print(f"header buffer block ({len(buffers)}): {buffers}")
    assert tuple(dims) == PIPELINE_DIM_FIELDS, \
        _difference("PIPELINE_DIM_FIELDS", dims, PIPELINE_DIM_FIELDS)
    assert tuple(buffers) == PIPELINE_BUFFER_FIELDS, \
        _difference("PIPELINE_BUFFER_FIELDS", buffers, PIPELINE_BUFFER_FIELDS)


def test_the_pipeline_field_lists_are_the_dataclass_fields():
    """Every field of the dataclasses the lists are copied from is in its list
    and every listed name is a field of it, so no field is silently dropped
    from the handoff and no name is read from an attribute that does not
    exist. `eps` is the one field outside `PIPELINE_DIM_FIELDS`: it is a
    `float` in the header and `build_pipeline_config` sets it on its own
    (`c.eps = resources.dims.eps`), while the list's names are the `int32_t`
    dims copied with `int(getattr(...))`."""
    dims = tuple(field.name for field in dataclasses.fields(PipelineDims))
    buffers = tuple(field.name for field in dataclasses.fields(PipelineBuffers))
    print(f"PipelineDims ({len(dims)}): {dims}")
    print(f"PipelineBuffers ({len(buffers)}): {buffers}")
    assert set(dims) == set(PIPELINE_DIM_FIELDS) | {"eps"}, \
        (f"PipelineDims fields the handoff never sets: "
         f"{sorted(set(dims) - set(PIPELINE_DIM_FIELDS) - {'eps'})}; "
         f"PIPELINE_DIM_FIELDS names without a field: "
         f"{sorted(set(PIPELINE_DIM_FIELDS) - set(dims))}")
    assert set(buffers) == set(PIPELINE_BUFFER_FIELDS), \
        (f"PipelineBuffers fields the handoff never sets: "
         f"{sorted(set(buffers) - set(PIPELINE_BUFFER_FIELDS))}; "
         f"PIPELINE_BUFFER_FIELDS names without a field: "
         f"{sorted(set(PIPELINE_BUFFER_FIELDS) - set(buffers))}")
