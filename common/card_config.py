"""Reading datacard configurations: one card, the files it includes, and the governor.

A *card* is one datacard configuration -- the file dc_make/create_datacards.py builds
datacards from. It may list shared files under ``include:``; they are merged into it by
the strict rules below, which never let one file override another. Anything set twice is
an error naming both files, so a duplicated or conflicting setting cannot hide behind an
include.

A *governor* names the cards an analysis builds, each under the name the law tasks use
for it in their branches and output paths::

    cards:
      DL_spin0: DL/dl_spin0.yaml
      DL_spin2: DL/dl_spin2.yaml

Merge rules (includes first, in their order, then the card itself):

* A top-level list of named entries (NAMED_LISTS: ``processes`` by their ``process``,
  a bare string entry by itself; ``uncertainties``, ``limit_plots`` and ``impact_plots``
  by their ``name``) is joined across the files. An entry name seen twice raises --
  also within one file, and also when both entries are identical.
* Any other top-level key may be set in one file only; set in two, it raises, whatever
  the values. Nothing is merged below the top level: a mapping such as ``model:`` is one
  setting, defined whole in one file.
* ``include:`` paths are relative to the directory of the file that includes them.
  Included files cannot include further files, cannot be included twice, and cannot be
  the card itself.

A card without ``include:`` reads exactly as yaml.safe_load() reads it, so every existing
configuration keeps working; the merged result has the same layout as such a card, so
whatever reads a plain card (dc_make, a preprocess script) reads a merged one unchanged.
"""

import os
import re

import yaml

#: top-level lists joined across files, and the key that names each entry
NAMED_LISTS = {
    "processes": "process",
    "uncertainties": "name",
    "limit_plots": "name",
    "impact_plots": "name",
}

INCLUDE_KEY = "include"
GOVERNOR_KEY = "cards"

#: a card name becomes a directory in the output paths
_CARD_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


class CardConfigError(ValueError):
    """A datacard configuration that cannot be read without ambiguity."""


def _read_yaml(path):
    if not os.path.isfile(path):
        raise CardConfigError(f"{path}: no such file")
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise CardConfigError(f"{path}: expected a mapping at the top level")
    return data


def _entry_name(key, entry, path, required):
    """The name of one entry of the named list `key`, or None when it has none and
    `required` is false."""
    if key == "processes" and isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and entry.get(NAMED_LISTS[key]) is not None:
        return str(entry[NAMED_LISTS[key]])
    if required:
        raise CardConfigError(
            f"{path}: an entry of `{key}` has no `{NAMED_LISTS[key]}`; in a card that "
            f"uses `{INCLUDE_KEY}:` every entry must be named, so that a duplicate can be "
            f"recognised: {entry!r}"
        )
    return None


def _include_paths(card_path, card):
    includes = card.get(INCLUDE_KEY)
    if includes is None:
        return []
    if isinstance(includes, str):
        includes = [includes]
    if not isinstance(includes, list) or not all(isinstance(i, str) for i in includes):
        raise CardConfigError(
            f"{card_path}: `{INCLUDE_KEY}:` must be a file name or a list of file names"
        )
    base = os.path.dirname(os.path.abspath(card_path))
    paths, seen = [], {os.path.realpath(card_path): card_path}
    for include in includes:
        path = os.path.normpath(os.path.join(base, include))
        real = os.path.realpath(path)
        if real in seen:
            raise CardConfigError(
                f"{card_path}: `{INCLUDE_KEY}:` lists {include}, which is "
                + (
                    "the card itself"
                    if seen[real] == card_path
                    else f"already included as {seen[real]}"
                )
            )
        seen[real] = path
        paths.append(path)
    return paths


def load_card_config_with_sources(path):
    """(merged card, sources) for the card at `path`.

    `sources` maps each top-level key to the file that sets it, and each named entry,
    as (key, name), to the file that defines it.
    """
    path = os.path.abspath(path)
    card = _read_yaml(path)
    if GOVERNOR_KEY in card:
        raise CardConfigError(
            f"{path}: `{GOVERNOR_KEY}:` belongs to a governor, which lists cards; a card "
            "cannot also be one"
        )
    include_paths = _include_paths(path, card)
    uses_includes = bool(include_paths)

    files = []
    for include in include_paths:
        data = _read_yaml(include)
        for forbidden in (INCLUDE_KEY, GOVERNOR_KEY):
            if forbidden in data:
                raise CardConfigError(
                    f"{include} (included by {path}) sets `{forbidden}:`; included files "
                    + (
                        "cannot include further files"
                        if forbidden == INCLUDE_KEY
                        else "cannot be governors"
                    )
                )
        files.append((include, data))
    files.append((path, {k: v for k, v in card.items() if k != INCLUDE_KEY}))

    merged, sources = {}, {}
    for file_path, data in files:
        for key, value in data.items():
            if key in NAMED_LISTS:
                if not isinstance(value, list):
                    raise CardConfigError(
                        f"{file_path}: `{key}:` must be a list of named entries"
                    )
                merged.setdefault(key, [])
                sources.setdefault(key, file_path)
                for entry in value:
                    name = _entry_name(key, entry, file_path, uses_includes)
                    if name is not None:
                        if (key, name) in sources:
                            first = sources[(key, name)]
                            where = (
                                f"twice in {file_path}"
                                if first == file_path
                                else f"in {first} and in {file_path}"
                            )
                            raise CardConfigError(
                                f"`{key}` entry '{name}' is defined {where}. Every entry "
                                "is defined once: keep it in one file, or, if it differs "
                                "between cards, in each card and not in a shared file."
                            )
                        sources[(key, name)] = file_path
                    merged[key].append(entry)
                continue
            if key in sources:
                raise CardConfigError(
                    f"`{key}:` is set in {sources[key]} and in {file_path}. A setting is "
                    "made in one file only, with no override: remove one of them."
                )
            sources[key] = file_path
            merged[key] = value
    return merged, sources


def load_card_config(path):
    """The card at `path` with its includes merged in; see the module docstring."""
    return load_card_config_with_sources(path)[0]


def dump_card_config(path, out_path, relative_to=None):
    """Write the merged card at `path` to `out_path`, as a plain card (no includes) that
    names the files it was merged from, relative to `relative_to` if given; returns the
    merged card."""
    merged, sources = load_card_config_with_sources(path)

    def shown(file_path):
        return os.path.relpath(file_path, relative_to) if relative_to else file_path

    files = list(dict.fromkeys(sources.values()))
    with open(out_path, "w") as f:
        f.write(f"# The datacard configuration {shown(os.path.abspath(path))}\n")
        if len(files) > 1:
            f.write("# merged with the files it includes, in this order:\n")
            for file_path in files:
                f.write(f"#   {shown(file_path)}\n")
        yaml.safe_dump(merged, f, sort_keys=False, default_flow_style=None, width=120)
    return merged


def load_governor(path):
    """{card name: absolute path of its card} from the governor at `path`, in its order.

    Card paths are relative to the governor's directory.
    """
    path = os.path.abspath(path)
    data = _read_yaml(path)
    unknown = sorted(k for k in data if k != GOVERNOR_KEY)
    if unknown:
        raise CardConfigError(
            f"{path}: a governor only lists `{GOVERNOR_KEY}:`; it does not take "
            f"{', '.join(unknown)}"
            + (
                " (is this a card? point StatInference's `config:` at it instead)"
                if any(k in data for k in ("model", "processes", "eras"))
                else ""
            )
        )
    cards = data.get(GOVERNOR_KEY)
    if not isinstance(cards, dict) or not cards:
        raise CardConfigError(
            f"{path}: `{GOVERNOR_KEY}:` must map each card name to its file"
        )
    base = os.path.dirname(path)
    result, seen = {}, {}
    for name, card in cards.items():
        name = str(name)
        if not _CARD_NAME.match(name):
            raise CardConfigError(
                f"{path}: card name '{name}' is used as a directory name; use letters, "
                "digits, '_', '-' and '.' only"
            )
        if not isinstance(card, str):
            raise CardConfigError(f"{path}: card '{name}' must name one file")
        card_path = os.path.normpath(os.path.join(base, card))
        real = os.path.realpath(card_path)
        if real in seen:
            raise CardConfigError(
                f"{path}: cards '{seen[real]}' and '{name}' are the same file {card}"
            )
        if not os.path.isfile(card_path):
            raise CardConfigError(f"{path}: card '{name}': no such file {card_path}")
        seen[real] = name
        result[name] = card_path
    return result


def card_name_of(path):
    """The name a lone card goes by when no governor names it: its file name without
    the extension."""
    return os.path.splitext(os.path.basename(path))[0]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Print a datacard configuration with its includes merged in."
    )
    parser.add_argument("card", help="the card file")
    parser.add_argument("--output", help="write it here instead of printing it")
    args = parser.parse_args()
    if args.output:
        dump_card_config(args.card, args.output)
    else:
        merged, sources = load_card_config_with_sources(args.card)
        for key, value in merged.items():
            if key in NAMED_LISTS:
                print(f"{key}:")
                for entry in value:
                    name = _entry_name(key, entry, args.card, False)
                    origin = sources.get((key, name), sources[key])
                    print(f"  {name}  <- {origin}")
            else:
                print(f"{key}  <- {sources[key]}")
