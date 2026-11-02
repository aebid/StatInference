"""Reading a datacard configuration, and its spin axis.

Every reader of a datacard configuration (dc_make, the law tasks, an analysis's preprocess
script) goes through `load_card_config`, so a card means the same thing everywhere.

A configuration may declare the signal hypotheses it is built for as a `spins:` axis, each
spin naming what differs between them (typically only the signal processes):

    spins:
      0: { SIGNAL_MODEL: GluGluToRadion, SPIN_NAME: radion }
      2: { SIGNAL_MODEL: GluGluToBulkGraviton, SPIN_NAME: graviton }
    processes:
      - process: ${SIGNAL_MODEL}_bbWW_2L_${MX}
      ...

The card of a spin is the configuration with that spin's values substituted for `${NAME}` in
every string, keys included, plus `${SPIN}`, the spin itself. Placeholders that are not spin
values (`${MX}`, `${ERA}`, CombineHarvester's `$BIN`) are left for the tools that fill them
later. The chain builds one card per era and spin; a configuration without `spins:` is one
card per era, as before, and is read unchanged.
"""

from string import Template

import yaml


def _read(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"datacard configuration {path}: not a mapping")
    return cfg


def _spin_table(cfg, path):
    """{spin (str): {name: value (str)}} of a configuration's `spins:`, validated, or None."""
    spins = cfg.get("spins")
    if spins is None:
        return None
    if not isinstance(spins, dict) or not spins:
        raise ValueError(
            f"datacard configuration {path}: spins must be a non-empty mapping "
            "{spin: {NAME: value}}"
        )
    table = {}
    for spin, values in spins.items():
        values = values or {}
        if not isinstance(values, dict):
            raise ValueError(
                f"datacard configuration {path}: the values of spin {spin} must be a mapping"
            )
        for name, value in values.items():
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                raise ValueError(
                    f"datacard configuration {path}: spin {spin}: {name} must be a string "
                    "or a number"
                )
        if "SPIN" in values:
            raise ValueError(
                f"datacard configuration {path}: spin {spin}: SPIN is the spin itself and "
                "is not set in its values"
            )
        table[str(spin)] = {name: str(value) for name, value in values.items()}
    return table


def _substitute(obj, values):
    if isinstance(obj, str):
        return Template(obj).safe_substitute(values) if "$" in obj else obj
    if isinstance(obj, dict):
        return {_substitute(k, values): _substitute(v, values) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute(v, values) for v in obj]
    return obj


def card_spins(cfg, path="<configuration>"):
    """The spins of a configuration as strings, in its order, or [None] without `spins:`."""
    table = _spin_table(cfg, path)
    return [None] if table is None else list(table)


def load_card_config(path, spin=None):
    """The datacard configuration at `path`.

    With `spin`, the card of that spin: its values substituted and `spins:` removed. Without,
    the configuration as written, `spins:` included -- what is common to every spin (eras,
    channels, categories, model, masses) can be read from it, the signal processes not.
    """
    cfg = _read(path)
    table = _spin_table(cfg, path)
    if spin is None:
        return cfg
    if table is None:
        raise ValueError(
            f"datacard configuration {path} declares no spins, so there is no spin {spin}"
        )
    if str(spin) not in table:
        raise ValueError(
            f"datacard configuration {path}: no spin {spin}; it declares "
            + ", ".join(table)
        )
    values = dict(table[str(spin)], SPIN=str(spin))
    return _substitute({k: v for k, v in cfg.items() if k != "spins"}, values)
