"""Reading a datacard configuration: one card built on another, with variables.

Every reader of a datacard configuration (dc_make, the law tasks, an analysis's preprocess
script) goes through `load_card_config`, so a card means the same thing everywhere.

    extends: other.yaml       # optional: the card this one builds on
    variables:                # optional: ${NAME} is replaced by the value in every string
      SIGNAL_MODEL: GluGluToRadion
    ...                       # the card itself

- `extends` names another configuration, relative to this file's directory. It is read first
  (it may extend another one in turn), and this file's keys are merged over it: a mapping is
  merged key by key, anything else (a list, a string, a number) replaces the base's value, and
  a key set to null is removed.
- `variables` are merged like any other mapping (a card can override a variable of its base),
  then `${NAME}` is substituted in every string of the merged card, keys included. A
  placeholder that is not a variable (`${MX}`, `${ERA}`, a CombineHarvester `$BIN`) is left as
  it is, for the tools that fill it later. A variable whose value is not a string or a number
  is an error, as is a cycle of `extends`.
- Neither `extends` nor `variables` is part of the returned card.
"""

import os
from string import Template

import yaml

_RESERVED = ("extends", "variables")


def _merge(base, override):
    out = dict(base)
    for key, value in override.items():
        if value is None:
            out.pop(key, None)
        elif isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


def _read(path, chain):
    path = os.path.abspath(path)
    if path in chain:
        raise ValueError(
            "datacard configuration: extends cycle " + " -> ".join(chain + [path])
        )
    with open(path, "r") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"datacard configuration {path}: not a mapping")
    base = cfg.get("extends")
    if base is None:
        return cfg
    if not isinstance(base, str):
        raise ValueError(f"datacard configuration {path}: extends must be a file name")
    parent = _read(os.path.join(os.path.dirname(path), base), chain + [path])
    return _merge(parent, cfg)


def _substitute(obj, variables):
    if isinstance(obj, str):
        return Template(obj).safe_substitute(variables) if "$" in obj else obj
    if isinstance(obj, dict):
        return {
            _substitute(k, variables): _substitute(v, variables) for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_substitute(v, variables) for v in obj]
    return obj


def load_card_config(path):
    """The datacard configuration at `path`, with `extends` merged and `variables` substituted."""
    cfg = _read(path, [])
    variables = cfg.get("variables") or {}
    if not isinstance(variables, dict):
        raise ValueError(f"datacard configuration {path}: variables must be a mapping")
    for name, value in variables.items():
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(
                f"datacard configuration {path}: variable {name} must be a string or a number"
            )
    variables = {name: str(value) for name, value in variables.items()}
    cfg = {k: v for k, v in cfg.items() if k not in _RESERVED}
    return _substitute(cfg, variables) if variables else cfg
