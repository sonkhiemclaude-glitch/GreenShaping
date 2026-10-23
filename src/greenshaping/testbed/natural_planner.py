"""Natural-language one-round planning, with fail-closed tool validation.

The prompt never includes reference plans. This is a bounded planner, not an
autonomous adaptive agent. Reference equivalence is checked separately by the caller.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any

from ..data.schemas import QUERY_FAMILIES

COLUMNS = {"elevation", "aspect", "slope", "wilderness", "cover_type", "elevation_scaled", "slope_scaled", "lodgepole"}


def planner_schema(n_sites: int) -> dict:
    def obj(properties):
        return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}

    number = {"type": "number"}
    columns = {"type": "array", "minItems": 1, "items": {"type": "string", "enum": sorted(COLUMNS)}}
    where = obj({"aspect": obj({"min": number, "max": number})})
    fields: dict[str, dict[str, Any]] = {"record_count": {}, "aggregate": {"columns": columns},
              "group_by": {"by": {"type": "string", "enum": sorted(COLUMNS)},
                           "column": {"type": "string", "enum": sorted(COLUMNS)}},
              "train_round": {"features": columns, "label": {"type": "string", "enum": ["lodgepole"]},
                              "weights": {"type": "array", "minItems": 1, "items": number}, "lr": number}}
    return {"oneOf": [obj({"family": {"type": "string", "enum": [family]},
                           "calls": {"type": "array", "minItems": 1, "maxItems": n_sites,
                                     "items": obj({"site": {"type": "integer", "minimum": 0, "maximum": n_sites - 1},
                                                   "params": obj({"where": where, **params})})}})
                      for family, params in fields.items()]}


def validate_plan(value: Any, n_sites: int) -> dict:
    if not isinstance(value, dict) or set(value) != {"family", "calls"} or value["family"] not in QUERY_FAMILIES:
        raise ValueError("expected family and calls")
    family, calls = value["family"], value["calls"]
    if not isinstance(calls, list) or not calls or len(calls) > n_sites:
        raise ValueError("expected a nonempty, bounded call list")
    seen = set()
    for c in calls:
        if not isinstance(c, dict) or set(c) != {"site", "params"}:
            raise ValueError("invalid call fields")
        site, p = c["site"], c["params"]
        if type(site) is not int or not 0 <= site < n_sites or site in seen:
            raise ValueError("invalid or duplicate site")
        seen.add(site)
        required = {"record_count": {"where"}, "aggregate": {"where", "columns"},
                    "group_by": {"where", "by", "column"},
                    "train_round": {"where", "features", "label", "weights", "lr"}}[family]
        if not isinstance(p, dict) or set(p) != required:
            raise ValueError("invalid tool argument fields")
        if not isinstance(p["where"], dict) or set(p["where"]) != {"aspect"}:
            raise ValueError("this benchmark permits an aspect-range predicate")
        bounds = p["where"]["aspect"]
        if not isinstance(bounds, dict) or set(bounds) != {"min", "max"}:
            raise ValueError("invalid predicate")
        nums = [bounds["min"], bounds["max"]]
        if any(type(x) not in (int, float) or not math.isfinite(x) for x in nums) or not 0 <= nums[0] <= nums[1] <= 360:
            raise ValueError("aspect outside valid inclusive range")
        for name in ("by", "column", "label"):
            if name in p and (not isinstance(p[name], str) or p[name] not in COLUMNS):
                raise ValueError("unknown column")
        for name in ("columns", "features"):
            if name in p and (not isinstance(p[name], list) or not p[name] or
                              any(not isinstance(x, str) or x not in COLUMNS for x in p[name]) or
                              len(p[name]) != len(set(p[name]))):
                raise ValueError("invalid columns")
        if family == "train_round":
            if p["label"] != "lodgepole" or not isinstance(p["weights"], list) or len(p["weights"]) != len(p["features"]) + 1:
                raise ValueError("invalid logistic model")
            if any(type(x) not in (int, float) or not math.isfinite(x) for x in [*p["weights"], p["lr"]]):
                raise ValueError("nonfinite training parameters")
            if not 0 < p["lr"] <= 1:
                raise ValueError("invalid learning rate")
    return {"family": family, "calls": sorted(calls, key=lambda c: c["site"])}


def validate_requested_sites(plan: dict, analyst_text: str) -> None:
    """Public allowlist for this explicit-site benchmark; never supplies tool arguments."""
    requested = {int(x) for x in re.findall(r"\bsite\s+(\d+)\b", analyst_text, flags=re.IGNORECASE)}
    if not requested or {c["site"] for c in plan["calls"]} != requested:
        raise ValueError("calls must include exactly the site IDs explicitly named in the analyst text")


def planner_prompt(text: str, n_sites: int) -> str:
    catalog = {
        "record_count": {"where": {"aspect": {"min": "number", "max": "number"}}},
        "aggregate": {"where": "same predicate", "columns": ["elevation"]},
        "group_by": {"where": "same predicate", "by": "wilderness", "column": "elevation"},
        "train_round": {"where": "same predicate", "features": ["elevation_scaled", "slope_scaled"],
                        "label": "lodgepole", "weights": [0., 0., 0.], "lr": .1},
    }
    return (f"Translate the analyst question into ONE round of tool calls at sites 0 through {n_sites - 1}. "
            "Return only JSON with exactly keys family and calls. calls is a list of {site: integer, params: object}. "
            "Use only explicitly requested sites, once each. All predicates are inclusive. "
            "Family MUST be aggregate for count+sum+sum of squares; group_by for grouped count+sum; "
            "train_round for a logistic gradient step; record_count ONLY for a count alone. "
            "Every params object must have where with the complete aspect min/max predicate. "
            "Do not execute tools or invent an answer. Tool catalog: " + json.dumps(catalog) +
            "\nOutput schema: " + json.dumps(planner_schema(n_sites)) + "\nAnalyst: " + text)
