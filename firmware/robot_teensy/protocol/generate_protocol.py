#!/usr/bin/env python3
"""Generate protocol/parameter artifacts from protocol/schema.json.

`--bootstrap` is a one-time migration helper which imports the legacy C++ and
Python mirrors into the schema. Normal development uses no flag to regenerate
or `--check` in CI to reject stale generated files.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
ROBOT = ROOT / "firmware" / "robot_teensy"
SCHEMA_PATH = Path(__file__).with_name("schema.json")


def _defines(path: Path, prefix: str) -> list[dict]:
    out = []
    pattern = re.compile(rf"^#define\s+({re.escape(prefix)}[A-Z0-9_]+)\s+(0x[0-9A-Fa-f]+|\d+)(?:\s*//\s*(.*))?$")
    for line in path.read_text(encoding="utf-8").splitlines():
        match = pattern.match(line)
        if match:
            out.append({"symbol": match[1], "id": int(match[2], 0),
                        "description": (match[3] or "").strip()})
    return out


def _python_param_descriptions(path: Path) -> dict[int, tuple[str, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        target = None
        value = None
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            target, value = node.target.id, node.value
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            target, value = node.targets[0].id, node.value
        if target == "_PARAM_DEFS":
            try:
                return ast.literal_eval(value)
            except ValueError:
                # The legacy mirror uses a few named description suffixes.
                # Import is acceptable only for this one-time bootstrap path;
                # generated/check operation has no PyQt dependency.
                sys.path.insert(0, str(path.parents[1]))
                from tabs.params_tab import _PARAM_DEFS  # type: ignore
                return _PARAM_DEFS
    raise RuntimeError("_PARAM_DEFS not found")


def bootstrap() -> dict:
    param_ids_path = ROBOT / "teensy/lib/ParamRegistry/param_ids.h"
    registry_path = ROBOT / "teensy/lib/ParamRegistry/param_registry.cpp"
    gui_path = ROOT / "software/gui/tabs/params_tab.py"
    symbols = {item["symbol"]: item["id"] for item in _defines(param_ids_path, "PARAM_")}
    groups = {item["symbol"]: item["id"] for item in _defines(param_ids_path, "GROUP_")}
    descriptions = _python_param_descriptions(gui_path)
    row_re = re.compile(
        r"\{\.id = (PARAM_[A-Z0-9_]+),\s*\.group_id = (GROUP_[A-Z0-9_]+),\s*"
        r"\.name = \"([^\"]+)\",\s*\.value = ([^,]+),\s*\.min_val = ([^,]+),\s*"
        r"\.max_val = ([^,]+),\s*\.flags = ([^,]+),\s*\.on_change = nullptr\},"
    )

    def number(text: str) -> float:
        return float(text.strip().rstrip("fF"))

    def flags(text: str) -> list[str]:
        mapping = {
            "PARAM_FLAG_PERSISTENT": "persistent",
            "PARAM_FLAG_READONLY": "readonly",
            "PARAM_FLAG_COMMAND": "command",
        }
        return [mapping[token] for token in text.replace(" ", "").split("|") if token in mapping]

    params = []
    for match in row_re.finditer(registry_path.read_text(encoding="utf-8")):
        symbol, group_symbol, name, default, minimum, maximum, flag_expr = match.groups()
        pid = symbols[symbol]
        gui_name, description = descriptions.get(pid, (name, ""))
        if gui_name != name:
            raise RuntimeError(f"parameter name mismatch for {symbol}: {name} != {gui_name}")
        params.append({
            "symbol": symbol, "id": pid, "group": group_symbol, "name": name,
            "default": number(default), "min": number(minimum), "max": number(maximum),
            "flags": flags(flag_expr), "description": description,
        })
    if len(params) != len(descriptions):
        raise RuntimeError(f"parameter import incomplete: C++={len(params)} GUI={len(descriptions)}")

    states = []
    state_text = (ROBOT / "teensy/src/robot_state.h").read_text(encoding="utf-8")
    for symbol, value in re.findall(r"^\s*(STATE_[A-Z0-9_]+)\s*=\s*(\d+)", state_text, re.M):
        states.append({"symbol": symbol, "id": int(value), "name": symbol.removeprefix("STATE_")})

    comm = ROBOT / "shared/comm_protocol.h"
    return {
        "schema_version": 1,
        "states": states,
        "faults": _defines(comm, "FAULT_"),
        "commands": _defines(comm, "CMD_ID_"),
        "groups": [{"symbol": key, "id": value} for key, value in groups.items()],
        "parameters": params,
    }


def _hex(value: int, width: int = 2) -> str:
    return f"0x{value:0{width}X}"


def _cpp_float(value: float) -> str:
    text = f"{value:.9g}"
    if "." not in text and "e" not in text.lower():
        text += ".0"
    return text + "f"


def render(schema: dict) -> dict[Path, str]:
    for section in ("states", "faults", "commands", "groups", "parameters"):
        items = schema[section]
        ids = [item["id"] for item in items]
        symbols = [item["symbol"] for item in items]
        if len(ids) != len(set(ids)):
            raise ValueError(f"duplicate numeric ID in {section}")
        if len(symbols) != len(set(symbols)):
            raise ValueError(f"duplicate symbol in {section}")
    names = [item["name"] for item in schema["parameters"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate parameter name")
    if any(len(name.encode("ascii")) > 19 for name in names):
        raise ValueError("parameter names must fit Param.name[20]")

    banner = "// Generated by protocol/generate_protocol.py from protocol/schema.json. DO NOT EDIT.\n"
    ids = ["#pragma once", "#include <stdint.h>", ""]
    for section in ("faults", "commands"):
        ids.append(f"// {section.title()}")
        for item in schema[section]:
            ids.append(f"#define {item['symbol']:<36} {_hex(item['id'])}")
        ids.append("")

    state = ["#pragma once", "#include <stdint.h>", "", "typedef enum : uint8_t {"]
    state += [f"    {item['symbol']:<24} = {item['id']}," for item in schema["states"]]
    state += ["} RobotStateEnum;", ""]

    param_ids = ["#pragma once", "", "// Parameter groups"]
    param_ids += [f"#define {item['symbol']:<24} {_hex(item['id'])}" for item in schema["groups"]]
    param_ids += ["", "// Parameters"]
    param_ids += [f"#define {item['symbol']:<42} {_hex(item['id'], 4)}" for item in schema["parameters"]]
    param_ids.append("")

    flag_expr = {
        "persistent": "PARAM_FLAG_PERSISTENT",
        "readonly": "PARAM_FLAG_READONLY",
        "command": "PARAM_FLAG_COMMAND",
    }
    table = ["// Generated Param rows; included inside g_params[]."]
    for p in schema["parameters"]:
        flags = "|".join(flag_expr[f] for f in p["flags"]) or "0"
        table.append(
            f'{{.id = {p["symbol"]}, .group_id = {p["group"]}, .name = "{p["name"]}", '
            f'.value = {_cpp_float(p["default"])}, .min_val = {_cpp_float(p["min"])}, '
            f'.max_val = {_cpp_float(p["max"])}, .flags = {flags}, .on_change = nullptr}},'
        )
    table.append("")

    py = ["# Generated by firmware/robot_teensy/protocol/generate_protocol.py. DO NOT EDIT."]
    for section in ("states", "faults", "commands"):
        py += [f"{p['symbol']} = {p['id']}" for p in schema[section]]
    py += [f"{p['symbol']} = {p['id']}" for p in schema["parameters"]]
    py.append("STATE_NAMES = " + repr({p["id"]: p["name"] for p in schema["states"]}))
    py.append("FAULT_NAMES = " + repr({p["id"]: p["symbol"].removeprefix("FAULT_") for p in schema["faults"]}))
    py.append("FAULT_DESCRIPTIONS = " + repr({p["id"]: p["description"] for p in schema["faults"]}))
    py.append("COMMAND_IDS = " + repr({p["symbol"]: p["id"] for p in schema["commands"]}))
    py.append("PARAM_IDS = " + repr({p["symbol"]: p["id"] for p in schema["parameters"]}))
    py.append("PARAM_DEFS = " + repr({p["id"]: (p["name"], p["description"]) for p in schema["parameters"]}))
    py.append("PARAM_BY_NAME = " + repr({p["name"]: p["id"] for p in schema["parameters"]}))
    py.append("")

    md = ["# Generated robot protocol reference", "", "Do not edit; generated from `protocol/schema.json`.", ""]
    for title, section in (("States", "states"), ("Faults", "faults"), ("Commands", "commands")):
        md += [f"## {title}", "", "| ID | Symbol | Description |", "| ---: | --- | --- |"]
        md += [f"| `{_hex(p['id'])}` | `{p['symbol']}` | {p.get('description', '')} |" for p in schema[section]]
        md.append("")
    md += ["## Parameters", "", "| ID | Symbol | Name | Default | Range | Flags |", "| ---: | --- | --- | ---: | --- | --- |"]
    md += [f"| `{_hex(p['id'],4)}` | `{p['symbol']}` | `{p['name']}` | {p['default']:.9g} | {p['min']:.9g} … {p['max']:.9g} | {', '.join(p['flags']) or '-'} |" for p in schema["parameters"]]
    md.append("")

    vectors = {
        "schema_version": schema["schema_version"],
        "states": {p["symbol"]: p["id"] for p in schema["states"]},
        "faults": {p["symbol"]: p["id"] for p in schema["faults"]},
        "commands": {p["symbol"]: p["id"] for p in schema["commands"]},
        "parameters": {p["symbol"]: p["id"] for p in schema["parameters"]},
    }
    return {
        ROBOT / "shared/generated_protocol_ids.h": banner + "\n".join(ids),
        ROBOT / "teensy/src/generated_robot_state.h": banner + "\n".join(state),
        ROBOT / "teensy/lib/ParamRegistry/generated_param_ids.h": banner + "\n".join(param_ids),
        ROBOT / "teensy/lib/ParamRegistry/generated_param_table.inc": banner + "\n".join(table),
        ROOT / "software/gui/tabs/generated_protocol.py": "\n".join(py),
        ROBOT / "protocol/protocol.generated.md": "\n".join(md),
        ROBOT / "protocol/generated_vectors.json": json.dumps(vectors, indent=2, sort_keys=True) + "\n",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap", action="store_true")
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.bootstrap:
        if SCHEMA_PATH.exists():
            raise SystemExit("schema.json already exists; refusing to overwrite")
        SCHEMA_PATH.write_text(json.dumps(bootstrap(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
    stale = []
    for path, content in render(schema).items():
        if args.check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                stale.append(str(path.relative_to(ROOT)))
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8", newline="\n")
    if stale:
        print("stale generated artifacts:\n  " + "\n  ".join(stale), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
