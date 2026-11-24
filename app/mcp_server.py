"""MCP server exposing local system-level tools over stdio.

Run standalone for debugging:  python -m app.mcp_server
The agent spawns it as a subprocess and talks to it through an MCP client session.
"""

from __future__ import annotations

import ast
import math
import operator
import platform
import shutil
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from mcp.server.fastmcp import FastMCP

LOG_FILE = Path(__file__).resolve().parent.parent / "logs" / "agent.log"

mcp = FastMCP("kyte-system-tools")


# ------------------------------------------------------------------ time
@mcp.tool()
def get_current_time(timezone: str = "Asia/Kolkata") -> str:
    """Get the current date and time in an IANA timezone (e.g. 'Asia/Kolkata',
    'America/New_York', 'Europe/London', 'UTC')."""
    try:
        now = datetime.now(ZoneInfo(timezone))
    except (ZoneInfoNotFoundError, ValueError):
        return f"Error: unknown timezone '{timezone}'. Use an IANA name like 'Asia/Kolkata'."
    return now.strftime(f"%A, %d %B %Y, %I:%M:%S %p ({timezone}, UTC%z)")


# ------------------------------------------------------------ calculator
_BIN: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY: dict[type[ast.unaryop], Callable[[float], float]] = {ast.UAdd: operator.pos, ast.USub: operator.neg}
_FUNCS: dict[str, Callable[..., float]] = {
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
    "floor": math.floor,
    "ceil": math.ceil,
    "min": min,
    "max": max,
}
_CONSTS = {"pi": math.pi, "e": math.e}


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > 1000:
            raise ValueError("exponent too large")
        return _BIN[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _UNARY[type(node.op)](_eval(node.operand))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _FUNCS[node.func.id](*[_eval(a) for a in node.args])
    raise ValueError("unsupported expression")


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression precisely. Supports + - * / // % **,
    parentheses, constants pi/e and functions sqrt, sin, cos, tan, log, log10,
    exp, abs, round, floor, ceil, min, max. Example: '10000 * (1 + 0.052) ** 3'."""
    try:
        result = _eval(ast.parse(expression.strip().replace("^", "**"), mode="eval"))
    except ZeroDivisionError:
        return "Error: division by zero"
    except Exception as exc:  # noqa: BLE001 - narrowed in Phase 2
        return f"Error: could not evaluate '{expression}' ({exc})"
    if isinstance(result, float):
        result = round(result, 10)
    return f"{expression} = {result}"


# --------------------------------------------------------------- weather
_WMO = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "rime fog",
    51: "light drizzle",
    53: "drizzle",
    55: "heavy drizzle",
    61: "light rain",
    63: "rain",
    65: "heavy rain",
    71: "light snow",
    73: "snow",
    75: "heavy snow",
    80: "rain showers",
    81: "heavy rain showers",
    82: "violent rain showers",
    95: "thunderstorm",
    96: "thunderstorm with hail",
    99: "severe thunderstorm with hail",
}


@mcp.tool()
def get_weather(city: str) -> str:
    """Get the current real-time weather (temperature, wind, humidity, conditions)
    for a city, e.g. 'Mumbai' or 'London'."""
    try:
        with httpx.Client(timeout=10) as client:
            geo = client.get(
                "https://geocoding-api.open-meteo.com/v1/search",
                params={"name": city, "count": 1},
            ).json()
            if not geo.get("results"):
                return f"Error: could not find a city called '{city}'."
            place = geo["results"][0]
            w = client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "wind_speed_10m,weather_code",
                },
            ).json()["current"]
    except Exception as exc:  # noqa: BLE001 - narrowed in Phase 2
        return f"Error: weather service unreachable ({exc})"
    where = ", ".join(x for x in (place["name"], place.get("admin1"), place.get("country")) if x)
    return (
        f"Weather in {where}: {_WMO.get(w['weather_code'], 'unknown conditions')}, "
        f"{w['temperature_2m']}°C (feels like {w['apparent_temperature']}°C), "
        f"humidity {w['relative_humidity_2m']}%, wind {w['wind_speed_10m']} km/h."
    )


# ------------------------------------------------------------ system info
@mcp.tool()
def get_system_info() -> str:
    """Report details about the machine the agent runs on: OS, Python version,
    CPU count and free disk space."""
    import os

    disk = shutil.disk_usage(Path.home())
    return (
        f"OS: {platform.system()} {platform.release()} ({platform.machine()}); "
        f"Python {platform.python_version()}; CPU cores: {os.cpu_count()}; "
        f"disk free: {disk.free / 1e9:.1f} GB of {disk.total / 1e9:.1f} GB."
    )


@mcp.tool()
def read_system_logs(lines: int = 15, level: str = "") -> str:
    """Read the most recent lines of the agent's own application log. Optionally
    filter by level (INFO, WARNING, ERROR). Useful to check recent errors."""
    if not LOG_FILE.exists():
        return "No log file yet."
    rows = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()
    if level:
        rows = [r for r in rows if f" {level.upper()} " in r]
    rows = rows[-max(1, min(lines, 100)) :]
    return "\n".join(rows) if rows else "No matching log entries."


if __name__ == "__main__":
    mcp.run(transport="stdio")
