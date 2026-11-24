"""MCP server exposing customer-safe utility tools over stdio.

Run standalone for debugging:  python -m app.mcp_server
The agent spawns it as a subprocess and talks to it through an MCP client session.

Every tool here can be reached by an anonymous customer through the LLM, so each one
must be side-effect free, bounded in CPU/memory/time, and must not reveal anything
about the host or other users. Host diagnostics live behind the authenticated
/admin endpoints instead (see app/admin.py). Note that stdout is the MCP protocol
channel: diagnostics go to the logger (stderr), never print().
"""

from __future__ import annotations

import ast
import logging
import math
import operator
from collections.abc import Callable
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx
from mcp.server.fastmcp import FastMCP

log = logging.getLogger("mcp_server")

# Per-request INFO lines from the MCP framework go to stderr unstructured; keep only warnings.
mcp = FastMCP("voice-agent-tools", log_level="WARNING")


# ------------------------------------------------------------------ time
MAX_TIMEZONE_CHARS = 64


@mcp.tool()
def get_current_time(timezone: str = "Asia/Kolkata") -> str:
    """Get the current date and time in an IANA timezone (e.g. 'Asia/Kolkata',
    'America/New_York', 'Europe/London', 'UTC')."""
    timezone = timezone.strip()
    if not timezone or len(timezone) > MAX_TIMEZONE_CHARS:
        return "Error: give an IANA timezone name like 'Asia/Kolkata'."
    try:
        now = datetime.now(ZoneInfo(timezone))
    except (ZoneInfoNotFoundError, ValueError):
        return f"Error: unknown timezone '{timezone}'. Use an IANA name like 'Asia/Kolkata'."
    return now.strftime(f"%A, %d %B %Y, %I:%M:%S %p ({timezone}, UTC%z)")


# ------------------------------------------------------------ calculator
# Bounds that keep every evaluation fast and small. Without them an expression such as
# ((10**999)**999)**99 runs for minutes and exhausts memory in the shared MCP process.
MAX_EXPRESSION_CHARS = 200
MAX_AST_NODES = 100
MAX_INT_BITS = 4096  # ~1,233 decimal digits


class CalculationError(ValueError):
    """The expression is invalid or exceeds the calculator's bounds."""


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


def _bounded(value: float) -> float:
    if isinstance(value, int) and value.bit_length() > MAX_INT_BITS:
        raise CalculationError("result is too large")
    if isinstance(value, float) and not math.isfinite(value):
        raise CalculationError("result is out of range")
    return value


def _check_pow(base: float, exponent: float) -> None:
    if abs(exponent) > 1000:
        raise CalculationError("exponent too large")
    # For integers, estimate the result size before computing it.
    if (
        isinstance(base, int)
        and isinstance(exponent, int)
        and exponent > 0
        and max(base.bit_length(), 1) * exponent > MAX_INT_BITS
    ):
        raise CalculationError("result is too large")


def _eval(node: ast.AST) -> float:
    if isinstance(node, ast.Expression):
        return _eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return _bounded(node.value)
    if isinstance(node, ast.Name) and node.id in _CONSTS:
        return _CONSTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN:
        left, right = _eval(node.left), _eval(node.right)
        if isinstance(node.op, ast.Pow):
            _check_pow(left, right)
        return _bounded(_BIN[type(node.op)](left, right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY:
        return _bounded(_UNARY[type(node.op)](_eval(node.operand)))
    if (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id in _FUNCS
        and not node.keywords
    ):
        return _bounded(_FUNCS[node.func.id](*[_eval(a) for a in node.args]))
    raise CalculationError("unsupported expression")


def evaluate(expression: str) -> float:
    """Safely evaluate an arithmetic expression. Raises CalculationError / ZeroDivisionError."""
    expression = expression.strip().replace("^", "**")
    if not expression:
        raise CalculationError("empty expression")
    if len(expression) > MAX_EXPRESSION_CHARS:
        raise CalculationError(f"expression longer than {MAX_EXPRESSION_CHARS} characters")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise CalculationError("invalid syntax") from exc
    if sum(1 for _ in ast.walk(tree)) > MAX_AST_NODES:
        raise CalculationError("expression is too complex")
    try:
        return _eval(tree)
    except (OverflowError, RecursionError) as exc:
        raise CalculationError("result is out of range") from exc
    except (TypeError, ValueError) as exc:  # wrong arg count, math domain errors (sqrt(-1), log(0))
        if isinstance(exc, CalculationError):
            raise
        raise CalculationError("invalid arguments for a function") from exc


@mcp.tool()
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression precisely. Supports + - * / // % **,
    parentheses, constants pi/e and functions sqrt, sin, cos, tan, log, log10,
    exp, abs, round, floor, ceil, min, max. Example: '10000 * (1 + 0.052) ** 3'."""
    try:
        result = evaluate(expression)
    except ZeroDivisionError:
        return "Error: division by zero"
    except CalculationError as exc:
        return f"Error: could not evaluate the expression ({exc})"
    if isinstance(result, float):
        result = round(result, 10)
    return f"{expression.strip()} = {result}"


# --------------------------------------------------------------- weather
MAX_CITY_CHARS = 100
WEATHER_TIMEOUT_SECONDS = 8.0

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
    city = city.strip()
    if not city or len(city) > MAX_CITY_CHARS:
        return "Error: give a city name, e.g. 'Mumbai'."
    try:
        with httpx.Client(timeout=WEATHER_TIMEOUT_SECONDS) as client:
            geo_resp = client.get(
                "https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1}
            )
            geo_resp.raise_for_status()
            results = geo_resp.json().get("results")
            if not results:
                return f"Error: could not find a city called '{city}'."
            place = results[0]
            w_resp = client.get(
                "https://api.open-meteo.com/v1/forecast",
                params={
                    "latitude": place["latitude"],
                    "longitude": place["longitude"],
                    "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "wind_speed_10m,weather_code",
                },
            )
            w_resp.raise_for_status()
            w = w_resp.json()["current"]
        where = ", ".join(x for x in (place["name"], place.get("admin1"), place.get("country")) if x)
        return (
            f"Weather in {where}: {_WMO.get(w['weather_code'], 'unknown conditions')}, "
            f"{w['temperature_2m']}°C (feels like {w['apparent_temperature']}°C), "
            f"humidity {w['relative_humidity_2m']}%, wind {w['wind_speed_10m']} km/h."
        )
    except httpx.HTTPError as exc:
        log.warning("weather service request failed: %s", type(exc).__name__)
        return "Error: the weather service is unavailable right now."
    except (KeyError, TypeError, ValueError) as exc:  # unexpected / malformed payload
        log.warning("weather service returned an unexpected payload: %s", type(exc).__name__)
        return "Error: the weather service returned an unexpected response."


if __name__ == "__main__":
    mcp.run(transport="stdio")
