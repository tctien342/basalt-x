"""Bounded, request-local tools for Flash-Next thinking."""
from __future__ import annotations

import ast
import datetime as dt
import itertools
import json
import math
import operator
import re
from decimal import Decimal, DecimalException
from fractions import Fraction
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import regex

OPEN, CLOSE = "<mini_batch>", "</mini_batch>"
MAX_BODY = 16384
MAX_BATCHES = 4
MAX_ITEMS = 8
TOOL_NAME = "engine_mini_batch"
TOOL = {
    "name": TOOL_NAME,
    "description": "Server-internal helpers. Verify every calculated quantity you report with calculate, including incidental figures. "
                   "Batch up to eight ordered calls. Earlier successful calculator IDs can be used as variables.",
    "parameters": {
        "type": "object",
        "properties": {"calls": {
            "type": "array", "minItems": 1, "maxItems": MAX_ITEMS,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "tool": {"type": "string", "enum": ["calculate", "datetime", "regex"]},
                    "expr": {"type": "string", "description": "Exact arithmetic: + - * / % **, parentheses, "
                             "and gcd(a,b), lcm(a,b), comb(n,k), perm(n,k). Use comb(n,k) for binomial coefficients."},
                    "operation": {"type": "string", "enum": ["now", "convert", "add_days", "difference", "find", "count"]},
                    **{key: {"type": "string"} for key in
                       ("timezone", "value", "start", "end", "pattern", "text", "flags")},
                    "days": {"type": "integer"},
                },
                "required": ["id", "tool"], "additionalProperties": False,
            },
        }},
        "required": ["calls"], "additionalProperties": False,
    },
}
INSTRUCTIONS = (
    'During thinking, use engine_mini_batch in the normal tool-call format to verify every quantity you calculate '
    'and report to the user, including incidental totals, estimates and unit conversions. '
    'Treat mental arithmetic as unverified until a successful calculator result. '
    'Do not use bash or Python as a substitute. Emit the call on its own line and wait for its result. '
    'For example, pass calls=[{"id":"a","tool":"calculate","expr":"3**2"},'
    '{"id":"b","tool":"calculate","expr":"a*12312/12"}]. '
    'Datetime: now takes optional IANA timezone (default UTC); convert takes value (ISO timestamp with offset) '
    'and timezone; add_days takes value (ISO date) and integer days; difference takes start and end. '
    'Use two ISO dates for days or two ISO timestamps for elapsed seconds. Timestamps without an offset '
    'use timezone (default UTC). '
    'For regex use pattern, optional text and optional ims flags. '
    'If a batch reports an error, correct the expression and retry, or state the uncertainty in the final answer. '
    'After a successful result, continue the answer or make the requested client tool call.'
)
_IDENTIFIER = re.compile(r"[A-Za-z_][A-Za-z_0-9]{0,31}")
_NUMBER = re.compile(r"(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?")
_BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
           ast.Div: operator.truediv, ast.Mod: operator.mod}
_FUNCTIONS = {"gcd": math.gcd, "lcm": math.lcm, "comb": math.comb, "perm": math.perm}
MAX_BITS = 4096


def calculate(expr: str, values: dict[str, Fraction]) -> Fraction:
    if not isinstance(expr, str) or not 1 <= len(expr) <= 2048:
        raise ValueError("expression must have 1–2048 characters")
    expr = expr.strip()
    tree = ast.parse(expr, mode="eval")
    if sum(1 for _ in ast.walk(tree)) > 256:
        raise ValueError("expression exceeds 256 syntax nodes")

    def visit(node, depth=0):
        if depth > 64:
            raise ValueError("expression exceeds depth 64")
        def child(item):
            return visit(item, depth + 1)
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            literal = ast.get_source_segment(expr, node)
            if literal is None or not _NUMBER.fullmatch(literal):
                raise ValueError("expected a decimal numeric literal")
            decimal = Decimal(literal)
            if abs(decimal.as_tuple().exponent) > 1024 or abs(decimal.adjusted()) > 1024:
                raise ValueError("decimal literal exceeds supported magnitude")
            value = Fraction(decimal)
        elif isinstance(node, ast.Name):
            if node.id not in values:
                raise ValueError(f"missing earlier calculator result: {node.id}")
            value = values[node.id]
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = child(node.operand)
            if isinstance(node.op, ast.USub):
                value = -value
        elif isinstance(node, ast.BinOp):
            a, b = child(node.left), child(node.right)
            if isinstance(node.op, ast.Pow):
                if b.denominator != 1 or abs(b) > 512:
                    raise ValueError("power requires an integer exponent in -512..512")
                if abs(b) * (max(a.numerator.bit_length(), a.denominator.bit_length()) - 1) > MAX_BITS:
                    raise ValueError("power exceeds the numeric budget")
                value = a ** int(b)
            elif type(node.op) in _BINARY:
                value = _BINARY[type(node.op)](a, b)
            else:
                raise ValueError("unsupported operator")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCTIONS:
            if len(node.args) != 2 or node.keywords:
                raise ValueError("expected two integer arguments")
            a, b = (child(arg) for arg in node.args)
            if a.denominator != 1 or b.denominator != 1:
                raise ValueError("expected integer arguments")
            if node.func.id in ("comb", "perm") and not (0 <= b <= a <= 4096):
                raise ValueError("requires 0 <= k <= n <= 4096")
            value = Fraction(_FUNCTIONS[node.func.id](int(a), int(b)))
        else:
            raise ValueError("unsupported calculator syntax")
        if max(value.numerator.bit_length(), value.denominator.bit_length()) > MAX_BITS:
            raise ValueError("result exceeds the numeric budget")
        return value

    return visit(tree.body)


def _iso_date_or_timestamp(value):
    try:
        return dt.date.fromisoformat(value)
    except ValueError:
        return dt.datetime.fromisoformat(value)


def date_result(operation="now", timezone="UTC", **args):
    if not isinstance(timezone, str) or len(timezone) > 128:
        raise ValueError("expected an IANA timezone")
    zone = ZoneInfo(timezone)
    if operation == "now" and not args:
        return dt.datetime.now(zone).isoformat()
    if operation == "convert" and set(args) == {"value"}:
        value = dt.datetime.fromisoformat(args["value"])
        if value.tzinfo is None:
            raise ValueError("convert requires an ISO datetime with offset")
        return value.astimezone(zone).isoformat()
    if operation == "add_days" and set(args) == {"value", "days"}:
        if type(args["days"]) is not int or abs(args["days"]) > 366000:
            raise ValueError("days must be an integer within -366000..366000")
        return (dt.date.fromisoformat(args["value"]) + dt.timedelta(days=args["days"])).isoformat()
    if operation == "difference" and set(args) == {"start", "end"}:
        start, end = (_iso_date_or_timestamp(args[key]) for key in ("start", "end"))
        if type(start) is not type(end):
            raise ValueError("difference requires two dates or two timestamps")
        if isinstance(start, dt.datetime):
            if start.tzinfo is None:
                start = start.replace(tzinfo=zone)
            if end.tzinfo is None:
                end = end.replace(tzinfo=zone)
            return (end.astimezone(dt.timezone.utc) - start.astimezone(dt.timezone.utc)).total_seconds()
        return (end - start).days
    raise ValueError("invalid datetime operation or arguments")


def regex_result(default_text, pattern, text=None, flags="", operation="find"):
    if not isinstance(pattern, str) or not 1 <= len(pattern) <= 256 or re.search(r"\{\d{5,}", pattern):
        raise ValueError("invalid or oversized regex pattern")
    if not isinstance(flags, str) or set(flags) - set("ims"):
        raise ValueError("regex flags must be i, m or s")
    text = default_text if text is None else text
    if not isinstance(text, str) or len(text.encode("utf-8")) > 4 * 1024 * 1024:
        raise ValueError("regex input must be at most 4 MiB")
    compiled = regex.compile(pattern, sum(bit for name, bit in (("i", regex.I), ("m", regex.M), ("s", regex.S)) if name in flags))
    matches = compiled.finditer(text, timeout=0.02)
    if operation == "count":
        return sum(1 for _ in matches)
    if operation == "find":
        found = [{"span": list(m.span()), "text": m.group()[:128]} for m in itertools.islice(matches, 9)]
        return {"matches": found[:8], "truncated": len(found) > 8}
    raise ValueError("regex operation must be find or count")


def execute(body, source, batch_number):
    try:
        if batch_number > MAX_BATCHES:
            raise ValueError("batch limit reached; continue with the available results")
        value = json.loads(body)
        if not isinstance(value, dict) or set(value) != {"calls"}:
            raise ValueError("expected a calls array")
        calls = value["calls"]
        if not isinstance(calls, list) or not 1 <= len(calls) <= MAX_ITEMS:
            raise ValueError("expected 1–8 batch items")
    except (ValueError, RecursionError) as exc:
        return [{"error": str(exc)[:160]}]

    results, values, seen = [], {}, set()
    for call in calls:
        row = {}
        try:
            if not isinstance(call, dict):
                raise ValueError("expected an object")
            args = dict(call)
            name, tool = args.pop("id", None), args.pop("tool", None)
            if not isinstance(name, str) or not _IDENTIFIER.fullmatch(name):
                raise ValueError("invalid item ID")
            row["id"] = name
            if name in seen:
                raise ValueError("duplicate item ID")
            seen.add(name)
            if tool == "calculate":
                number = calculate(values=values, **args)
                values[name] = number
                row["value"] = str(number)
                row["expr"] = args["expr"]
            elif tool == "datetime":
                row["value"] = date_result(**args)
            elif tool == "regex":
                row["value"] = regex_result(source, **args)
            else:
                raise ValueError("unknown mini tool")
        except (TypeError, ValueError, ArithmeticError, DecimalException, SyntaxError, TimeoutError,
                RecursionError, regex.error, ZoneInfoNotFoundError) as exc:
            row["error"] = str(exc)[:160]
        results.append(row)
    return results


def packed(results):
    rows = [{k: v for k, v in row.items() if k != "expr"} for row in results]
    return ("\n<mini_result>" + json.dumps(rows, ensure_ascii=False, separators=(",", ":"))
            .replace("<", r"\u003c").replace(">", r"\u003e") + "</mini_result>\n")


def result_ids(tok, results, native, reasoning):
    data = tok.encode(packed(results), parse_special=False)
    if not native:
        return data
    # A standard function call ends an assistant turn. Resume with the same role markers as the Qwen template.
    close = "\n</think>" if reasoning else ""
    before = close + "<|im_end|>\n<|im_start|>user\n<tool_response>"
    after = "</tool_response><|im_end|>\n<|im_start|>assistant\n<think>\n"
    return tok.encode(before, parse_special=True) + data + tok.encode(after, parse_special=True)


def notice(results):
    ok = sum("value" in row for row in results)
    lines = [f"\n\n**🧮 Tool batch** · {ok}/{len(results)} succeeded\n"]
    for row in results:
        label = row.get("id", "batch")
        if "expr" in row:
            label = f"{label} · {row['expr']}"
        value = row.get("value", row.get("error"))
        if isinstance(value, dict):
            count = len(value["matches"])
            value = f"{count} " + ("match" if count == 1 else "matches") + (" (truncated)" if value["truncated"] else "")
        clean = lambda x: str(x).replace("`", "′").replace("<", "‹").replace(">", "›").replace("\n", " ").replace("\r", " ")[:160]
        lines.append(f"- `{clean(label)}` → `{clean(value)}`\n")
    return "".join(lines) + "\n"
