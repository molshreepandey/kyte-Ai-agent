"""Branding guard: the old project name must not reappear in shipped files, and the
assistant persona must come from configuration."""

from __future__ import annotations

from pathlib import Path

from app.agent import build_rag_tool, build_system_prompt

ROOT = Path(__file__).resolve().parent.parent
OLD_NAME = "ky" + "te"  # split so this file does not match itself
SKIP_DIRS = {".git", ".venv", "venv", "node_modules", "data", "logs", "test-results", "playwright-report"}
SKIP_SUFFIXES = {".pyc", ".npz", ".png", ".ico"}
SKIP_FILES = {"package-lock.json"}


def _shipped_files() -> list[Path]:
    out = []
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if (
            not p.is_file()
            or set(rel.parts) & SKIP_DIRS
            or any(part.startswith(".") and part != ".github" for part in rel.parts[:-1])
        ):
            continue
        if p.suffix in SKIP_SUFFIXES or p.name in SKIP_FILES:
            continue
        out.append(p)
    return out


def test_old_brand_absent_from_repository() -> None:
    offenders = [
        str(p.relative_to(ROOT))
        for p in _shipped_files()
        if OLD_NAME in p.name.lower() or OLD_NAME in p.read_text(encoding="utf-8", errors="ignore").lower()
    ]
    assert offenders == []


def test_persona_is_configurable() -> None:
    prompt = build_system_prompt("Acme Bank", "a retail bank", "accounts and loans", agent_name="Ava")
    assert prompt.startswith("You are Ava, a friendly voice assistant for Acme Bank, a retail bank.")
    assert "Acme Bank (accounts and loans)" in prompt
    anon = build_system_prompt("Acme Bank", "a retail bank", "accounts and loans")
    assert anon.startswith("You are a friendly voice assistant for Acme Bank")
    tool = build_rag_tool("Acme Bank", "accounts and loans")
    assert "Acme Bank knowledge base" in tool["function"].get("description", "")
