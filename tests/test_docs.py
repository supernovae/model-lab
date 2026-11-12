"""Documentation stays connected to the code it describes.

These checks are structural: every page is indexed, links and anchors resolve,
and each ``klab`` command a current page shows exists with the options shown.
They do not judge wording.
"""

from __future__ import annotations

import argparse
import re
import shlex
import subprocess
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_LINK = re.compile(r"\]\(([^)\s]+)\)")


def _pages() -> list[Path]:
    listed = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return [Path(name) for name in listed]


def _slug(heading: str) -> str:
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", heading.strip().lower())
    return re.sub(r"[^\w\- ]", "", text.replace("`", "")).replace(" ", "-")


@cache
def _anchors(page: Path) -> frozenset[str]:
    found: set[str] = set()
    seen: dict[str, int] = {}
    fenced = False
    for line in (ROOT / page).read_text(encoding="utf-8").splitlines():
        if line.lstrip().startswith("```"):
            fenced = not fenced
        heading = None if fenced else re.match(r"#{1,6}\s+(.*)", line)
        if heading:
            slug = _slug(heading.group(1))
            count = seen.get(slug, 0)
            seen[slug] = count + 1
            found.add(slug if count == 0 else f"{slug}-{count}")
    return frozenset(found)


def test_links_and_anchors_resolve() -> None:
    broken = []
    for page in _pages():
        for target in _LINK.findall((ROOT / page).read_text(encoding="utf-8")):
            if "://" in target or target.startswith("mailto:"):
                continue
            path, _, fragment = target.partition("#")
            destination = (ROOT / page.parent / path) if path else ROOT / page
            if not destination.exists():
                broken.append(f"{page}: {target}")
            elif fragment and destination.suffix == ".md":
                relative = destination.resolve().relative_to(ROOT)
                if fragment not in _anchors(relative):
                    broken.append(f"{page}: {target}")
    assert not broken, broken


def test_every_docs_page_is_indexed_once() -> None:
    def linked(index: Path) -> list[str]:
        targets = _LINK.findall((ROOT / index).read_text(encoding="utf-8"))
        return [target.partition("#")[0] for target in targets if "/" not in target]

    index = Path("docs/README.md")
    pages = {path.name for path in (ROOT / "docs").glob("*.md")} - {"README.md"}
    listed = [name for name in linked(index) if name in pages]
    assert sorted(listed) == sorted(pages), (
        f"{index} must list every page in docs exactly once: "
        f"missing {sorted(pages - set(listed))}, "
        f"repeated {sorted({name for name in listed if listed.count(name) > 1})}"
    )


def _subcommands(parser: argparse.ArgumentParser) -> dict:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    return {}


def _options(parser: argparse.ArgumentParser) -> set[str]:
    return {name for action in parser._actions for name in action.option_strings}


def _shown_commands(text: str) -> list[str]:
    """``klab`` invocations in fenced blocks and inline code."""
    commands, block, fenced = [], [], False
    for line in text.splitlines():
        if line.lstrip().startswith("```"):
            if fenced:
                commands += re.sub(r"\\\n\s*", " ", "\n".join(block)).splitlines()
                block = []
            fenced = not fenced
        elif fenced:
            block.append(line)
        else:
            commands += re.findall(r"`([^`]*\bklab [^`]*)`", line)
    return commands


def command_problems(page: object, text: str) -> list[str]:
    """Each ``klab`` command ``text`` shows that does not exist, or its unknown options."""
    from klab.cli.main import build_parser
    from klab.runtime_env_cli import parser as runtime_env_parser

    root = build_parser()
    global_options = _options(root)
    problems = []
    for shown in _shown_commands(text):
        invocation = re.search(r"(?:^|[\s\"'/])klab[\"']?\s+(.*)", shown)
        if invocation is None or "-m klab." in shown:
            continue
        words = re.split(r"\s(?:\||>|>>|&&|;|2>|<)\s?|\s#\s", invocation.group(1))[0]
        try:
            tokens = shlex.split(words)
        except ValueError:
            tokens = words.split()
        parser, chain = root, []
        if tokens[:2] == ["runtime", "env"]:
            parser, chain, tokens = runtime_env_parser(), tokens[:2], tokens[2:]
        # Global options come before the subcommand and may take a value.
        while tokens and tokens[0].split("=")[0] in global_options:
            takes_value = "=" not in tokens[0] and tokens[0] != "--version"
            tokens = tokens[2 if takes_value else 1 :]
        while tokens and _subcommands(parser):
            word = tokens[0]
            if word in _subcommands(parser):
                parser, chain, tokens = (
                    _subcommands(parser)[word],
                    [*chain, word],
                    tokens[1:],
                )
                continue
            if re.fullmatch(r"[a-z][a-z0-9\-]*", word):
                problems.append(f"{page}: no command `klab {' '.join(chain)} {word}`")
            break
        else:
            known = _options(parser) | global_options
            for token in tokens:
                option = token.split("=")[0]
                if re.fullmatch(r"--[a-z][a-z0-9\-]*", option) and (
                    option not in known
                ):
                    problems.append(f"{page}: `klab {' '.join(chain)}` has no {option}")
    return problems


def test_commands_shown_in_current_pages_exist() -> None:
    problems = [
        problem
        for page in _pages()
        for problem in command_problems(page, (ROOT / page).read_text(encoding="utf-8"))
    ]
    assert not problems, problems
