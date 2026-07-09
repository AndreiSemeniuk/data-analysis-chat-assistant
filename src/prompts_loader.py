"""Prompt & persona loading with hot-reload.

Prompts live in prompts/*.md and the persona in config/persona.yaml - plain
files a non-developer can edit. They are re-read from disk on every turn
(mtime-checked), so changing the report tone requires no redeploy and no
restart: edit the YAML, the very next answer uses it. This is the
"Agility / Persona Management" requirement; in production the same loader
reads from a config service or object storage instead of local disk.
"""

from pathlib import Path

import yaml

from src.settings import settings

_cache: dict[Path, tuple[float, str]] = {}


def _read_fresh(path: Path) -> str:
    mtime = path.stat().st_mtime
    cached = _cache.get(path)
    if cached and cached[0] == mtime:
        return cached[1]
    text = path.read_text()
    _cache[path] = (mtime, text)
    return text


def load_prompt(name: str) -> str:
    return _read_fresh(settings.prompts_dir / f"{name}.md")


def load_persona() -> dict:
    return yaml.safe_load(_read_fresh(settings.persona_file))


def persona_as_text() -> str:
    p = load_persona()
    lines = [f"- Tone: {p.get('tone', 'professional')}",
             f"- Audience: {p.get('audience', 'retail executives')}",
             f"- Default report style: {p.get('report_style', 'concise summary')}"]
    if p.get("extra_instructions"):
        lines.append(f"- Additional instructions: {p['extra_instructions']}")
    return "\n".join(lines)
