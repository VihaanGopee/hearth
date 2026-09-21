"""Scheduled jobs: launchd on macOS, cron on Linux.

A job runs `python -m src.run_once "<prompt>"` on a schedule. Jobs run
headless — keep prompts self-contained and have the agent write results
to a file in the workspace.
"""
from __future__ import annotations
import re
import shlex
import subprocess
import sys
from pathlib import Path
from .registry import register_tool

NAME_RE = re.compile(r"^[a-z0-9][a-z0-9_-]{0,39}$")
TIME_RE = re.compile(r"^(\d{1,2}):(\d{2})$")
LABEL_PREFIX = "com.hearth.job."


def _xml(s: str) -> str:
    return (s.replace("&", "&amp;").replace("<", "&lt;")
             .replace(">", "&gt;").replace('"', "&quot;"))


def _jobs_dir(ctx: dict) -> Path:
    d = Path(ctx["data_dir"]) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _validate(name: str, every_minutes, at_time):
    if not NAME_RE.match(name or ""):
        return "Name must match [a-z0-9_-], e.g. 'morning-brief'"
    if every_minutes is None and at_time is None:
        return "Give every_minutes or at_time (HH:MM)."
    if every_minutes is not None and int(every_minutes) < 1:
        return "every_minutes must be >= 1."
    if at_time is not None and not TIME_RE.match(at_time):
        return "at_time must look like '08:30'."
    return None


def _add_launchd(ctx, name, prompt, every_minutes, at_time):
    project = Path(ctx["project_dir"])
    jobs = _jobs_dir(ctx)
    label = LABEL_PREFIX + name
    plist = Path.home() / "Library/LaunchAgents" / f"{label}.plist"
    if every_minutes:
        trigger = (f"<key>StartInterval</key>"
                   f"<integer>{int(every_minutes) * 60}</integer>")
    else:
        hh, mm = TIME_RE.match(at_time).groups()
        trigger = ("<key>StartCalendarInterval</key><dict>"
                   f"<key>Hour</key><integer>{int(hh)}</integer>"
                   f"<key>Minute</key><integer>{int(mm)}</integer></dict>")
    content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>Label</key><string>{label}</string>
<key>ProgramArguments</key><array>
<string>{_xml(sys.executable)}</string><string>-m</string><string>src.run_once</string><string>{_xml(prompt)}</string>
</array>
<key>WorkingDirectory</key><string>{_xml(str(project))}</string>
{trigger}
<key>StandardOutPath</key><string>{_xml(str(jobs / f'{name}.log'))}</string>
<key>StandardErrorPath</key><string>{_xml(str(jobs / f'{name}.err'))}</string>
</dict></plist>
"""
    plist.parent.mkdir(parents=True, exist_ok=True)
    plist.write_text(content, encoding="utf-8")
    subprocess.run(["launchctl", "load", str(plist)], capture_output=True)
    return {"ok": True, "job": name, "schedule": str(plist)}


def _add_cron(ctx, name, prompt, every_minutes, at_time):
    project = Path(ctx["project_dir"])
    jobs = _jobs_dir(ctx)
    log = jobs / f"{name}.log"
    if every_minutes:
        sched = f"*/{int(every_minutes)} * * * *"
    else:
        hh, mm = TIME_RE.match(at_time).groups()
        sched = f"{int(mm)} {int(hh)} * * *"
    line = (f"{sched} cd {shlex.quote(str(project))} && "
            f"{shlex.quote(sys.executable)} -m src.run_once {shlex.quote(prompt)} "
            f">> {shlex.quote(str(log))} 2>&1  # hearth:{name}")
    cur = subprocess.run(["crontab", "-l"], capture_output=True,
                         text=True).stdout
    lines = [l for l in cur.splitlines() if f"# hearth:{name}" not in l]
    lines.append(line)
    subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n",
                   text=True, check=True)
    return {"ok": True, "job": name, "cron": sched}


def register(ctx: dict) -> None:
    def schedule_add(name: str, prompt: str,
                     every_minutes: int | None = None,
                     at_time: str | None = None):
        err = _validate(name, every_minutes, at_time)
        if err:
            return {"ok": False, "error": err}
        if sys.platform == "darwin":
            return _add_launchd(ctx, name, prompt, every_minutes, at_time)
        return _add_cron(ctx, name, prompt, every_minutes, at_time)

    def schedule_list():
        found = []
        if sys.platform == "darwin":
            for p in (Path.home() / "Library/LaunchAgents").glob(LABEL_PREFIX + "*.plist"):
                found.append(p.stem[len(LABEL_PREFIX):])
        else:
            cur = subprocess.run(["crontab", "-l"], capture_output=True,
                                 text=True).stdout
            for line in cur.splitlines():
                m = re.search(r"# hearth:([a-z0-9_-]+)", line)
                if m:
                    found.append(m.group(1))
        return {"ok": True, "jobs": sorted(set(found))}

    def schedule_remove(name: str):
        if sys.platform == "darwin":
            plist = Path.home() / "Library/LaunchAgents" / f"{LABEL_PREFIX}{name}.plist"
            if not plist.exists():
                return {"ok": False, "error": f"No such job: {name}"}
            subprocess.run(["launchctl", "unload", str(plist)], capture_output=True)
            plist.unlink()
        else:
            cur = subprocess.run(["crontab", "-l"], capture_output=True,
                                 text=True).stdout
            lines = [l for l in cur.splitlines() if f"# hearth:{name}" not in l]
            if len(lines) == len(cur.splitlines()):
                return {"ok": False, "error": f"No such job: {name}"}
            subprocess.run(["crontab", "-"], input="\n".join(lines) + "\n",
                           text=True, check=True)
        return {"ok": True, "removed": name}

    register_tool(
        "schedule_add",
        "Create a recurring background job that runs a prompt on a schedule. "
        "On macOS this installs a launchd agent; on Linux a cron entry. "
        "Keep the prompt self-contained and have it write results to a workspace file.",
        {"properties": {
            "name": {"type": "string", "description": "lowercase id, e.g. morning-brief"},
            "prompt": {"type": "string"},
            "every_minutes": {"type": "integer", "description": "Repeat interval in minutes"},
            "at_time": {"type": "string", "description": "Daily time as HH:MM, e.g. 08:30"}},
         "required": ["name", "prompt"]},
        schedule_add)
    register_tool(
        "schedule_list", "List installed recurring jobs.",
        {"properties": {}, "required": []}, schedule_list)
    register_tool(
        "schedule_remove", "Remove a recurring job by name.",
        {"properties": {"name": {"type": "string"}}, "required": ["name"]},
        schedule_remove)
