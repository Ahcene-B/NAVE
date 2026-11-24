# SPDX-License-Identifier: MIT
"""
Python environment for the NAVE worker (standard library only, runs inside GIMP).

Order: a path set by the user -> the saved environment -> a venv made from a local
Python 3.10+ -> (after asking) a venv made by a pinned, checksum-verified uv.
"""
import glob
import hashlib
import io
import json
import os
import platform
import shutil
import stat
import subprocess
import sys
import tarfile
import threading
import time
import urllib.request
import zipfile

REQUIREMENTS = ["numpy>=1.23,<3", "onnxruntime>=1.17,<2"]
MIN_PY = (3, 10)
UV_VERSION = "0.13.5"
UV_PYTHON = "3.12"
UV_SHA256 = {
    "x86_64-unknown-linux-gnu.tar.gz": "f68d291c75783122dad58ab77def0e13f3233f358731db5d6d7e8715603b928f",
    "aarch64-unknown-linux-gnu.tar.gz": "4bb69cf95ed13f23713bb7bc09f241896ea0c75b7c557e615a20604b82f9c403",
    "aarch64-apple-darwin.tar.gz": "743f85035a1996ecb350659e01cd9faea9709829be09b1b4b03c73eaf3c9cd73",
    "x86_64-apple-darwin.tar.gz": "b0f52b0e3945b4107d89024ea38e625683011fbfc05a0d98f9fa2d6dcee28c66",
    "x86_64-pc-windows-msvc.zip": "a38b8d1902b7cfbc76b423ba1bb456f623f42064cdc72628b6075b89afd1019a",
    "aarch64-pc-windows-msvc.zip": "09f47edf90726084a70f9426f6e3ca4f46e9bf7cbf23b1716ec64d60b055fae3",
}
UV_DOWNLOAD_MB = 100      # uv + standalone Python + numpy + onnxruntime, approximate
WINDOWS = os.name == "nt"
NO_WINDOW = 0x08000000 if WINDOWS else 0


class SetupError(RuntimeError):
    pass


def cache_dir():
    if sys.platform == "darwin":
        base = os.path.expanduser("~/Library/Caches")
    elif WINDOWS:
        base = os.environ.get("LOCALAPPDATA") or os.path.expanduser("~\\AppData\\Local")
    else:
        base = os.environ.get("XDG_CACHE_HOME") or os.path.expanduser("~/.cache")
    return os.path.join(base, "nave")


def _config_path():
    return os.path.join(cache_dir(), "config.json")


def load_config():
    try:
        with open(_config_path()) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_config(cfg):
    os.makedirs(cache_dir(), exist_ok=True)
    with open(_config_path(), "w") as f:
        json.dump(cfg, f, indent=1)


def venv_python(venv):
    return os.path.join(venv, "Scripts", "python.exe") if WINDOWS else os.path.join(venv, "bin", "python")


# --------------------------------------------------------------------------
# Running commands without freezing progress
# --------------------------------------------------------------------------
def run(cmd, timeout, pulse=None, env=None):
    """Run a command, calling pulse() while it works. Returns (returncode, output)."""
    with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                          env=env, creationflags=NO_WINDOW) as p:
        start = time.time()
        lines = []
        t = threading.Thread(target=lambda: lines.extend(p.stdout), daemon=True)
        t.start()
        while p.poll() is None:
            if time.time() - start > timeout:
                p.kill()
                return -1, "".join(lines) + f"\n(timed out after {timeout}s)"
            if pulse:
                pulse()
            time.sleep(0.2)
        t.join(5)
    return p.returncode, "".join(lines)


def env_works(py, pulse=None):
    if not py or not os.path.exists(py):
        return False
    code, _ = run([py, "-c", "import numpy, onnxruntime"], 120, pulse)
    return code == 0


# --------------------------------------------------------------------------
# Local Python discovery
# --------------------------------------------------------------------------
def _candidates():
    names = ["python3"] + [f"python3.{m}" for m in range(14, MIN_PY[1] - 1, -1)]
    found = []
    if sys.platform == "darwin":
        dirs = ["/opt/homebrew/bin", "/usr/local/bin"]
        found += [os.path.join(d, n) for d in dirs for n in names]
        found += sorted(glob.glob("/Library/Frameworks/Python.framework/Versions/3.*/bin/python3"), reverse=True)
        try:   # /usr/bin/python3 is a stub that opens an installer without the Command Line Tools
            if subprocess.run(["xcode-select", "-p"], capture_output=True, timeout=10).returncode == 0:
                found.append("/usr/bin/python3")
        except (OSError, subprocess.SubprocessError):
            pass
    elif WINDOWS:
        try:
            r = subprocess.run(["py", "-3", "-c", "import sys; print(sys.executable)"], capture_output=True,
                               text=True, timeout=20, creationflags=NO_WINDOW)
            if r.returncode == 0:
                found.append(r.stdout.strip())
        except (OSError, subprocess.SubprocessError):
            pass
        for n in ("python", "python3"):
            p = shutil.which(n)
            if p and "windowsapps" not in p.lower():        # skip the Microsoft Store alias
                found.append(p)
    else:
        found += [p for p in (shutil.which(n) for n in names) if p]
    seen, out = set(), []
    for p in found:
        if os.path.exists(p):
            real = os.path.realpath(p)
            if real not in seen:
                seen.add(real)
                out.append(p)
    return out


def local_pythons(pulse=None):
    """[(path, None)] for usable candidates, [(path, reason)] for rejected ones."""
    probe = ("import sys\ntry:\n import venv, ensurepip\n ok=''\nexcept ImportError as e:\n ok=str(e)\n"
             "print('%d.%d|%s' % (sys.version_info[0], sys.version_info[1], ok))")
    results = []
    for p in _candidates():
        code, out = run([p, "-c", probe], 30, pulse)
        if code != 0:
            results.append((p, "does not run"))
            continue
        ver, _, missing = out.strip().splitlines()[-1].partition("|")
        major, minor = map(int, ver.split("."))
        if (major, minor) < MIN_PY:
            results.append((p, f"Python {ver} is older than {MIN_PY[0]}.{MIN_PY[1]}"))
        elif missing:
            hint = " (on Debian/Ubuntu: install the python3-venv package)" if sys.platform.startswith("linux") else ""
            results.append((p, f"Python {ver} has no venv module{hint}"))
        else:
            results.append((p, None))
    return results


def make_venv(base_python, venv, pulse=None):
    shutil.rmtree(venv, ignore_errors=True)
    code, out = run([base_python, "-m", "venv", venv], 300, pulse)
    if code != 0:
        return False, out
    py = venv_python(venv)
    code, out = run([py, "-m", "pip", "install", "--disable-pip-version-check", "--no-input"] + REQUIREMENTS,
                    1200, pulse)
    if code != 0:
        return False, out
    return env_works(py, pulse), out


# --------------------------------------------------------------------------
# uv fallback
# --------------------------------------------------------------------------
def uv_target():
    m = platform.machine().lower()
    arch = "aarch64" if m in ("arm64", "aarch64") else "x86_64" if m in ("x86_64", "amd64") else None
    if arch is None:
        return None
    if sys.platform == "darwin":
        return f"{arch}-apple-darwin.tar.gz"
    if WINDOWS:
        return f"{arch}-pc-windows-msvc.zip"
    if sys.platform.startswith("linux"):
        return f"{arch}-unknown-linux-gnu.tar.gz"
    return None


def install_uv(pulse=None):
    target = uv_target()
    if target not in UV_SHA256:
        raise SetupError(f"No uv build for this system ({platform.system()} {platform.machine()}).")
    dest = os.path.join(cache_dir(), "uv")
    exe = os.path.join(dest, "uv.exe" if WINDOWS else "uv")
    if os.path.exists(exe):
        return exe
    url = f"https://github.com/astral-sh/uv/releases/download/{UV_VERSION}/uv-{target}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "nave-gimp-plugin"})
        with urllib.request.urlopen(req, timeout=60) as r:
            data = r.read()
    except Exception as e:
        raise SetupError(f"Could not download uv from {url}:\n{e}")
    if pulse:
        pulse()
    if hashlib.sha256(data).hexdigest() != UV_SHA256[target]:
        raise SetupError("The downloaded uv does not match its expected checksum; not using it.")
    os.makedirs(dest, exist_ok=True)
    wanted = "uv.exe" if WINDOWS else "uv"
    if target.endswith(".zip"):
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            name = next(n for n in z.namelist() if os.path.basename(n) == wanted)
            with open(exe, "wb") as f:
                f.write(z.read(name))
    else:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as t:
            member = next(m for m in t.getmembers() if os.path.basename(m.name) == wanted and m.isfile())
            with open(exe, "wb") as f:
                f.write(t.extractfile(member).read())
    os.chmod(exe, os.stat(exe).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return exe


def make_venv_with_uv(venv, pulse=None):
    uv = install_uv(pulse)
    env = dict(os.environ,
               UV_PYTHON_INSTALL_DIR=os.path.join(cache_dir(), "python"),
               UV_CACHE_DIR=os.path.join(cache_dir(), "uv-cache"),
               UV_PYTHON_PREFERENCE="only-managed")
    shutil.rmtree(venv, ignore_errors=True)
    code, out = run([uv, "venv", "--python", UV_PYTHON, venv], 900, pulse, env=env)
    if code != 0:
        raise SetupError("uv could not create the environment:\n" + out[-2000:])
    py = venv_python(venv)
    code, out = run([uv, "pip", "install", "--link-mode", "copy", "--python", py] + REQUIREMENTS,
                    1200, pulse, env=env)
    shutil.rmtree(env["UV_CACHE_DIR"], ignore_errors=True)    # packages were copied into the venv
    if code != 0 or not env_works(py, pulse):
        raise SetupError("uv could not install numpy and onnxruntime:\n" + out[-2000:])
    return py


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def ensure_python(user_python="", ask_uv=None, status=None, pulse=None):
    """Return a Python that can run the worker.
    ask_uv(message) -> bool asks the user; None means non-interactive (never download uv).
    status(text) reports progress; pulse() keeps a progress bar alive."""
    status = status or (lambda t: None)
    if user_python:
        if env_works(user_python, pulse):
            return user_python
        raise SetupError(f"The External Python '{user_python}' cannot import numpy and onnxruntime.")

    cfg = load_config()
    if env_works(cfg.get("python"), pulse):
        return cfg["python"]

    venv = os.path.join(cache_dir(), "venv")
    os.makedirs(cache_dir(), exist_ok=True)
    log_path = os.path.join(cache_dir(), "setup.log")
    rejected = []
    with open(log_path, "w") as log:
        status("NAVE: looking for Python")
        for py, reason in local_pythons(pulse):
            if reason:
                rejected.append(f"{py}: {reason}")
                continue
            status(f"NAVE: installing numpy and onnxruntime with {py}")
            log.write(f"\n=== {py}\n")
            ok, out = make_venv(py, venv, pulse)
            log.write(out)
            if ok:
                save_config({"python": venv_python(venv), "base": py})
                return venv_python(venv)
            rejected.append(f"{py}: installing numpy/onnxruntime failed (see {log_path})")

    found = "\n".join(rejected) or "No Python 3 was found."
    message = (f"NAVE needs Python {MIN_PY[0]}.{MIN_PY[1]}+ with numpy and onnxruntime.\n\n{found}\n\n"
               f"Download a private Python with uv instead? About {UV_DOWNLOAD_MB} MB goes into\n"
               f"{cache_dir()}\nand nothing else on your system changes. Delete that folder to remove it.")
    if ask_uv is None:
        raise SetupError(message.split("\n\nDownload")[0] + "\n\nRun the plugin interactively once to set it up.")
    if not ask_uv(message):
        raise SetupError("Setup cancelled. Install Python 3.10+ (or put the path of a Python that has numpy "
                         "and onnxruntime in the External Python field) and try again.")
    status("NAVE: downloading uv, Python, numpy and onnxruntime")
    py = make_venv_with_uv(venv, pulse)
    save_config({"python": py, "base": "uv"})
    return py
