"""Prepare a separate CPU-only runtime using reviewed official wheel sources."""

from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import subprocess
import sys
from urllib.parse import urlparse, unquote
from urllib.request import Request, urlopen

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ENV = HERE / "venv_cpu"
ACTIVE = Path('<LOCAL_HOST_PATH_02>')


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, obj):
    with path.open("x", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def active_fingerprint():
    names = [ACTIVE / "pyvenv.cfg", ACTIVE / "Lib/site-packages/torch/version.py"]
    names += sorted((ACTIVE / "Lib/site-packages").glob("*.dist-info/METADATA"))
    return {str(x): digest(x) for x in names}


def fetch(url, path):
    if path.exists():
        return {"url": url, "final_url": None, "path": path.name,
                "sha256": digest(path), "bytes": path.stat().st_size,
                "reused_first_attempt_metadata": True}
    req = Request(url, headers={"User-Agent": "PFE-UNet-local-research-audit"})
    with urlopen(req, timeout=90) as response:
        data = response.read(16 * 1024**2 + 1)
        if len(data) > 16 * 1024**2:
            raise RuntimeError("Metadata response exceeds limit")
        with path.open("xb") as f:
            f.write(data)
        return {"url": url, "final_url": response.url, "path": path.name,
                "sha256": digest(path), "bytes": len(data)}


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == "a":
            self.links.extend(v for k, v in attrs if k == "href")


def run(command, name, env):
    with (HERE / (name + ".log")).open("x", encoding="utf-8") as log:
        log.write(json.dumps(command) + "\n")
        log.flush()
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, env=env, check=True)


def main():
    if ENV.exists():
        raise RuntimeError("Existing runtime: inspect it; do not overwrite or reprovision")
    if HERE.drive.upper() != "D:" or ENV.resolve().is_relative_to(ACTIVE.resolve()):
        raise RuntimeError("Runtime must be isolated on D:")
    before = active_fingerprint()
    if (HERE / "active_before.json").exists():
        if json.loads((HERE / "active_before.json").read_text()) != before:
            raise RuntimeError("Active environment changed since first metadata fetch")
    else:
        save(HERE / "active_before.json", before)
    evidence = HERE / "official_evidence"
    evidence.mkdir(exist_ok=True)
    sources = [fetch(url, evidence / name) for url, name in (
        ("https://pypi.org/pypi/torch/json", "torch_pypi.json"),
        ("https://api.github.com/repos/pytorch/pytorch/security-advisories", "pytorch_advisories.json"),
        ("https://download.pytorch.org/whl/cpu/torch/", "torch_cpu_index.html"),
    )]
    advisories = json.loads((evidence / "pytorch_advisories.json").read_text())
    required = {"GHSA-63cw-57p8-fm3p", "GHSA-53q9-r3pm-6pq6"}
    if not required <= {x["ghsa_id"] for x in advisories}:
        raise RuntimeError("Required security advisories missing")
    parser = Links()
    parser.feed((evidence / "torch_cpu_index.html").read_text())
    filename = "torch-2.14.0+cpu-cp311-cp311-win_amd64.whl"
    selected = [x for x in parser.links if unquote(urlparse(x).path).endswith("/" + filename)]
    if len(selected) != 1 or urlparse(selected[0]).hostname not in {"download.pytorch.org", "download-r2.pytorch.org"}:
        raise RuntimeError(f"Expected exactly one official CPU wheel, found {selected}")
    if not urlparse(selected[0]).fragment.startswith("sha256="):
        raise RuntimeError("Official wheel hash missing")
    choice = {"utc": datetime.now(timezone.utc).isoformat(), "sources": sources,
              "selected_wheel": selected[0], "torch_version": "2.14.0+cpu",
              "purpose": "CPU checkpoint inspection only; not DirectML training",
              "security_scope": "Reviewed public PyTorch advisories; not a security guarantee or OS sandbox",
              "external_checkpoint_loaded": False, "training_started": False}
    save(HERE / "runtime_selection.json", choice)
    (HERE / "tmp").mkdir()
    (HERE / "wheelhouse").mkdir()
    env = os.environ.copy()
    env.update(TEMP=str(HERE / "tmp"), TMP=str(HERE / "tmp"),
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1",
               PIP_CONFIG_FILE=os.devnull, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    env.pop("PYTHONPATH", None)
    run([sys.executable, "-m", "venv", str(ENV)], "venv_create", env)
    python = str(ENV / "Scripts/python.exe")
    # A dry run records every resolved artifact before any package is installed.
    run([python, "-m", "pip", "--isolated", "--disable-pip-version-check", "install",
         "--dry-run", "--ignore-installed", "--only-binary=:all:", "--no-cache-dir",
         "--index-url", "https://pypi.org/simple", "--report", str(HERE / "pip_resolution.json"),
         selected[0], "numpy==2.2.6", "safetensors==0.6.2"], "pip_resolution", env)
    resolution = json.loads((HERE / "pip_resolution.json").read_text())
    lines = []
    for item in resolution["install"]:
        info = item["download_info"]
        host = urlparse(info["url"]).hostname
        if host not in {"download.pytorch.org", "download-r2.pytorch.org", "files.pythonhosted.org"}:
            raise RuntimeError("Unexpected package host: " + host)
        if not unquote(urlparse(info["url"]).path).endswith(".whl"):
            raise RuntimeError("Source distribution not permitted")
        sha = info["archive_info"]["hashes"]["sha256"]
        lines.append(f"{item['metadata']['name']} @ {info['url']} --hash=sha256:{sha}")
    with (HERE / "resolved_requirements.txt").open("x", encoding="ascii") as f:
        f.write("\n".join(lines) + "\n")
    run([python, "-m", "pip", "--isolated", "--disable-pip-version-check", "download",
         "--require-hashes", "--no-deps", "--no-cache-dir", "--only-binary=:all:",
         "--dest", str(HERE / "wheelhouse"), "-r", str(HERE / "resolved_requirements.txt")],
        "pip_download", env)
    wheels = sorted((HERE / "wheelhouse").glob("*.whl"))
    expected = {x["download_info"]["archive_info"]["hashes"]["sha256"] for x in resolution["install"]}
    actual = {digest(x) for x in wheels}
    if actual != expected or len(wheels) != len(resolution["install"]):
        raise RuntimeError("Wheelhouse hash mismatch")
    save(HERE / "wheelhouse_manifest.json", {x.name: digest(x) for x in wheels})
    run([python, "-m", "pip", "--isolated", "--disable-pip-version-check", "install",
         "--no-index", "--no-deps", "--no-cache-dir"] + [str(x) for x in wheels], "pip_install", env)
    run([python, "-m", "pip", "check"], "pip_check", env)
    probe = "import json,sys,torch,numpy,safetensors; print(json.dumps(dict(python=sys.executable,prefix=sys.prefix,torch=torch.__version__,torch_path=torch.__file__,cuda=torch.version.cuda,numpy=numpy.__version__,safetensors=safetensors.__version__)))"
    result = json.loads(subprocess.check_output([python, "-I", "-c", probe], env=env, text=True))
    if result["torch"] != "2.14.0+cpu" or result["cuda"] is not None:
        raise RuntimeError("Unexpected runtime")
    if not Path(result["torch_path"]).resolve().is_relative_to(ENV.resolve()):
        raise RuntimeError("Torch imported outside isolated runtime")
    after = active_fingerprint()
    save(HERE / "active_after.json", after)
    if before != after:
        raise RuntimeError("Active environment fingerprint changed")
    result.update(active_environment_unchanged=True, external_checkpoint_loaded=False,
                  training_started=False, utc=datetime.now(timezone.utc).isoformat())
    save(HERE / "runtime_ready.json", result)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
