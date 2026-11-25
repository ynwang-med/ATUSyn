"""Create a source-only release archive; never edit training artifacts."""
import argparse
import hashlib
import re
import zipfile
from pathlib import Path

ROOT_FILES = {".gitignore", "README.md", "LICENSE", "requirements.txt"}
TREE_SUFFIXES = {
    "atusyn": {".py", ".cpp", ".cu", ".h", ".cuh"},
    "segment_anything": {".py"},
    "scripts": {".py", ".sh"},
    "tests": {".py"},
    "configs": {".yaml", ".yml"},
    "LICENSES": {".txt"},
}
EXTRA_FILES = {
    "atusyn/models/op/LICENSE_MIT",
    "tests/source_fingerprints.json",
    "splits/ReMIND/split.csv",
    "checkpoints/README.md",
}
SENSITIVE = {
    "absolute drive path": re.compile(r"(?<![A-Za-z0-9_])[A-Za-z]:[\\/]"),
    "private Unix path": re.compile(r"/(?:home|Users|Data\d*|mnt|media)/[A-Za-z0-9_.-]+/"),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    "access token": re.compile(r"(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{40,}|hf_[A-Za-z0-9]{30,}|AKIA[A-Z0-9]{16})"),
    "assigned secret": re.compile(r"""(?i)(?:api[_-]?key|access[_-]?token|password|client[_-]?secret)["']?\s*[:=]\s*["'][A-Za-z0-9_+/=-]{12,}["']"""),
}


def is_public(relative):
    p = Path(relative)
    if any(part.startswith(".") or part == "__pycache__" for part in p.parts):
        return p.as_posix() == ".gitignore"
    if ".local." in p.name:
        return False
    name = p.as_posix()
    return (name in ROOT_FILES or name in EXTRA_FILES or
            (len(p.parts) > 1 and p.suffix in TREE_SUFFIXES.get(p.parts[0], set())))


def collect(root):
    root = Path(root).resolve()
    candidates = {root/name for name in ROOT_FILES | EXTRA_FILES}
    for directory in TREE_SUFFIXES:
        base = root/directory
        if base.is_symlink():
            raise ValueError("Symlink is not allowed: " + directory)
        if base.exists():
            candidates.update(base.rglob("*"))
    selected = []
    for path in sorted(candidates):
        rel = path.relative_to(root)
        if path.is_symlink():
            raise ValueError("Symlink is not allowed: " + rel.as_posix())
        if not path.is_file() or not is_public(rel):
            continue
        content = path.read_bytes()
        text = content.decode("utf-8-sig")
        for label, pattern in SENSITIVE.items():
            if pattern.search(text):
                # Report file/category only; never echo potentially sensitive text.
                raise ValueError(label + " detected in " + rel.as_posix())
        selected.append((rel.as_posix(), content))
    if not selected:
        raise ValueError("No public source files found")
    return selected


def package(root, output):
    files = collect(root)
    output = Path(output)
    if output.suffix.lower() != ".zip":
        raise ValueError("Output must end in .zip")
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, content in files:
            archive.writestr("ATUSyn/" + name, content)
    with zipfile.ZipFile(output) as archive:
        expected = {"ATUSyn/" + name for name, _ in files}
        if set(archive.namelist()) != expected or archive.testzip() is not None:
            raise ValueError("Release archive verification failed")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    return len(files), digest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--check-only", action="store_true")
    p.add_argument("--output", default="dist/ATUSyn-code.zip")
    a = p.parse_args()
    root = Path(__file__).resolve().parents[1]
    if a.check_only:
        files = collect(root)
        print("[CHECK OK] %d source-only files; no archive created." % len(files))
    else:
        count, digest = package(root, a.output)
        print("[DONE] %d files: %s" % (count, a.output))
        print("SHA256:", digest)


if __name__ == "__main__":
    main()

