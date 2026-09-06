"""Stage 3 verification: adversarial tests for the path-containment boundary.

Builds a real temp filesystem (including a Windows directory junction that
escapes the allowed root) and asserts each case is allowed or rejected.

Run:
    venv\\Scripts\\python.exe scripts\\test_safety.py
"""
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from safety import PathNotAllowed, resolve_within  # noqa: E402

ALLOW = "ALLOW"
DENY = "DENY"


def build_fixture(base: Path):
    """allowed/ is the root. secret/ and allowed_evil/ must stay unreachable."""
    allowed = base / "allowed"
    (allowed / "sub").mkdir(parents=True)
    (allowed / "inside.txt").write_text("ok")
    (allowed / "sub" / "deep.txt").write_text("ok")

    secret = base / "secret"
    secret.mkdir()
    (secret / "stolen.txt").write_text("SENSITIVE")

    # Sibling whose name has the root as a string prefix -- catches naive
    # startswith() containment checks.
    evil = base / "allowed_evil"
    evil.mkdir()
    (evil / "bad.txt").write_text("SENSITIVE")

    # Directory junction inside the root pointing OUT of it. Junctions need no
    # admin rights on Windows, so this is a realistic escape route.
    junction = allowed / "escape"
    made_junction = False
    if sys.platform == "win32":
        r = subprocess.run(["cmd", "/c", "mklink", "/J", str(junction), str(secret)],
                           capture_output=True, text=True)
        made_junction = r.returncode == 0
        if not made_junction:
            print(f"  (note: could not create junction: {r.stderr.strip() or r.stdout.strip()})")
    return allowed, secret, evil, junction, made_junction


def main():
    base = Path(tempfile.mkdtemp(prefix="jarvis_safety_"))
    failures = 0
    try:
        allowed, secret, evil, junction, made_junction = build_fixture(base)
        roots = [allowed]

        cases = [
            (ALLOW, "file directly inside root",      allowed / "inside.txt"),
            (ALLOW, "the root itself",                allowed),
            (ALLOW, "nested file",                    allowed / "sub" / "deep.txt"),
            (ALLOW, "not-yet-created file (write)",   allowed / "new_file.txt"),
            (ALLOW, "case-insensitive match",         Path(str(allowed).upper()) / "INSIDE.TXT"),
            (DENY,  "outside the root",               secret / "stolen.txt"),
            (DENY,  "'..' traversal out",             allowed / ".." / "secret" / "stolen.txt"),
            (DENY,  "sibling prefix (allowed_evil)",  evil / "bad.txt"),
            (DENY,  "alternate data stream",          str(allowed / "inside.txt") + ":hidden"),
            (DENY,  "reserved device name",           allowed / "CON"),
            (DENY,  "empty path",                     ""),
        ]
        if made_junction:
            cases.append((DENY, "junction escaping root", junction / "stolen.txt"))
        else:
            print("  SKIPPED junction-escape case (could not create junction)")

        print(f"\nroot = {allowed}\n")
        for expected, name, candidate in cases:
            try:
                resolved = resolve_within(candidate, roots)
                actual, detail = ALLOW, str(resolved)
            except PathNotAllowed as exc:
                actual, detail = DENY, str(exc)

            ok = actual == expected
            failures += not ok
            print(f"  [{'ok ' if ok else 'BAD'}] expect {expected:5s} got {actual:5s}  {name}")
            if not ok:
                print(f"         -> {detail}")

        # Fails closed when misconfigured.
        try:
            resolve_within(allowed / "inside.txt", [])
            print("  [BAD] expect DENY  got ALLOW  empty root list must reject everything")
            failures += 1
        except PathNotAllowed:
            print("  [ok ] expect DENY  got DENY   empty root list rejects everything")

    finally:
        shutil.rmtree(base, ignore_errors=True)

    print(f"\n{'FAIL' if failures else 'PASS'}: {failures} failing case(s)")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
