import sys
import subprocess
from pathlib import Path

def run(backend: str, model: Path, image: Path) -> int:
    py = Path(f"/usr/bin/python-{backend}")
    cmd = [str(py), "challenge/infer_one.py", str(model), str(image)]
    out = subprocess.check_output(cmd, text=True)
    return int(out.strip())

def main():
    if len(sys.argv) != 3:
        print("usage: infer_two.py <model.pt> <image.png>", file=sys.stderr)
        sys.exit(2)

    model = Path(sys.argv[1])
    image = Path(sys.argv[2])

    p_openblas = run("openblas", model, image)
    p_blis     = run("blis",     model, image)

    print("openblas:", p_openblas)
    print("blis:    ", p_blis)
    print("chimera:", p_openblas != p_blis)

if __name__ == "__main__":
    main()