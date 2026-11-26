"""Monta combinado_v3: PKLot + CARPK anotado (sem pseudo-livres).

Exige que anotacao/carpk_livres/labels contenha ao menos algumas linhas classe 0.

Uso:
    python src/montar_combinado_v3.py --anotacao anotacao/carpk_livres --pklot pklot --saida combinado_v3
"""
from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

YAML = """path: {root}
train: train/images
val: valid/images
test: test/images
names:
  0: space-empty
  1: space-occupied
"""


def _contar_classe(lbl: Path, cls: int) -> int:
    if not lbl.exists():
        return 0
    return sum(1 for ln in lbl.read_text(encoding="utf-8").splitlines() if ln.startswith(f"{cls} "))


def copiar_pklot(origem: Path, destino: Path):
    for split in ("train", "valid", "test"):
        for tipo in ("images", "labels"):
            src = origem / split / tipo
            dst = destino / split / tipo
            dst.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                continue
            for f in src.iterdir():
                if f.is_file():
                    shutil.copy2(f, dst / f"pklot_{f.name}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--anotacao", type=Path, default=Path("anotacao/carpk_livres"))
    p.add_argument("--pklot", type=Path, default=Path("pklot"))
    p.add_argument("--saida", type=Path, default=Path("combinado_v3"))
    p.add_argument("--valid-frac", type=float, default=0.15)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    img_dir = args.anotacao / "images"
    lbl_dir = args.anotacao / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        raise SystemExit(f"Pasta de anotacao incompleta: {args.anotacao}")

    pares = []
    n_livres = 0
    for img in sorted(list(img_dir.glob("*.png")) + list(img_dir.glob("*.jpg"))):
        lbl = lbl_dir / f"{img.stem}.txt"
        livres = _contar_classe(lbl, 0)
        n_livres += livres
        pares.append((img, lbl, livres))

    if n_livres < 10:
        raise SystemExit(
            f"Poucas vagas livres anotadas (classe 0): {n_livres}. "
            "Anote space-empty em anotacao/carpk_livres/labels e rode de novo."
        )

    random.seed(args.seed)
    random.shuffle(pares)
    n_valid = max(1, int(len(pares) * args.valid_frac))
    valid = pares[:n_valid]
    train = pares[n_valid:]

    if args.saida.exists():
        shutil.rmtree(args.saida)
    args.saida.mkdir(parents=True)

    print("Copiando PKLot...")
    copiar_pklot(args.pklot, args.saida)

    for split, subset in (("train", train), ("valid", valid), ("test", valid)):
        idir = args.saida / split / "images"
        ldir = args.saida / split / "labels"
        idir.mkdir(parents=True, exist_ok=True)
        ldir.mkdir(parents=True, exist_ok=True)
        for img, lbl, _ in subset:
            shutil.copy2(img, idir / f"carpk_{img.name}")
            if lbl.exists():
                shutil.copy2(lbl, ldir / f"carpk_{img.stem}.txt")
            else:
                (ldir / f"carpk_{img.stem}.txt").write_text("")

    root = args.saida.resolve().as_posix()
    (args.saida / "combinado_v3.yaml").write_text(YAML.format(root=root), encoding="utf-8")

    for split in ("train", "valid", "test"):
        n = len(list((args.saida / split / "images").glob("*")))
        print(f"  {split}: {n} imagens")
    print(f"CARPK livres anotadas (total classe 0): {n_livres}")
    print(f"YAML: {args.saida / 'combinado_v3.yaml'}")
    print("Treine: invoke treinar-combinado-v3")


if __name__ == "__main__":
    main()
