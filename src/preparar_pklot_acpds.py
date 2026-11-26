"""Monta dataset YOLO com PKLot ampliado + ACPDS (vagas livres/ocupadas reais).

- PKLot: copia de pklot/ (ja no formato YOLO)
- ACPDS: quadrilateros normalizados -> AABB YOLO (0=empty, 1=occupied)
- Sem CARPK / sem pseudo-rotulos

Uso:
    python src/preparar_pklot_acpds.py --acpds-zip acpds_dl/rois_gopro.zip --pklot pklot --saida pklot_acpds
"""
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from pathlib import Path

import cv2
import numpy as np

YAML = """path: {root}
train: train/images
val: valid/images
test: test/images
names:
  0: space-empty
  1: space-occupied
"""


def quad_para_yolo(pts, ocupado: bool) -> str:
    """pts: [[x,y],...] normalizados -> linha YOLO AABB."""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
    y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
    bw, bh = x2 - x1, y2 - y1
    if bw < 1e-4 or bh < 1e-4:
        return ""
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    cls = 1 if ocupado else 0
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def extrair_acpds(zip_path: Path, bruto: Path):
    bruto.mkdir(parents=True, exist_ok=True)
    if not (bruto / "annotations.json").exists():
        print(f"Extraindo ACPDS para {bruto}...")
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(bruto)


def converter_acpds(bruto: Path, destino: Path):
    ann = json.loads((bruto / "annotations.json").read_text(encoding="utf-8"))
    n_livre = n_ocup = 0
    for split_src, split_dst in (("train", "train"), ("valid", "valid"), ("test", "test")):
        img_dir = destino / split_dst / "images"
        lbl_dir = destino / split_dst / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        a = ann[split_src]
        for fname, rois, occs in zip(a["file_names"], a["rois_list"], a["occupancy_list"]):
            src = bruto / "images" / fname
            if not src.exists():
                # case
                cands = list((bruto / "images").glob(fname)) + list((bruto / "images").glob(fname.upper()))
                if not cands:
                    continue
                src = cands[0]
            dest_img = img_dir / f"acpds_{Path(fname).stem}.jpg"
            if not dest_img.exists():
                # reencode para jpg padronizado
                im = cv2.imread(str(src))
                if im is None:
                    continue
                cv2.imwrite(str(dest_img), im, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            linhas = []
            for pts, occ in zip(rois, occs):
                ln = quad_para_yolo(pts, bool(occ))
                if not ln:
                    continue
                linhas.append(ln)
                if ln.startswith("0 "):
                    n_livre += 1
                else:
                    n_ocup += 1
            (lbl_dir / f"acpds_{Path(fname).stem}.txt").write_text(
                "\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8"
            )
    return n_livre, n_ocup


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
                    alvo = dst / (f.name if f.name.startswith("pklot_") else f"pklot_{f.name}")
                    if not alvo.exists():
                        shutil.copy2(f, alvo)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--acpds-zip", type=Path, default=Path("acpds_dl/rois_gopro.zip"))
    p.add_argument("--acpds-bruto", type=Path, default=Path("acpds_dl/raw"))
    p.add_argument("--pklot", type=Path, default=Path("pklot"))
    p.add_argument("--saida", type=Path, default=Path("pklot_acpds"))
    args = p.parse_args()

    if args.saida.exists():
        shutil.rmtree(args.saida)
    args.saida.mkdir(parents=True)

    extrair_acpds(args.acpds_zip, args.acpds_bruto)
    print("Convertendo ACPDS -> YOLO...")
    n_l, n_o = converter_acpds(args.acpds_bruto, args.saida)
    print(f"  ACPDS: livres={n_l} ocupadas={n_o}")
    print("Copiando PKLot...")
    copiar_pklot(args.pklot, args.saida)

    root = args.saida.resolve().as_posix()
    (args.saida / "pklot_acpds.yaml").write_text(YAML.format(root=root), encoding="utf-8")
    for split in ("train", "valid", "test"):
        n = len(list((args.saida / split / "images").glob("*")))
        print(f"  {split}: {n} imagens")
    print(f"YAML: {args.saida / 'pklot_acpds.yaml'}")


if __name__ == "__main__":
    main()
