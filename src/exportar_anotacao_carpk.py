"""Exporta ~100 imagens CARPK para anotacao manual de vagas livres.

Gera:
  anotacao/carpk_livres/images/
  anotacao/carpk_livres/labels/   (so classe 1 = ocupada; acrescente classe 0)
  anotacao/carpk_livres/classes.txt
  anotacao/carpk_livres/manifest.csv

Uso:
    python src/exportar_anotacao_carpk.py --zip carpk_dl/carpk.zip --maximo 100
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import zipfile
from collections import defaultdict
from pathlib import Path

from pseudo_vagas_carpk import coco_para_xyxy, xyxy_para_yolo


def _site(nome: str) -> str:
    parts = Path(nome).stem.split("_")
    return parts[1] if len(parts) > 2 else "UNK"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=Path, default=Path("carpk_dl/carpk.zip"))
    p.add_argument("--saida", type=Path, default=Path("anotacao/carpk_livres"))
    p.add_argument("--maximo", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    random.seed(args.seed)
    img_dir = args.saida / "images"
    lbl_dir = args.saida / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(args.zip) as z:
        train_ann = json.loads(z.read("carpk_coco/train_annotations.json"))
        test_ann = json.loads(z.read("carpk_coco/test_annotations.json"))

        by_img = defaultdict(list)
        for a in train_ann["annotations"]:
            by_img[a["image_id"]].append(a["bbox"])
        for a in test_ann["annotations"]:
            by_img[a["image_id"]].append(a["bbox"])

        imgs = []
        for info in train_ann["images"]:
            imgs.append(("train", info, by_img[info["id"]]))
        for info in test_ann["images"]:
            imgs.append(("test", info, by_img[info["id"]]))

        # diversificar por site
        por_site = defaultdict(list)
        for item in imgs:
            por_site[_site(item[1]["file_name"])].append(item)
        selecionados = []
        sites = sorted(por_site.keys())
        while len(selecionados) < args.maximo and any(por_site.values()):
            for site in sites:
                if len(selecionados) >= args.maximo:
                    break
                if not por_site[site]:
                    continue
                random.shuffle(por_site[site])
                selecionados.append(por_site[site].pop())

        classes = "space-empty\nspace-occupied\n"
        (args.saida / "classes.txt").write_text(classes, encoding="utf-8")
        # LabelImg YOLO mode le classes.txt na pasta de labels
        (lbl_dir / "classes.txt").write_text(classes, encoding="utf-8")
        rows = []
        for pasta_zip, info, bboxes in selecionados:
            nome = info["file_name"]
            membro = f"carpk_coco/{pasta_zip}/{nome}"
            if membro not in z.namelist():
                cands = [n for n in z.namelist() if n.endswith("/" + nome)]
                if not cands:
                    continue
                membro = cands[0]
            dest = img_dir / nome
            if not dest.exists():
                dest.write_bytes(z.read(membro))
            linhas = [
                xyxy_para_yolo(coco_para_xyxy(bb), info["width"], info["height"], 1)
                for bb in bboxes
            ]
            (lbl_dir / f"{Path(nome).stem}.txt").write_text(
                "\n".join(linhas) + ("\n" if linhas else ""), encoding="utf-8"
            )
            rows.append(
                {
                    "arquivo": nome,
                    "site": _site(nome),
                    "split_origem": pasta_zip,
                    "ocupadas": len(bboxes),
                    "w": info["width"],
                    "h": info["height"],
                }
            )

    with open(args.saida / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["arquivo", "site", "split_origem", "ocupadas", "w", "h"])
        w.writeheader()
        w.writerows(rows)

    print(f"Exportadas {len(rows)} imagens em {args.saida}")
    print("Proximo: anote classe 0 (space-empty) nos labels e rode invoke montar-combinado-v3")


if __name__ == "__main__":
    main()
