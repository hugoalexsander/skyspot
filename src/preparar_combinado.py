"""Monta um dataset YOLO combinando PKLot (empty/occupied) e CARPK.

O CARPK anota apenas veículos (COCO). Cada carro vira a classe 1 (space-occupied).
Com --pseudo-livres (padrão), lacunas restritas entre carros viram classe 0 (space-empty).

Uso:
    python src/preparar_combinado.py --carpk-zip carpk_dl/carpk.zip --pklot pklot --saida combinado
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path

from pseudo_vagas_carpk import linhas_yolo_carpk

YAML = """path: {root}
train: train/images
val: valid/images
test: test/images
names:
  0: space-empty
  1: space-occupied
"""


def copiar_pklot(origem: Path, destino: Path, forcar=False):
    for split_src, split_dst in (("train", "train"), ("valid", "valid"), ("test", "test")):
        for tipo in ("images", "labels"):
            src = origem / split_src / tipo
            dst = destino / split_dst / tipo
            dst.mkdir(parents=True, exist_ok=True)
            if not src.exists():
                continue
            for f in src.iterdir():
                if f.is_file():
                    alvo = dst / f"pklot_{f.name}"
                    if forcar or not alvo.exists():
                        shutil.copy2(f, alvo)


def limpar_carpk(destino: Path):
    """Remove imagens/labels CARPK anteriores para regenerar rótulos."""
    for split in ("train", "valid", "test"):
        for tipo in ("images", "labels"):
            pasta = destino / split / tipo
            if not pasta.exists():
                continue
            for f in pasta.glob("carpk_*"):
                f.unlink()


def extrair_carpk(
    zip_path: Path,
    destino: Path,
    treino=800,
    valid=189,
    seed=0,
    pseudo_livres=True,
):
    random.seed(seed)
    n_livres = 0
    n_ocup = 0
    with zipfile.ZipFile(zip_path) as z:
        train_ann = json.loads(z.read("carpk_coco/train_annotations.json"))
        test_ann = json.loads(z.read("carpk_coco/test_annotations.json"))

        by_img = defaultdict(list)
        for a in train_ann["annotations"]:
            by_img[a["image_id"]].append(a["bbox"])
        imagens = train_ann["images"][:]
        random.shuffle(imagens)
        n_train = min(treino, len(imagens))
        n_valid = min(valid, max(0, len(imagens) - n_train))
        splits = {
            "train": imagens[:n_train],
            "valid": imagens[n_train : n_train + n_valid],
        }

        test_by_img = defaultdict(list)
        for a in test_ann["annotations"]:
            test_by_img[a["image_id"]].append(a["bbox"])
        test_imgs = test_ann["images"][:]
        random.shuffle(test_imgs)
        splits["test"] = test_imgs[: min(200, len(test_imgs))]
        meta = {"train": by_img, "valid": by_img, "test": test_by_img}

        novos = 0
        for split, imgs in splits.items():
            img_dir = destino / split / "images"
            lbl_dir = destino / split / "labels"
            img_dir.mkdir(parents=True, exist_ok=True)
            lbl_dir.mkdir(parents=True, exist_ok=True)
            for info in imgs:
                nome = info["file_name"]
                pasta_zip = "train" if split in ("train", "valid") else "test"
                membro = f"carpk_coco/{pasta_zip}/{nome}"
                if membro not in z.namelist():
                    cands = [n for n in z.namelist() if n.endswith("/" + nome)]
                    if not cands:
                        continue
                    membro = cands[0]
                dest_img = img_dir / f"carpk_{nome}"
                if not dest_img.exists():
                    dest_img.write_bytes(z.read(membro))
                    novos += 1
                bboxes = meta[split][info["id"]]
                linhas = linhas_yolo_carpk(
                    bboxes, info["width"], info["height"], pseudo_livres=pseudo_livres
                )
                for ln in linhas:
                    if ln.startswith("0 "):
                        n_livres += 1
                    else:
                        n_ocup += 1
                (lbl_dir / f"carpk_{Path(nome).stem}.txt").write_text(
                    "\n".join(linhas) + ("\n" if linhas else "")
                )
        return novos, {k: len(v) for k, v in splits.items()}, n_livres, n_ocup


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--carpk-zip", type=Path, default=Path("carpk_dl/carpk.zip"))
    p.add_argument("--pklot", type=Path, default=Path("pklot"))
    p.add_argument("--saida", type=Path, default=Path("combinado"))
    p.add_argument("--carpk-treino", type=int, default=800)
    p.add_argument("--carpk-valid", type=int, default=189)
    p.add_argument(
        "--pseudo-livres",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Gera space-empty por lacunas restritas no CARPK (padrão: sim)",
    )
    p.add_argument(
        "--limpar-carpk",
        action="store_true",
        help="Apaga imagens/labels carpk_* antes de regenerar",
    )
    args = p.parse_args()

    args.saida.mkdir(parents=True, exist_ok=True)
    if args.limpar_carpk:
        print("Limpando CARPK anterior...")
        limpar_carpk(args.saida)
    print("Copiando PKLot...")
    copiar_pklot(args.pklot, args.saida)
    modo = "ocupadas + pseudo-livres" if args.pseudo_livres else "só ocupadas"
    print(f"Extraindo CARPK ({modo})...")
    novos, counts, n_livres, n_ocup = extrair_carpk(
        args.carpk_zip,
        args.saida,
        args.carpk_treino,
        args.carpk_valid,
        pseudo_livres=args.pseudo_livres,
    )
    root = args.saida.resolve().as_posix()
    (args.saida / "combinado.yaml").write_text(YAML.format(root=root))
    for split in ("train", "valid", "test"):
        n_img = len(list((args.saida / split / "images").glob("*")))
        print(f"  {split}: {n_img} imagens")
    print(
        f"CARPK novos arquivos: {novos} "
        f"(train={counts['train']}, valid={counts['valid']}, test={counts['test']})"
    )
    print(f"Rótulos CARPK: ocupadas={n_ocup} | livres(pseudo)={n_livres}")
    print(f"YAML: {args.saida / 'combinado.yaml'}")
    print("Treine com: invoke treinar_combinado")


if __name__ == "__main__":
    main()
