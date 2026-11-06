"""Extrai um subconjunto do ZIP do PKLot (formato YOLO do Roboflow) e cria pklot/pklot.yaml.

O PKLot tem muitos quadros parecidos. Um subconjunto basta e treina mais rápido.
O conjunto de teste é extraído por inteiro.

Uso:
    python src/preparar_pklot.py --zip PKLot.v1-raw.yolov8.zip --treino 2500 --valid 500
"""
import argparse
import random
import zipfile
from pathlib import Path

YAML = """train: train/images
val: valid/images
test: test/images
names:
  0: space-empty
  1: space-occupied
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", required=True, type=Path)
    p.add_argument("--saida", default=Path("pklot"), type=Path)
    p.add_argument("--treino", default=2500, type=int)
    p.add_argument("--valid", default=500, type=int)
    args = p.parse_args()

    random.seed(0)
    with zipfile.ZipFile(args.zip) as z:
        nomes = z.namelist()
        existentes = set(nomes)
        escolhidos = [n for n in ("data.yaml",) if n in existentes]
        for pasta, qtd in (("train", args.treino), ("valid", args.valid)):
            imagens = sorted(n for n in nomes if n.startswith(f"{pasta}/images/") and not n.endswith("/"))
            for img in random.sample(imagens, min(qtd, len(imagens))):
                rotulo = img.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
                escolhidos += [img, rotulo]
        escolhidos += [n for n in nomes if n.startswith("test/") and not n.endswith("/")]

        novos = 0
        for n in escolhidos:
            if n in existentes and not (args.saida / n).exists():
                z.extract(n, args.saida)
                novos += 1

    (args.saida / "pklot.yaml").write_text(YAML)
    print(f"{novos} arquivos extraídos em {args.saida}")
    print(f"Treine com: invoke treinar")


if __name__ == "__main__":
    main()
