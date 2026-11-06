"""Converte as anotações do PKLot (Roboflow, formato YOLO) para resultados/vagas.json.

Assim você não precisa marcar as vagas à mão.

Uso:
    python src/pklot_para_vagas.py --pasta pklot/test --maximo 20

Espera <pasta>/images, <pasta>/labels e data.yaml na pasta acima.
Classes: space-empty vira "livre" e space-occupied vira "ocupada".
"""
import argparse
import json
import random
from pathlib import Path

import cv2
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pasta", required=True, type=Path, help="ex.: pklot/test")
    p.add_argument("--yaml", type=Path, default=None, help="padrão: <pasta>/../data.yaml")
    p.add_argument("--maximo", default=20, type=int, help="número de imagens sorteadas")
    p.add_argument("--saida", default=Path("resultados/vagas.json"), type=Path)
    args = p.parse_args()

    arq_yaml = args.yaml or args.pasta.parent / "data.yaml"
    nomes = yaml.safe_load(arq_yaml.read_text())["names"]
    if isinstance(nomes, dict):
        nomes = [nomes[k] for k in sorted(nomes)]
    estado = {}
    for i, nome in enumerate(nomes):
        nome = str(nome).lower()
        if "empty" in nome:
            estado[i] = "livre"
        elif "occupied" in nome:
            estado[i] = "ocupada"
    if len(estado) != 2:
        raise SystemExit(f"Não achei as classes empty e occupied em {arq_yaml}: {nomes}")

    imagens = sorted(
        f for f in (args.pasta / "images").iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    random.seed(0)
    random.shuffle(imagens)

    dados = json.loads(args.saida.read_text()) if args.saida.exists() else {}
    n = 0
    for img in imagens:
        if n >= args.maximo:
            break
        rotulo = args.pasta / "labels" / f"{img.stem}.txt"
        if not rotulo.exists():
            continue
        h, w = cv2.imread(str(img)).shape[:2]
        vagas = []
        for linha in rotulo.read_text().splitlines():
            partes = linha.split()
            if len(partes) < 5 or int(partes[0]) not in estado:
                continue
            xc, yc, bw, bh = map(float, partes[1:5])
            vagas.append({
                "caixa": [round((xc - bw / 2) * w), round((yc - bh / 2) * h),
                          round((xc + bw / 2) * w), round((yc + bh / 2) * h)],
                "real": estado[int(partes[0])],
            })
        dados[img.name] = vagas
        n += 1

    args.saida.parent.mkdir(parents=True, exist_ok=True)
    args.saida.write_text(json.dumps(dados, indent=2))
    print(f"{n} imagens salvas em {args.saida}")
    print(f"Avalie com: invoke avaliar --imagens {args.pasta / 'images'}")


if __name__ == "__main__":
    main()
