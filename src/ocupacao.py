"""Detecta vagas livres e ocupadas em uma imagem, com o YOLOv8 treinado no PKLot.

Uso:
    python src/ocupacao.py --imagem img.jpg --pesos runs/detect/pklot/weights/best.pt
"""
import argparse
from pathlib import Path

import cv2


def iou(a, b):
    """IoU entre duas caixas [x1, y1, x2, y2]."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])
    uniao = area(a) + area(b) - inter
    return inter / uniao if uniao > 0 else 0.0


def detectar_vagas(modelo, imagem, conf=0.25, imgsz=640):
    """Retorna (livres, ocupadas) como listas de caixas [x1, y1, x2, y2]."""
    r = modelo(imagem, conf=conf, imgsz=imgsz, verbose=False)[0]
    livres, ocupadas = [], []
    for caixa, cls in zip(r.boxes.xyxy.cpu().tolist(), r.boxes.cls.cpu().tolist()):
        vazia = "empty" in str(modelo.names[int(cls)]).lower()
        (livres if vazia else ocupadas).append(caixa)
    return livres, ocupadas


def desenhar(imagem, livres, ocupadas):
    saida = imagem.copy()
    for caixas, cor in ((ocupadas, (0, 0, 255)), (livres, (0, 200, 0))):  # BGR
        for caixa in caixas:
            x1, y1, x2, y2 = map(int, caixa)
            cv2.rectangle(saida, (x1, y1), (x2, y2), cor, 2)
    return saida


def main():
    from ultralytics import YOLO

    p = argparse.ArgumentParser()
    p.add_argument("--imagem", required=True, type=Path)
    p.add_argument("--pesos", required=True, help="ex.: runs/detect/pklot/weights/best.pt")
    p.add_argument("--conf", default=0.25, type=float)
    p.add_argument("--imgsz", default=640, type=int)
    p.add_argument("--saida", default=Path("resultados"), type=Path)
    args = p.parse_args()

    imagem = cv2.imread(str(args.imagem))
    if imagem is None:
        raise SystemExit(f"Não foi possível abrir {args.imagem}")
    livres, ocupadas = detectar_vagas(YOLO(args.pesos), imagem, args.conf, args.imgsz)

    args.saida.mkdir(parents=True, exist_ok=True)
    destino = args.saida / f"{args.imagem.stem}_ocupacao.png"
    cv2.imwrite(str(destino), desenhar(imagem, livres, ocupadas))
    print(f"Livres: {len(livres)} | Ocupadas: {len(ocupadas)}")
    print(f"Imagem salva em {destino}")


if __name__ == "__main__":
    main()
