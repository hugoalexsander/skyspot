"""Roda deteccao nas amostras aereas com pesos v3 (ou v2 se v3 nao existir).

Uso:
    python src/validar_outros_v3.py
"""
from __future__ import annotations

from pathlib import Path

import cv2
from ultralytics import YOLO

try:
    from ocupacao import detectar_vagas, desenhar
except ImportError:
    from src.ocupacao import detectar_vagas, desenhar


def main():
    v3 = Path("runs/detect/pklot_carpk_v3/weights/best.pt")
    v2 = Path("runs/detect/pklot_carpk_v2/weights/best.pt")
    pesos = v3 if v3.exists() else v2
    if not pesos.exists():
        raise SystemExit("Nenhum peso v2/v3 encontrado.")
    print(f"Pesos: {pesos}")
    modelo = YOLO(str(pesos))
    root = Path("resultados/outros_v3" if v3.exists() else "resultados/outros_v2_filtros")
    for nome, pasta in [
        ("carpk_amostras", Path("outros_datasets/carpk_amostras")),
        ("carpk_extra", Path("outros_datasets/carpk_extra")),
        ("drone", Path("outros_datasets/drone")),
    ]:
        if not pasta.exists():
            continue
        saida = root / nome
        saida.mkdir(parents=True, exist_ok=True)
        print(f"\n=== {nome} ===")
        for p in sorted(list(pasta.glob("*.png")) + list(pasta.glob("*.jpg"))):
            im = cv2.imread(str(p))
            L, O = detectar_vagas(modelo, im, conf=0.25, conf_livre=0.45, imgsz=800, modo="auto")
            cv2.imwrite(str(saida / f"{p.stem}_ocupacao.png"), desenhar(im, L, O))
            print(f"{p.name[:50]:50} L={len(L):3d} O={len(O):3d}")


if __name__ == "__main__":
    main()
