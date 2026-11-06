"""Mede precisão, revocação e F1 de cada classe (livre e ocupada) e o FPS.

Compara as vagas previstas com as vagas de resultados/vagas.json (gerado por pklot_para_vagas.py).

Uso:
    python src/avaliar_ocupacao.py --imagens pklot/test/images \
        --pesos runs/detect/pklot/weights/best.pt

Grava resultados/metricas.csv.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import cv2

from ocupacao import detectar_vagas, iou

LIMIAR_PAREAMENTO = 0.5  # IoU mínimo para uma vaga prevista contar como acerto
CLASSES = ("livre", "ocupada")


def parear(previstas, reais):
    """Pareamento guloso por IoU. Retorna o número de acertos (TP)."""
    usadas, tp = set(), 0
    for p in previstas:
        melhor, melhor_iou = None, LIMIAR_PAREAMENTO
        for k, r in enumerate(reais):
            valor = iou(p, r)
            if k not in usadas and valor >= melhor_iou:
                melhor, melhor_iou = k, valor
        if melhor is not None:
            usadas.add(melhor)
            tp += 1
    return tp


def main():
    from ultralytics import YOLO

    p = argparse.ArgumentParser()
    p.add_argument("--imagens", required=True, type=Path)
    p.add_argument("--vagas", default=Path("resultados/vagas.json"), type=Path)
    p.add_argument("--pesos", required=True)
    p.add_argument("--conf", default=0.25, type=float)
    p.add_argument("--saida", default=Path("resultados"), type=Path)
    args = p.parse_args()

    modelo = YOLO(args.pesos)
    vagas = json.loads(args.vagas.read_text())
    tp = {c: 0 for c in CLASSES}
    previstas_total = {c: 0 for c in CLASSES}
    reais_total = {c: 0 for c in CLASSES}
    tempo, n_imagens = 0.0, 0

    for nome, lista in vagas.items():
        imagem = cv2.imread(str(args.imagens / nome))
        if imagem is None:
            print(f"Imagem não encontrada: {nome}")
            continue
        inicio = time.perf_counter()
        livres, ocupadas = detectar_vagas(modelo, imagem, args.conf)
        tempo += time.perf_counter() - inicio
        n_imagens += 1

        previstas = {"livre": livres, "ocupada": ocupadas}
        for c in CLASSES:
            reais = [v["caixa"] for v in lista if v["real"] == c]
            reais_total[c] += len(reais)
            previstas_total[c] += len(previstas[c])
            tp[c] += parear(previstas[c], reais)

    if n_imagens == 0:
        raise SystemExit("Nenhuma imagem avaliada.")

    args.saida.mkdir(parents=True, exist_ok=True)
    with open(args.saida / "metricas.csv", "w", newline="") as arq:
        w = csv.writer(arq)
        w.writerow(["classe", "precisao", "revocacao", "f1"])
        for c in CLASSES:
            prec = tp[c] / previstas_total[c] if previstas_total[c] else 0.0
            rev = tp[c] / reais_total[c] if reais_total[c] else 0.0
            f1 = 2 * prec * rev / (prec + rev) if prec + rev else 0.0
            w.writerow([c, round(prec, 4), round(rev, 4), round(f1, 4)])
            print(f"{c}: precisão {prec:.2%} | revocação {rev:.2%} | F1 {f1:.2f}")
    print(f"Imagens: {n_imagens} | FPS: {n_imagens / tempo:.1f}")


if __name__ == "__main__":
    main()
