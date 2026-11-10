"""Detecta vagas livres e ocupadas.

Modos:
  direto  — classes do YOLO (space-empty / space-occupied)
  inferir — estima livres pelas lacunas restritas (fallback aereo)
  auto    — so YOLO + filtros (sem merge geometrico); se nao houver livres YOLO
            e houver ocupadas, nao inventa lacunas por padrao

Uso:
    python src/ocupacao.py --imagem img.jpg --pesos runs/detect/pklot_carpk_v2/weights/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2

try:
    from filtros_vagas import filtrar_livres
    from pseudo_vagas_carpk import inferir_vagas_livres_restrito
except ImportError:
    from src.filtros_vagas import filtrar_livres
    from src.pseudo_vagas_carpk import inferir_vagas_livres_restrito


def iou(a, b):
    """IoU entre duas caixas [x1, y1, x2, y2]."""
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    uniao = area(a) + area(b) - inter
    return inter / uniao if uniao > 0 else 0.0


def inferir_vagas_livres(ocupadas, fator_vaga=1.55):
    del fator_vaga
    return inferir_vagas_livres_restrito(ocupadas)


def detectar_yolo(modelo, imagem, conf=0.25, conf_livre=0.45, imgsz=640):
    """Retorna (livres, ocupadas). Livres usam conf_livre (mais rigoroso)."""
    # Roda com o menor limiar e filtra livres depois pela conf por classe
    conf_min = min(conf, conf_livre)
    r = modelo(imagem, conf=conf_min, imgsz=imgsz, verbose=False)[0]
    livres, ocupadas = [], []
    if r.boxes is None or len(r.boxes) == 0:
        return livres, ocupadas
    xyxy = r.boxes.xyxy.cpu().tolist()
    clss = r.boxes.cls.cpu().tolist()
    confs = r.boxes.conf.cpu().tolist()
    for caixa, cls, score in zip(xyxy, clss, confs):
        nome = str(modelo.names[int(cls)]).lower()
        vazia = "empty" in nome or "livre" in nome or "vacant" in nome
        if vazia:
            if score >= conf_livre:
                livres.append(caixa)
        else:
            if score >= conf:
                ocupadas.append(caixa)
    return livres, ocupadas


def detectar_vagas(
    modelo,
    imagem,
    conf=0.25,
    imgsz=640,
    modo="auto",
    fator_vaga=1.55,
    conf_livre=0.45,
    filtrar=True,
):
    """Retorna (livres, ocupadas) como listas de caixas [x1, y1, x2, y2]."""
    livres_yolo, ocupadas = detectar_yolo(modelo, imagem, conf, conf_livre, imgsz)

    if modo == "direto":
        livres = filtrar_livres(imagem, livres_yolo, ocupadas) if filtrar else livres_yolo
        return livres, ocupadas

    if modo == "auto":
        # Sem merge geometrico: so YOLO + filtros (evita via/canteiro inventados)
        livres = filtrar_livres(imagem, livres_yolo, ocupadas) if filtrar else livres_yolo
        return livres, ocupadas

    # modo inferir
    livres_inf = inferir_vagas_livres(ocupadas, fator_vaga=fator_vaga)
    livres = filtrar_livres(imagem, livres_inf, ocupadas) if filtrar else livres_inf
    return livres, ocupadas


def desenhar(imagem, livres, ocupadas):
    saida = imagem.copy()
    for caixas, cor in ((ocupadas, (0, 0, 255)), (livres, (0, 200, 0))):
        for caixa in caixas:
            x1, y1, x2, y2 = map(int, caixa)
            cv2.rectangle(saida, (x1, y1), (x2, y2), cor, 2)
    return saida


def main():
    from ultralytics import YOLO

    p = argparse.ArgumentParser()
    p.add_argument("--imagem", required=True, type=Path)
    p.add_argument("--pesos", required=True)
    p.add_argument("--conf", default=0.25, type=float)
    p.add_argument("--conf-livre", default=0.45, type=float, dest="conf_livre")
    p.add_argument("--imgsz", default=640, type=int)
    p.add_argument("--modo", default="auto", choices=("auto", "direto", "inferir"))
    p.add_argument("--fator-vaga", default=1.55, type=float, dest="fator_vaga")
    p.add_argument("--sem-filtro", action="store_true")
    p.add_argument("--saida", default=Path("resultados"), type=Path)
    args = p.parse_args()

    imagem = cv2.imread(str(args.imagem))
    if imagem is None:
        raise SystemExit(f"Nao foi possivel abrir {args.imagem}")
    livres, ocupadas = detectar_vagas(
        YOLO(args.pesos),
        imagem,
        args.conf,
        args.imgsz,
        modo=args.modo,
        fator_vaga=args.fator_vaga,
        conf_livre=args.conf_livre,
        filtrar=not args.sem_filtro,
    )

    args.saida.mkdir(parents=True, exist_ok=True)
    destino = args.saida / f"{args.imagem.stem}_ocupacao.png"
    cv2.imwrite(str(destino), desenhar(imagem, livres, ocupadas))
    print(f"Modo: {args.modo} | Livres: {len(livres)} | Ocupadas: {len(ocupadas)}")
    print(f"Imagem salva em {destino}")


if __name__ == "__main__":
    main()
