"""Filtros para reduzir falsos positivos de vagas livres.

Rejeita caixas livres que:
- diferem muito do tamanho/aspecto das ocupadas
- nao se alinham a fileiras/colunas de ocupadas
- cobrem vegetacao (canteiro)
"""
from __future__ import annotations

import cv2
import numpy as np

try:
    from pseudo_vagas_carpk import _agrupar, _centro, _altura, _largura, _mediana
except ImportError:
    from src.pseudo_vagas_carpk import _agrupar, _centro, _altura, _largura, _mediana


def _area(c):
    return max(1.0, _largura(c) * _altura(c))


def _aspecto(c):
    return _largura(c) / _altura(c)


def rejeita_vegetacao(imagem, caixa, limiar=0.28):
    """True se a ROI tem muitos pixels verdes saturados (canteiro/arbusto)."""
    h, w = imagem.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in caixa]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 + 2 or y2 <= y1 + 2:
        return True
    roi = imagem[y1:y2, x1:x2]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    # verde folhagem tipico
    mask = cv2.inRange(hsv, (35, 40, 40), (95, 255, 255))
    return float(np.count_nonzero(mask)) / float(mask.size) >= limiar


def tamanho_ok(livre, ocupadas, tol=0.45):
    if not ocupadas:
        return True
    med_a = _mediana([_area(c) for c in ocupadas])
    med_r = _mediana([_aspecto(c) for c in ocupadas])
    if med_a <= 0 or med_r <= 0:
        return True
    a, r = _area(livre), _aspecto(livre)
    if abs(a - med_a) / med_a > tol:
        return False
    if abs(r - med_r) / med_r > tol:
        return False
    return True


def alinhada_com_ocupadas(livre, ocupadas, min_grupo=2, fator_tol=0.55):
    """Centro da livre deve pertencer a uma fileira ou coluna com >= min_grupo ocupadas."""
    if len(ocupadas) < min_grupo:
        return False
    cx, cy = _centro(livre)

    for eixo, coord in (("y", cy), ("x", cx)):
        grupos = _agrupar(ocupadas, eixo=eixo, fator_tol=fator_tol)
        med_h = _mediana([_altura(c) for c in ocupadas])
        med_w = _mediana([_largura(c) for c in ocupadas])
        tol = fator_tol * (med_h if eixo == "y" else med_w)
        for g in grupos:
            if len(g) < min_grupo:
                continue
            refs = [_centro(c)[1 if eixo == "y" else 0] for c in g]
            if abs(coord - _mediana(refs)) <= tol:
                return True
    return False


def filtrar_livres(imagem, livres, ocupadas, usar_vegetacao=True, usar_alinhamento=True, usar_tamanho=True):
    """Aplica filtros e devolve apenas livres plausiveis.

    Alinhamento so e exigido quando ha poucas livres vs ocupadas (caso aereo
    com muitos FPs). No PKLot, com muitas livres YOLO, so tamanho+vegetacao.
    """
    # Heuristica: se ja ha muitas livres, confiar mais no YOLO (PKLot)
    estrito = len(livres) < max(3, int(0.25 * max(1, len(ocupadas))))
    alinhar = usar_alinhamento and estrito
    checar_tam = usar_tamanho and estrito
    saida = []
    for lv in livres:
        if checar_tam and ocupadas and not tamanho_ok(lv, ocupadas):
            continue
        if alinhar and not alinhada_com_ocupadas(lv, ocupadas):
            continue
        if usar_vegetacao and rejeita_vegetacao(imagem, lv):
            continue
        saida.append(lv)
    return saida
