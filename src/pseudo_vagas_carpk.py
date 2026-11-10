"""Pseudo-rotulos de vagas livres a partir de caixas de carros (CARPK).

Regras (restritas para evitar corredores):
- Agrupa carros em fileiras OU colunas e escolhe o eixo dominante.
- Passo = mediana das distancias entre vizinhos proximos.
- Preenche lacunas ~(n+1)*passo com n em 1..max_vagas e gap < max_gap_fator*passo.
- Projeta fileiras/colunas paralelas vazias quando o espacamento entre grupos
  densos indica uma fileira de vagas ausente (nao corredor).
"""
from __future__ import annotations


def _mediana(vals):
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        return 0.0
    m = n // 2
    return vals[m] if n % 2 else 0.5 * (vals[m - 1] + vals[m])


def _centro(c):
    return (c[0] + c[2]) / 2, (c[1] + c[3]) / 2


def _largura(c):
    return max(1.0, c[2] - c[0])


def _altura(c):
    return max(1.0, c[3] - c[1])


def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = lambda r: max(0, r[2] - r[0]) * max(0, r[3] - r[1])
    uniao = area(a) + area(b) - inter
    return inter / uniao if uniao > 0 else 0.0


def _nms(caixas, limiar=0.3):
    if not caixas:
        return []
    ordem = sorted(caixas, key=lambda c: _largura(c) * _altura(c), reverse=True)
    keep = []
    for c in ordem:
        if all(_iou(c, k) < limiar for k in keep):
            keep.append(c)
    return keep


def coco_para_xyxy(bbox):
    x, y, w, h = bbox
    return [float(x), float(y), float(x + w), float(y + h)]


def xyxy_para_yolo(caixa, w_img, h_img, cls):
    x1, y1, x2, y2 = caixa
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2 / w_img
    cy = (y1 + y2) / 2 / h_img
    return f"{cls} {cx:.6f} {cy:.6f} {bw / w_img:.6f} {bh / h_img:.6f}"


def _agrupar(caixas, eixo="y", fator_tol=0.55):
    if not caixas:
        return []
    med_h = _mediana([_altura(c) for c in caixas])
    med_w = _mediana([_largura(c) for c in caixas])
    tol = fator_tol * (med_h if eixo == "y" else med_w)
    idx = 1 if eixo == "y" else 0

    ordenadas = sorted(caixas, key=lambda c: _centro(c)[idx])
    grupos, atual, refs = [], [], []
    for c in ordenadas:
        ref = _centro(c)[idx]
        if not atual:
            atual, refs = [c], [ref]
            continue
        if abs(ref - _mediana(refs)) <= tol:
            atual.append(c)
            refs.append(ref)
        else:
            grupos.append(atual)
            atual, refs = [c], [ref]
    if atual:
        grupos.append(atual)
    return grupos


def _passo_vizinhos(grupo, ao_longo="x", max_vizinho_fator=1.85):
    if len(grupo) < 2:
        return 0.0
    i_ord = 0 if ao_longo == "x" else 1
    med_w = _mediana([_largura(c) for c in grupo])
    med_h = _mediana([_altura(c) for c in grupo])
    tamanho = med_w if ao_longo == "x" else med_h
    ordenadas = sorted(grupo, key=lambda c: _centro(c)[i_ord])
    dists = []
    for a, b in zip(ordenadas, ordenadas[1:]):
        d = abs(_centro(b)[i_ord] - _centro(a)[i_ord])
        if tamanho < d <= max_vizinho_fator * tamanho:
            dists.append(d)
    if dists:
        return _mediana(dists)
    return tamanho * 1.15


def _lacunas_restritas(grupo, ao_longo="x", max_vagas=4, max_gap_fator=2.55):
    if len(grupo) < 2:
        return []
    i_ord = 0 if ao_longo == "x" else 1
    med_w = _mediana([_largura(c) for c in grupo])
    med_h = _mediana([_altura(c) for c in grupo])
    passo = _passo_vizinhos(grupo, ao_longo=ao_longo)
    if passo <= 1:
        return []

    ordenadas = sorted(grupo, key=lambda c: _centro(c)[i_ord])
    livres = []
    for a, b in zip(ordenadas, ordenadas[1:]):
        ca, cb = _centro(a), _centro(b)
        dist = abs(cb[i_ord] - ca[i_ord])
        if dist < 1.35 * passo:
            continue
        if dist > max_gap_fator * passo:
            continue
        n = int(round(dist / passo)) - 1
        n = max(0, min(max_vagas, n))
        if n == 0:
            continue
        esperado = (n + 1) * passo
        if abs(dist - esperado) > 0.45 * passo:
            continue
        for k in range(1, n + 1):
            t = k / (n + 1)
            cx = ca[0] + t * (cb[0] - ca[0])
            cy = ca[1] + t * (cb[1] - ca[1])
            livres.append([cx - med_w / 2, cy - med_h / 2, cx + med_w / 2, cy + med_h / 2])
    return livres


def _projetar_fileiras_paralelas(grupos, ao_longo="x", eixo_grupo="y"):
    """Insere vagas numa fileira/coluna vazia entre dois grupos densos vizinhos.

    So age quando a distancia entre centros dos grupos ~ 1 slot (nao corredor).
    """
    densos = [g for g in grupos if len(g) >= 3]
    if len(densos) < 2:
        return []

    idx = 1 if eixo_grupo == "y" else 0  # eixo perpendicular as fileiras
    i_ord = 0 if ao_longo == "x" else 1
    med_w = _mediana([_largura(c) for g in densos for c in g])
    med_h = _mediana([_altura(c) for g in densos for c in g])
    espessura = med_h if eixo_grupo == "y" else med_w

    # centros dos grupos ao longo do eixo perpendicular
    infos = []
    for g in densos:
        refs = [_centro(c)[idx] for c in g]
        infos.append((_mediana(refs), g))
    infos.sort(key=lambda t: t[0])

    # espacamento tipico entre fileiras ocupadas vizinhas (curto = vagas encostadas)
    gaps = [infos[i + 1][0] - infos[i][0] for i in range(len(infos) - 1)]
    gaps_curtos = [g for g in gaps if g < 2.2 * espessura]
    passo_fileira = _mediana(gaps_curtos) if gaps_curtos else espessura * 1.2

    livres = []
    for (c0, g0), (c1, g1) in zip(infos, infos[1:]):
        dist = c1 - c0
        # uma fileira vazia entre duas ocupadas: dist ~ 2 * passo_fileira
        if not (1.6 * passo_fileira <= dist <= 2.4 * passo_fileira):
            continue
        y_or_x = 0.5 * (c0 + c1)
        # grade ao longo: uniao das posicoes dos dois grupos
        coords = sorted([_centro(c)[i_ord] for c in g0 + g1])
        passo = _passo_vizinhos(g0, ao_longo=ao_longo) or _passo_vizinhos(g1, ao_longo=ao_longo)
        if passo <= 1:
            continue
        # amostra posicoes candidatas a cada passo entre min e max
        inicio, fim = coords[0], coords[-1]
        n_slots = int(round((fim - inicio) / passo)) + 1
        n_slots = max(1, min(40, n_slots))
        for k in range(n_slots):
            t = k / max(1, n_slots - 1) if n_slots > 1 else 0.5
            pos = inicio + t * (fim - inicio)
            if ao_longo == "x":
                cx, cy = pos, y_or_x
            else:
                cx, cy = y_or_x, pos
            livres.append([cx - med_w / 2, cy - med_h / 2, cx + med_w / 2, cy + med_h / 2])
    return livres


def _livres_no_eixo(ocupadas, eixo_grupo, ao_longo, max_vagas, max_gap_fator):
    grupos = _agrupar(ocupadas, eixo=eixo_grupo)
    livres = []
    for g in grupos:
        if len(g) >= 3:
            livres.extend(
                _lacunas_restritas(g, ao_longo=ao_longo, max_vagas=max_vagas, max_gap_fator=max_gap_fator)
            )
    livres.extend(_projetar_fileiras_paralelas(grupos, ao_longo=ao_longo, eixo_grupo=eixo_grupo))
    densos = sum(1 for g in grupos if len(g) >= 3)
    score = densos * 10 + sum(len(g) for g in grupos if len(g) >= 3) + len(livres) * 2
    return score, livres


def inferir_vagas_livres_restrito(ocupadas_xyxy, max_vagas=4, max_gap_fator=2.55):
    """Estima vagas livres [x1,y1,x2,y2] com um unico eixo dominante."""
    if len(ocupadas_xyxy) < 3:
        return []

    score_h, livres_h = _livres_no_eixo(ocupadas_xyxy, "y", "x", max_vagas, max_gap_fator)
    score_v, livres_v = _livres_no_eixo(ocupadas_xyxy, "x", "y", max_vagas, max_gap_fator)
    livres = livres_h if score_h >= score_v else livres_v

    livres = [lv for lv in livres if all(_iou(lv, oc) < 0.12 for oc in ocupadas_xyxy)]
    return _nms(livres, limiar=0.3)


def linhas_yolo_carpk(bboxes_coco, w_img, h_img, pseudo_livres=True):
    """Gera linhas YOLO: classe 1 ocupadas + classe 0 pseudo-livres."""
    ocupadas = [coco_para_xyxy(bb) for bb in bboxes_coco]
    linhas = [xyxy_para_yolo(c, w_img, h_img, 1) for c in ocupadas]
    if pseudo_livres and ocupadas:
        for lv in inferir_vagas_livres_restrito(ocupadas):
            x1 = max(0.0, min(w_img - 1.0, lv[0]))
            y1 = max(0.0, min(h_img - 1.0, lv[1]))
            x2 = max(0.0, min(w_img, lv[2]))
            y2 = max(0.0, min(h_img, lv[3]))
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            linhas.append(xyxy_para_yolo([x1, y1, x2, y2], w_img, h_img, 0))
    return linhas
