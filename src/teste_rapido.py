"""Teste rápido do fluxo: dados, GPU, treino de 1 época em poucas imagens e detecção.

Confirma que o treino real vai funcionar. Não gera um modelo útil.
Grava em runs/.../teste e resultados/teste (não mexe no treino real, que usa o nome "pklot").

Uso:
    python src/teste_rapido.py --device 0
"""
import argparse
import sys
import time
from pathlib import Path

import cv2

from ocupacao import desenhar, detectar_vagas

LARGURA = 62


def titulo(texto):
    print("\n" + "=" * LARGURA)
    print(f" {texto}")
    print("=" * LARGURA)


def etapa(n, texto):
    print(f"\n[{n}/4] {texto}")
    print("-" * LARGURA)


def ok(texto):
    print(f"  [OK]    {texto}")


def info(texto):
    print(f"          {texto}")


def erro(texto, dica):
    print(f"  [ERRO]  {texto}")
    print(f"          Como corrigir: {dica}")
    titulo("RESULTADO: FALHOU")
    raise SystemExit(1)


def contar(pasta):
    imagens = [f for f in (pasta / "images").glob("*") if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    rotulos = list((pasta / "labels").glob("*.txt"))
    return len(imagens), len(rotulos)


def main():
    import torch
    from ultralytics import YOLO

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    p = argparse.ArgumentParser()
    p.add_argument("--device", default="0", help='"0" para a GPU ou "cpu"')
    p.add_argument("--fracao", default=0.02, type=float, help="fração das imagens de treino")
    args = p.parse_args()
    inicio = time.perf_counter()

    titulo("SKYSPOT - TESTE RÁPIDO DO FLUXO")
    info(f"Treino de 1 época com {args.fracao:.0%} das imagens. Não gera um modelo útil.")

    # 1. Dados
    etapa(1, "DADOS")
    if not Path("pklot/pklot.yaml").exists():
        erro("Arquivo pklot/pklot.yaml não encontrado.", "invoke preparar --zip <arquivo do PKLot>")
    for parte in ("train", "valid", "test"):
        n_img, n_rot = contar(Path("pklot") / parte)
        if n_img == 0 or n_img != n_rot:
            erro(f"pklot/{parte}: {n_img} imagens e {n_rot} rótulos.", "invoke preparar --zip <arquivo do PKLot>")
        ok(f"{parte:<6} {n_img:>5} imagens | {n_rot:>5} rótulos")

    # 2. GPU
    etapa(2, "PLACA DE VÍDEO")
    if args.device == "cpu":
        ok("Usando a CPU (o treino real será lento).")
    elif torch.cuda.is_available():
        ok(f"{torch.cuda.get_device_name(0)} (PyTorch {torch.__version__})")
    else:
        erro("O PyTorch deste Python não enxerga a GPU.", "invoke gpu  (ou use: invoke testar --device cpu)")

    # 3. Treino
    etapa(3, "TREINO DE 1 ÉPOCA")
    info("O progresso do Ultralytics aparece abaixo.\n")
    modelo = YOLO("yolov8n.pt")
    modelo.train(
        data="pklot/pklot.yaml", epochs=1, imgsz=640, batch=8, fraction=args.fracao,
        device=args.device, name="teste", exist_ok=True, plots=False,
    )
    pesos = Path(modelo.trainer.best)
    if not pesos.exists():
        pesos = Path(modelo.trainer.last)
    print()
    ok(f"Pesos gerados: {pesos}")

    # 4. Detecção
    etapa(4, "DETECÇÃO EM UMA IMAGEM DE TESTE")
    imagem_arq = sorted(Path("pklot/test/images").iterdir())[0]
    imagem = cv2.imread(str(imagem_arq))
    # Com 1 época o modelo é fraco. Confiança baixa só para ver alguma saída.
    livres, ocupadas = detectar_vagas(YOLO(str(pesos)), imagem, conf=0.01)
    saida = Path("resultados/teste")
    saida.mkdir(parents=True, exist_ok=True)
    destino = saida / f"{imagem_arq.stem[:40]}_teste.png"
    cv2.imwrite(str(destino), desenhar(imagem, livres, ocupadas))
    ok(f"Vagas livres: {len(livres)} | ocupadas: {len(ocupadas)}")
    ok(f"Imagem salva em: {destino}")

    titulo("RESULTADO: TUDO CERTO")
    info(f"Tempo total: {time.perf_counter() - inicio:.0f} s")
    info("O modelo deste teste é fraco (1 época). Isso é esperado.")
    info("Próximo passo:  invoke treinar")


if __name__ == "__main__":
    main()
