"""Comandos do SkySpot. Veja a lista com: invoke --list

Todos os comandos usam o Python e o yolo da pasta .venv do projeto,
mesmo que o invoke tenha sido iniciado por outro Python.
"""
import os
import sys
from pathlib import Path

from invoke import task

RAIZ = Path(__file__).resolve().parent
BIN = RAIZ / ".venv" / ("Scripts" if os.name == "nt" else "bin")
EXT = ".exe" if os.name == "nt" else ""
_py, _yolo = BIN / f"python{EXT}", BIN / f"yolo{EXT}"
PY = f'"{_py if _py.exists() else sys.executable}"'
YOLO = f'"{_yolo}"' if _yolo.exists() else "yolo"


def rodar(c, comando):
    """Executa o comando na raiz do projeto, com saída em UTF-8 (evita caracteres quebrados no Windows)."""
    with c.cd(str(RAIZ)):
        c.run(comando, env={"PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}, encoding="utf-8")


@task
def instalar(c):
    """Instala as dependências na .venv."""
    rodar(c, f"{PY} -m pip install -r requirements.txt")


@task
def gpu(c):
    """Mostra se o PyTorch da .venv enxerga a GPU (CUDA)."""
    codigo = (
        "import sys, torch; ok = torch.cuda.is_available(); "
        "print('Python:', sys.executable); "
        "print('GPU disponível:', ok); "
        "print('Placa:', torch.cuda.get_device_name(0) if ok else 'nenhuma (PyTorch só de CPU ou sem driver)')"
    )
    rodar(c, f'{PY} -c "{codigo}"')


@task
def preparar(c, zip, treino=2500, valid=500):
    """Extrai um subconjunto do ZIP do PKLot. Ex.: invoke preparar --zip PKLot.v1-raw.yolov8.zip"""
    rodar(c, f'{PY} src/preparar_pklot.py --zip "{zip}" --treino {treino} --valid {valid}')


@task
def testar(c, device="0"):
    """Teste rápido do fluxo (1 época em poucas imagens). Sem GPU: invoke testar --device cpu"""
    rodar(c, f"{PY} src/teste_rapido.py --device {device}")


@task
def treinar(c, epocas=20, imgsz=640, batch=16, device="0"):
    """Treina o YOLOv8 no PKLot. Sem GPU: invoke treinar --device cpu"""
    rodar(
        c,
        f"{YOLO} detect train model=yolov8n.pt data=pklot/pklot.yaml epochs={epocas} "
        f"imgsz={imgsz} batch={batch} device={device} name=pklot exist_ok=True",
    )


@task
def validar(c, pesos, device="0"):
    """Mede mAP no conjunto de teste. Ex.: invoke validar --pesos runs/detect/pklot/weights/best.pt"""
    rodar(c, f'{YOLO} detect val model="{pesos}" data=pklot/pklot.yaml split=test device={device}')


@task
def pklot(c, pasta="pklot/test", maximo=20):
    """Converte as anotações do teste para vagas.json. Ex.: invoke pklot"""
    rodar(c, f'{PY} src/pklot_para_vagas.py --pasta "{pasta}" --maximo {maximo}')


@task
def detectar(c, imagem, pesos):
    """Detecta as vagas em uma imagem. Ex.: invoke detectar --imagem img.jpg --pesos best.pt"""
    rodar(c, f'{PY} src/ocupacao.py --imagem "{imagem}" --pesos "{pesos}"')


@task
def avaliar(c, pesos, imagens="pklot/test/images"):
    """Avalia vagas livres e ocupadas. Ex.: invoke avaliar --pesos best.pt"""
    rodar(c, f'{PY} src/avaliar_ocupacao.py --imagens "{imagens}" --pesos "{pesos}"')
