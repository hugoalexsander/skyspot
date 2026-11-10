# SkySpot

Detecta vagas de estacionamento livres e ocupadas em imagens, com o YOLOv8 treinado no PKLot (classes `space-empty` e `space-occupied`).

## Instalação

Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Se o PowerShell bloquear o script, execute antes: `Set-ExecutionPolicy -Scope Process RemoteSigned`.

Para treinar com GPU NVIDIA, instale o PyTorch com CUDA (comando em [pytorch.org](https://pytorch.org/get-started/locally/)). Confira com:

```bash
invoke gpu
```

## Uso

Os comandos usam o [Invoke](https://www.pyinvoke.org/). Veja a lista com `invoke --list`.

```bash
# 1. Baixe o PKLot (Roboflow, versão raw, formato YOLOv8) e extraia um subconjunto
invoke preparar --zip PKLot.v1-raw.yolov8.zip

# 2. Teste o fluxo rapidamente (1 época em poucas imagens) e depois treine de verdade
#    (sem GPU: --device cpu; com pouca memória: --batch 8)
invoke testar
invoke treinar

# 3. Meça o mAP no conjunto de teste
invoke validar --pesos runs/detect/pklot/weights/best.pt

# 4. Gere a referência (vagas.json) e avalie livres e ocupadas
invoke pklot
invoke avaliar --pesos runs/detect/pklot/weights/best.pt

# 5. Melhor treino atual: PKLot completo + ACPDS (sem pseudo CARPK)
invoke preparar-pklot-acpds
invoke treinar-pklot-acpds
# pesos: runs/detect/pklot_acpds/weights/best.pt  (YOLOv8s, imgsz 960)

# 6. Detecte vagas (auto aplica filtros anti-FP)
invoke detectar --imagem img.jpg --pesos runs/detect/pklot_acpds/weights/best.pt --modo auto
```

O caminho dos pesos aparece no fim do treino. A imagem anotada é salva em `resultados/`: verde = livre, vermelho = ocupada.

No Windows, use `\` nos caminhos se necessário. No Linux, use `python3` se `python` não existir.

## Resultados

| Métrica | Valor |
|---|---|
| mAP@0,5 (treino combinado v2) | ~0,98–0,99 |
| Precisão, revocação e F1 (livre, PKLot 20 imgs, v2) | 93% / 100% / 0,96 |
| Precisão, revocação e F1 (ocupada) | 99% / 100% / 1,00 |
| FPS | ~9 |

## Limitações

O PKLot usa câmeras fixas elevadas, não drones. O conjunto de teste vem dos mesmos estacionamentos do treino, então o resultado tende a ser otimista. O modelo pode generalizar mal para outros estacionamentos e câmeras. O treino usa um subconjunto do PKLot.

## Dados

PKLot: Almeida et al. (2015), *PKLot – A robust dataset for parking lot classification*, Expert Systems with Applications. Licença CC BY 4.0.

## Autores

Matheus Fleury Mainardi e Hugo Alex Sander Oliveira Araujo (UFG).
