# Documentação do SkySpot

## Objetivo

Detectar vagas de estacionamento livres e ocupadas em imagens, com o YOLOv8 treinado no PKLot.

## Pipeline

```
ZIP do PKLot → subconjunto (invoke preparar) → treino (invoke treinar) → best.pt
best.pt + imagem → vagas livres (verde) e ocupadas (vermelho) → resultados\
```

## Estrutura

```
skyspot/
├── src/
│   ├── preparar_pklot.py      # extrai o subconjunto e cria pklot.yaml
│   ├── pklot_para_vagas.py    # converte as anotações do teste em vagas.json
│   ├── ocupacao.py            # detecta vagas em uma imagem
│   └── avaliar_ocupacao.py    # precisão, revocação, F1 e FPS
├── pklot/                     # dados (não versionado)
├── resultados/                # saídas geradas e vagas.json (referência da avaliação)
├── tasks.py                   # comandos do Invoke
├── requirements.txt
└── README.md
```

## Dados

- **PKLot** (Roboflow, versão raw, formato YOLOv8): imagens de 1280×720 de câmeras fixas elevadas.
- Classes: `space-empty` (0) e `space-occupied` (1).
- O treino usa 2.500 imagens, a validação 500 e o teste todas as 1.242 do conjunto de teste.
- Cada imagem tem um `.txt` em `labels\`. Cada linha é uma vaga: `classe x_centro y_centro largura altura`, em fração do tamanho da imagem.

## Comandos

| Comando | Função |
|---|---|
| `invoke instalar` | Instala as dependências. |
| `invoke preparar --zip <arquivo>` | Extrai o subconjunto do PKLot e cria `pklot\pklot.yaml`. |
| `invoke treinar` | Treina o YOLOv8n a partir dos pesos do COCO. Opções: `--epocas`, `--imgsz`, `--batch`, `--device`. |
| `invoke validar --pesos <best.pt>` | Calcula o mAP no conjunto de teste (`yolo val`). |
| `invoke pklot` | Gera `resultados\vagas.json` com 20 imagens do teste. |
| `invoke detectar --imagem <img> --pesos <best.pt>` | Salva a imagem anotada em `resultados\`. |
| `invoke avaliar --pesos <best.pt>` | Calcula precisão, revocação, F1 e FPS. Grava `resultados\metricas.csv`. |

## Imagem de saída

| Cor | Significado |
|---|---|
| Verde | Vaga livre |
| Vermelho | Vaga ocupada |

## Formato de `resultados/vagas.json`

```json
{
  "img001.jpg": [
    {"caixa": [x1, y1, x2, y2], "real": "livre"},
    {"caixa": [x1, y1, x2, y2], "real": "ocupada"}
  ]
}
```

## Avaliação

- **Detector:** mAP@0,5 e mAP@0,5:0,95 por classe, com `invoke validar`.
- **Vagas:** cada vaga prevista é pareada com uma vaga de referência da mesma classe quando o IoU é de pelo menos 0,5. O script calcula precisão, revocação e F1 de `livre` e de `ocupada`.
- **Velocidade:** imagens por segundo.

## Parâmetros

| Parâmetro | Padrão | Efeito |
|---|---|---|
| `conf` | 0,25 | Confiança mínima. Valor menor detecta mais vagas e mais erros. |
| `imgsz` | 640 | Tamanho da imagem de entrada. |
| `epocas` | 20 | Épocas de treino. |
| `batch` | 16 | Imagens por lote. Reduza para 8 se faltar memória da GPU. |

## Limitações

- O PKLot usa câmeras fixas elevadas, não drones.
- O conjunto de teste vem dos mesmos estacionamentos do treino, então o resultado tende a ser otimista.
- O modelo pode generalizar mal para outros estacionamentos e câmeras.
- O treino usa um subconjunto do PKLot e o modelo menor (`yolov8n`).
- Sem testes noturnos ou com chuva forte.

## Referência

Almeida, P. et al. *PKLot – A robust dataset for parking lot classification*. Expert Systems with Applications, 42(11):4937-4949, 2015. Licença CC BY 4.0.
