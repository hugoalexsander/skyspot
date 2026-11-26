# Anota��o de vagas livres (CARPK)

O modelo ainda marca via/canteiro como vaga livre porque o CARPK **n�o tem** slots vazios reais anotados. Este pacote serve para voc� marcar s� as vagas pintadas e vazias.

## Preparar o pacote

Na raiz do `skyspot`:

```bash
invoke exportar-anotacao-carpk
```

Isso cria `anotacao/carpk_livres/` com ~100 imagens. Os labels j� trazem **ocupadas** (classe `1`). Voc� s� acrescenta **livres** (classe `0`).

## Classes

| id | nome |
|---|---|
| 0 | space-empty |
| 1 | space-occupied |

Arquivo `classes.txt` precisa estar em `labels/classes.txt` (o LabelImg procura ali).

## Como anotar

1. Abra o LabelImg (ou Roboflow) apontando para `anotacao/carpk_livres/images`.
2. Formato YOLO; pasta de labels = `anotacao/carpk_livres/labels`.
3. **Marque** ret�ngulos s� em vagas com pintura vis�vel e **sem carro**.
4. **N�o marque**: via, seta no asfalto, canteiro, cal�ada, vaga parcial fora do quadro.
5. N�o apague as caixas de ocupadas j� existentes (classe 1), a menos que estejam erradas.

## Depois de anotar

```bash
invoke montar-combinado-v3
invoke treinar-combinado-v3
```

O pipeline recusa montar se houver menos de ~10 caixas `space-empty` no total.
