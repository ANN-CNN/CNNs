# CNNs para detecção de objetos

![Ilustração de um neurônio](img/neuronio.png)

Repositório de apoio a um trabalho acadêmico sobre treinamento e avaliação de
modelos de visão computacional para detecção de objetos. Os notebooks reúnem
experimentos com Faster R-CNN, SSD e diferentes versões do YOLO, além de uma
rotina para comparar métricas.

## Conteúdos

| Arquivo                                                                              | Descrição                                                                    |
| ------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------- |
| [`src/FasterRCNN.ipynb`](src/FasterRCNN.ipynb)                                       | Preparação dos dados, treinamento, inferência e avaliação com Faster R-CNN.  |
| [`src/SSD.ipynb`](src/SSD.ipynb)                                                     | Preparação dos dados, treinamento, inferência e avaliação com SSD.           |
| [`src/Yolo5.ipynb`](src/Yolo5.ipynb)                                                 | Experimento de detecção com YOLOv5.                                          |
| [`src/Yolo8.ipynb`](src/Yolo8.ipynb)                                                 | Experimento de detecção com YOLOv8.                                          |
| [`src/Yolo11.ipynb`](src/Yolo11.ipynb)                                               | Experimento de detecção com YOLO11.                                          |
| [`src/Avaliacao_Metricas_Faster_SSD.ipynb`](src/Avaliacao_Metricas_Faster_SSD.ipynb) | Cálculo e visualização de métricas para os resultados de Faster R-CNN e SSD. |
| [`src/utils.ipynb`](src/utils.ipynb)                                                 | Utilitários e verificações auxiliares para os experimentos.                  |
| [`src/yolo_metrics.py`](src/yolo_metrics.py)                                         | Implementação reutilizável de métricas e gráficos no estilo YOLOv5.          |
| [`conversion/COCO to YOLO.pdf`](conversion/COCO%20to%20YOLO.pdf)                     | Guia de conversão de anotações do formato COCO para YOLO.                    |

## Como executar

Os notebooks de treinamento foram preparados para o Google Colab e usam o
Google Drive para acessar os dados e armazenar resultados.

1. Abra o notebook desejado no Google Colab.
2. Se o experimento usar aceleração, selecione uma GPU em **Ambiente de
   execução > Alterar o tipo de ambiente de execução**.
3. Execute as células na ordem. Autorize o acesso ao Google Drive quando
   solicitado.
4. Antes do treinamento ou da avaliação, ajuste nos notebooks os caminhos dos
   arquivos, conjuntos de dados e pastas de resultados para a sua estrutura no
   Drive.
5. Confira o formato esperado pelo modelo. Para conversão de anotações COCO
   para YOLO, consulte o guia em [`conversion/`](conversion/).

As dependências são instaladas nas células de cada notebook, pois variam entre
os modelos. O repositório não possui um arquivo único de dependências para
execução local.

## Métricas

O módulo [`src/yolo_metrics.py`](src/yolo_metrics.py) reúne funções para calcular
IoU, identificar verdadeiros positivos por limiar de IoU, calcular Average
Precision e gerar curvas Precision-Recall, Precision-Confidence,
Recall-Confidence e F1-Confidence. A classe `ConfusionMatrix` também permite
acumular e visualizar uma matriz de confusão com uma classe de _background_.

## Estrutura

```text
.
├── conversion/   # Material de conversão de anotações
├── img/          # Imagens usadas na documentação
└── src/          # Notebooks e módulo de métricas
```
