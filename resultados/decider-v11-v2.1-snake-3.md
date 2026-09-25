# Teste local do decider no Snake

## Escopo

Teste curto dos dois checkpoints solicitados do projeto [Mapika/decider](https://github.com/Mapika/decider), usando a mesma demo Snake deste repositório. O objetivo foi confirmar carregamento, inferência tipada real e comportamento do adaptador, não estimar a qualidade geral dos modelos.

## Ambiente

| Campo | Valor observado |
| --- | --- |
| Sistema | Linux 7.0.0-34-generic, x86_64 |
| Python do jogo | 3.11.16 |
| PyTorch | 2.14.0+cu130 |
| Transformers | 5.17.0 |
| decider-ai | 1.5.0 |
| GPU disponível | NVIDIA GeForce RTX 2050, 4096 MiB |
| Dispositivo usado no teste | CPU, para permitir a execução dos dois checkpoints sem disputar os 4 GiB de VRAM |
| Aceleração linear | O Transformers informou fallback para a implementação PyTorch de referência; a execução foi válida, mas mais lenta |

O ambiente do decider foi separado em `.venv-decider`, reutilizando as bibliotecas locais do `.venv` por `PYTHONPATH`. Os pesos foram baixados antes das partidas.

## Checkpoints

| Modelo | Revisão do Hub | Versão no `decider_config.json` | Tamanho de `model.safetensors` | SHA-256 LFS do Hub |
| --- | --- | --- | ---: | --- |
| decider-2b v11 | `533964dae8be954c5b5e19fa4948e48408094c1e` | `2b-v11` | 3.763.692.048 bytes | `acaef2228b134dcdc20cad4ee79219482c927ec819aa3687b9b8a575c338817f` |
| decider-4b v2.1 | `eb5fbdfc9448473ec25e399882912863afbdb70e` | `4b-v2.1` | 8.411.558.400 bytes | `ee8ce585b3cedd93206dd149b09b4bdd683174874211f85c77a36090b90c9fdd` |

Os diretórios locais foram `models/decider-2b-v11` e `models/decider-4b-v2.1`. Ambos têm contexto configurado para até 32k tokens, embora o teste tenha usado entradas curtas.

## Método

Configuração comum:

| Campo | Valor |
| --- | --- |
| Tabuleiro | 8 × 6 |
| Semente | 7 |
| Comprimento inicial | 6 |
| Aquecimento | 6 decisões fora da medição |
| Partida medida | 3 passos, máximo de velocidade |
| Escudo | Ativo |
| Perguntas | `move` como `choice`; `risk` e `food` como `noul` |
| Estado | Texto compacto com rota segura e alcance atual da comida |
| Registro | Estado do tabuleiro antes da ação, distribuição original, ação proposta, ação executada e resumo |

A pergunta `move` recebeu quatro alternativas, `UP`, `DOWN`, `LEFT` e `RIGHT`, com descrições derivadas pelo planejador determinístico. O modelo não recebeu o tabuleiro cru. A pergunta `risk` perguntou se havia rota segura e `food` perguntou se a comida era alcançável pelas casas vazias atuais.

Comando efetivamente usado, trocando apenas o checkpoint e o arquivo de saída:

```bash
PYTHONPATH="$PWD/.venv/lib/python3.11/site-packages" \
USE_TF=0 .venv-decider/bin/python -m snake_linux \
  --backend decider --model models/decider-2b-v11 \
  --decider-device cpu --headless --width 8 --height 6 \
  --seed 7 --steps 3 --max-speed \
  --record resultados/decider-2b-v11-snake-3.jsonl
```

## Resultado observado

| Modelo | Passos | Inferências medidas | Comida | Mortes | Intervenções | Média por inferência | Ritmo da partida |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| decider-2b v11 | 3 | 3 | 1 | 0 | 0 | 14.209,3 ms | 0,0704 passos/s |
| decider-4b v2.1 | 3 | 3 | 1 | 0 | 0 | 37.113,1 ms | 0,0269 passos/s |

As duas execuções propuseram e executaram a mesma sequência: `LEFT`, `DOWN`, `DOWN`. No primeiro frame, o 2B deu `LEFT` com 0,6819 e o 4B com 0,9532. As probabilidades registradas para as quatro direções somaram 1,0 dentro do arredondamento dos arquivos JSONL.

Arquivos completos:

* [`decider-2b-v11-snake-3.jsonl`](decider-2b-v11-snake-3.jsonl)
* [`decider-4b-v2.1-snake-3.jsonl`](decider-4b-v2.1-snake-3.jsonl)

A validação reproduziu os três estados gravados com `SnakeGame`, confirmou que cada frame representa o estado anterior à ação, verificou as três distribuições e confirmou que todas as ações executadas estavam na lista segura. O teste automatizado do adaptador também passou.

## Limitações

* Três passos não permitem concluir qual modelo é melhor.
* O escudo determinístico reduz o risco de morte e pode substituir uma proposta insegura. Neste teste não houve intervenção.
* Os tempos são de CPU com fallback PyTorch e não são comparação de desempenho de CUDA.
* As probabilidades de `risk` e `food` são julgamentos tipados sobre os atributos fornecidos pelo planejador, não probabilidades calibradas de morte ou de vitória no Snake.
* Os números publicados pelo projeto, incluindo 0,752 para o 2B e 0,784 para o 4B no regression set, não foram reproduzidos aqui. Eles são resultados de avaliação do autor em outro conjunto e não devem ser confundidos com esta partida.
