# Teste local do Tev1-0.8B-experimental no Snake

## Objetivo e escopo

Teste curto para confirmar que `togethercomputer/Tev1-0.8B-experimental` carrega localmente e escolhe opções estruturadas nos três tipos de pergunta usados pela demo: direção, rota segura e alcance da comida. Não é benchmark comparativo, teste de acurácia nem integração permanente de backend.

O checkpoint é um fine-tune experimental de Qwen3.5-0.8B que responde com uma letra de opção. Ele é Jev-inspired, mas continua sendo um modelo autoregressivo com cabeça de linguagem, não um runtime Jev não autoregressivo. O card não declara suporte multilíngue abrangente. A licença das weights está sendo finalizada, portanto o teste não redistribui os pesos.

## Artefato e ambiente

| Campo | Valor |
| --- | --- |
| Repositório | [togethercomputer/Tev1-0.8B-experimental](https://huggingface.co/togethercomputer/Tev1-0.8B-experimental) |
| Revisão do Hub | `6bb2dff14b38fea90ddb14d870166ccaf77374e9` |
| Arquivo | `models/tev1-0.8b-experimental/model.safetensors-00001-of-00001.safetensors` |
| Tamanho local | 1.746.942.600 bytes |
| SHA-256 local | `197de1eb141b93984a15e2def8a840726c5dd1349ea4e1b73b8e53abb9257408` |
| Python | 3.11.16 |
| PyTorch / Transformers | `2.14.0+cu130` / `5.17.0` |
| Dispositivo | NVIDIA GeForce RTX 2050, 4 GiB VRAM |
| Precisão carregada | FP16 |

Os pesos foram baixados antes da partida. A inferência foi local e offline. A memória CUDA alocada após a carga foi 1.656 MiB e o pico observado foi 1.713,6 MiB. Transformers usou implementações PyTorch de referência para `causal_conv1d` e `flash-linear-attention`, pois os kernels opcionais não estão instalados.

## Método

O script `scripts/test_tev1_snake.py` prepara uma decisão para cada pergunta, conforme o formato do card: estado, pergunta e opções rotuladas de A em diante. Faz três chamadas ao modelo por movimento, usando a mesma string de estado e instruções de pergunta do `LayaPolicy`. A resposta gerada precisa ser exatamente uma das letras listadas.

Para encaixar o teste na política Snake sem alterar o código de produção, o script também lê os logits do primeiro token e normaliza por softmax somente entre as letras das opções. Essa distribuição é uma conversão experimental dos logits do modelo, não uma distribuição tipada emitida pelo Tev1 e não uma confiança calibrada. O teste interrompe se a letra gerada divergir da opção de maior logit normalizado.

Execução realizada a partir da raiz do projeto:

```bash
PYTHONPATH="$PWD" USE_TF=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=4 \
  .venv/bin/python scripts/test_tev1_snake.py
```

Jogo: tabuleiro 8 × 6, semente 7, escudo de segurança ativo, três decisões reais. Não houve aquecimento separado antes dessas decisões. A primeira chamada inclui efeitos de inicialização de kernels e não deve ser tomada como latência estável.

## Resultado observado

| Passo | Pontuação antes da ação | Proposta | Executada | Intervenção | Latência reportada pela política |
| --- | ---: | --- | --- | --- | ---: |
| 1 | 0 | LEFT | LEFT | Não | 1.598,93 ms |
| 2 | 1 | DOWN | DOWN | Não | 498,83 ms |
| 3 | 1 | DOWN | DOWN | Não | 570,34 ms |

A execução terminou viva, com pontuação 1 e corpo de tamanho 7. A média das três latências foi 889,37 ms por decisão; a média dos dois últimos passos foi 534,59 ms. Cada decisão inclui três chamadas ao modelo, uma para cada pergunta.

Probabilidades direcionais convertidas dos logits entre as quatro opções:

| Passo | UP | DOWN | LEFT | RIGHT |
| --- | ---: | ---: | ---: | ---: |
| 1 | 0,0083 | 0,0442 | 0,9019 | 0,0456 |
| 2 | 0,0048 | 0,7522 | 0,2155 | 0,0274 |
| 3 | 0,0042 | 0,7043 | 0,0070 | 0,2845 |

Esses números são preferências relativas calculadas pelo harness e não probabilidades de acerto ou segurança. `DEAD-END RISK` e `FOOD REACHABLE` também são respostas do modelo sobre atributos preparados pelo planejador, não estimativas calibradas de risco futuro.

## Limitações

- Três passos em uma única semente só confirmam carga e execução local, não qualidade geral nem comparação com outros modelos.
- O card identifica o modelo como experimental e afirma que comportamento multilíngue, calibração e robustez fora de distribuição não foram avaliados de forma abrangente.
- O adaptador no script é apenas um harness de teste; o backend não foi adicionado ao CLI e as probabilidades normalizadas não fazem parte da interface oficial do Tev1.
- A licença das weights ainda está sendo finalizada. Não redistribuir nem incorporar os pesos ao repositório.
- As latências dependem da máquina, da versão das bibliotecas e do caminho de kernels usado.
