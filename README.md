# Testes de modelos de decisão tipada inspirados no Jev

Este repositório reúne pesquisas e experimentos locais com modelos que recebem um estado e respondem a perguntas de formato delimitado, como `choice`, `score` e `noul`. O foco prático é a demo Snake com Laya no Linux e um registro pequeno de execuções com Laya, Kev, SemIf e Mapika decider.

Este projeto não é o Jev, não é um SDK oficial da TypeSafe e não demonstra equivalência entre esses modelos. As notas distinguem resultados publicados por terceiros de inferências realmente executadas neste ambiente.

## Conteúdo

- [`AGENTS.md`](AGENTS.md): espinha dorsal, invariantes e regras de trabalho para agentes.
- [`memory.md`](memory.md): histórico operacional e descobertas recentes do projeto.
- [`sobre.md`](sobre.md): introdução ao Laya e ao Jev.
- [`alternativas-ao-jev.md`](alternativas-ao-jev.md): pesquisa documental sobre projetos relacionados e limites das evidências.
- [`DEMO.md`](DEMO.md): arquitetura, instalação e uso da demo Snake, além dos registros locais.
- [`snake_linux/`](snake_linux/): motor determinístico, adaptadores, política e interface de terminal.
- [`tests/`](tests/): testes de regras, política, carregamento, adaptadores, interface e gravações reais.
- [`resultados/`](resultados/): gravações JSONL selecionadas, relatórios e um índice dos registros.
- [`artefatos/snake-frame.svg`](artefatos/snake-frame.svg): prévia de um estado da demo.
- [`LICENSE.upstream`](LICENSE.upstream) e [`NOTICE.upstream`](NOTICE.upstream): atribuição e licença do código adaptado da demo original Laya-MLX.

## Modelos e referências externas

Estes são os projetos e arquivos de modelo associados às execuções registradas neste repositório:

| Teste | Código ou projeto | Modelo ou checkpoint usado |
| --- | --- | --- |
| Laya multilíngue | [Repositório Laya](https://github.com/NandhaKishorM/laya) | [Checkpoint multilíngue](https://huggingface.co/convaiinnovations/laya/tree/main/multilingual) |
| Kev-0.8B | [Repositório Kev](https://github.com/jaredpalmer/kev) | [Checkpoint Kev-0.8B](https://huggingface.co/jaredpalmer/kev-0.8b) |
| SemIf-4B Q4_K_M | [Projeto SemIf](https://github.com/TheoLeeCJ/SemIf) | [Qwen3.5-4B GGUF Q4_K_M](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/blob/main/Qwen_Qwen3.5-4B-Q4_K_M.gguf) |
| Mapika decider-2b v11 e decider-4b v2.1 | [Repositório decider](https://github.com/Mapika/decider) | [decider-2b](https://huggingface.co/Mapika/decider-2b) e [decider-4b](https://huggingface.co/Mapika/decider-4b) |

O teste SemIf também usa o [tokenizer Qwen3.5-4B na revisão fixada](https://huggingface.co/Qwen/Qwen3.5-4B/tree/851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a). Nesse backend, o SemIf pontua as opções usando os logits do Qwen congelado. O arquivo GGUF não é, por si só, um checkpoint de decisão treinado. Os detalhes das versões e das execuções estão em [`DEMO.md`](DEMO.md) e [`resultados/`](resultados/).

A demo Snake deste projeto foi adaptada da [demo original Laya Snake do Laya-MLX](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_DEMO.md), disponível no [repositório Laya-MLX](https://github.com/mizorewww/laya-mlx). A original usa MLX em Apple Silicon; esta adaptação executa PyTorch no Linux.

Para contexto sobre o sistema que inspirou o formato de decisões, consulte a [apresentação do Jev pela TypeSafe](https://typesafe.ai/blog/introducing-system-one-models-and-jev) e a [documentação da TypeSafe](https://docs.typesafe.ai/introduction.md). Laya, Kev e SemIf são projetos distintos, não o Jev nem implementações oficiais da TypeSafe.

## Testes automatizados

Requer Python 3.10 ou superior. Na raiz do repositório:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
USE_TF=0 .venv/bin/python -m pytest tests/ -q
```

Os testes usam agentes simulados para verificar o formato da API e os adaptadores, além de reproduzir as gravações incluídas. Isso não substitui uma inferência real com os pesos.

## Demo Snake

Para instalar os pesos do Laya e executar a demo, siga as instruções em [`DEMO.md`](DEMO.md). Os pesos, os ambientes virtuais e os clones dos projetos externos não são armazenados neste repositório. Kev, SemIf e Mapika decider são backends opcionais e também exigem seus respectivos arquivos e configurações locais. O teste local dos dois checkpoints decider está documentado em [`resultados/decider-v11-v2.1-snake-3.md`](resultados/decider-v11-v2.1-snake-3.md).

## Como interpretar os resultados

O Snake combina inferência do modelo com planejamento e proteção determinísticos. As propriedades do tabuleiro já são resumidas antes de chegar ao modelo, e o escudo pode substituir uma direção insegura. Portanto, a demo não mostra aprendizado autônomo de Snake nem mede, por si só, a qualidade geral de um modelo de decisão.

As gravações documentam execuções reais e curtas, não um benchmark controlado. Os tempos dependem de hardware, checkpoint, backend, quantização, aquecimento e configuração. Para comparar qualidade, use os mesmos exemplos rotulados, estado, perguntas e critérios, com revisão humana para decisões sensíveis.
