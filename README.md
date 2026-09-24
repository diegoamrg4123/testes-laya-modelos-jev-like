# Testes de modelos de decisão tipada inspirados no Jev

Este repositório reúne pesquisas e experimentos locais com modelos que recebem um estado e respondem a perguntas de formato delimitado, como `choice`, `score` e `noul`. O foco prático é a demo Snake com Laya no Linux e um registro pequeno de execuções com Laya, Kev e SemIf.

Este projeto não é o Jev, não é um SDK oficial da TypeSafe e não demonstra equivalência entre esses modelos. As notas distinguem resultados publicados por terceiros de inferências realmente executadas neste ambiente.

## Conteúdo

- [`sobre.md`](sobre.md): introdução ao Laya e ao Jev.
- [`alternativas-ao-jev.md`](alternativas-ao-jev.md): pesquisa documental sobre projetos relacionados e limites das evidências.
- [`DEMO.md`](DEMO.md): arquitetura, instalação e uso da demo Snake, além dos registros locais.
- [`snake_linux/`](snake_linux/): motor determinístico, adaptadores, política e interface de terminal.
- [`tests/`](tests/): testes de regras, política, carregamento, adaptadores, interface e gravações reais.
- [`resultados/`](resultados/): cinco gravações JSONL selecionadas e um índice dos registros.
- [`artefatos/snake-frame.svg`](artefatos/snake-frame.svg): prévia de um estado da demo.
- [`LICENSE.upstream`](LICENSE.upstream) e [`NOTICE.upstream`](NOTICE.upstream): atribuição e licença do código adaptado da demo original Laya-MLX.

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

Para instalar os pesos do Laya e executar a demo, siga as instruções em [`DEMO.md`](DEMO.md). Os pesos, os ambientes virtuais e os clones dos projetos externos não são armazenados neste repositório. Kev e SemIf são backends opcionais e também exigem seus respectivos arquivos e configurações locais.

## Como interpretar os resultados

O Snake combina inferência do modelo com planejamento e proteção determinísticos. As propriedades do tabuleiro já são resumidas antes de chegar ao modelo, e o escudo pode substituir uma direção insegura. Portanto, a demo não mostra aprendizado autônomo de Snake nem mede, por si só, a qualidade geral de um modelo de decisão.

As gravações documentam execuções reais e curtas, não um benchmark controlado. Os tempos dependem de hardware, checkpoint, backend, quantização, aquecimento e configuração. Para comparar qualidade, use os mesmos exemplos rotulados, estado, perguntas e critérios, com revisão humana para decisões sensíveis.

## Arquivos locais ignorados

O `.gitignore` exclui ambientes virtuais, caches, arquivos de modelo, clones externos, logs e gravações não selecionadas. Os arquivos mantidos em `resultados/` são pequenos registros usados na documentação e nos testes. Não envie pesos nem credenciais ao repositório.
