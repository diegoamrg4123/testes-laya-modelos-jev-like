# Laya Snake no Linux

Para agentes que continuarem este projeto, as regras locais estão em [AGENTS.md](AGENTS.md); este guia concentra uso e explicações para leitura humana.

Demo jogável do Snake comandado por **inferências reais** do checkpoint multilíngue Laya, com adaptação do [Laya-MLX Snake](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_DEMO.md) para PyTorch em Linux. O MLX é específico de Apple Silicon e **não é usado aqui**. O código do tabuleiro, a proteção por ciclo, a interface Rich e a estrutura da gravação foram adaptados do projeto original, commit `0a859518634112655cb97c745dbf04f5191aaf13`, Apache 2.0. Consulte `LICENSE.upstream` e `NOTICE.upstream`.[1]

## Rodar agora

Os pesos do Laya e o ambiente virtual são necessários para executar novas inferências, mas **não são incluídos neste repositório**. Siga [Instalar de novo em outro Linux](#instalar-de-novo-em-outro-linux) antes da primeira execução. Abra um terminal com pelo menos **104 colunas × 35 linhas** e, na raiz do repositório, execute:

```bash
USE_TF=0 .venv/bin/python -m snake_linux
```

Teclas: `Espaço` pausa, `↑` e `↓` alteram a velocidade alvo, `R` reinicia, `Q` sai. O alvo inicial é 12 decisões por segundo, mas o limite real depende da inferência e da renderização. `--max-speed` dispensa a espera artificial, `--width 8 --height 6` usa tabuleiro menor e `--unassisted` executa a escolha bruta do modelo sem correção, podendo morrer cedo. Sem TTY, use `--headless`.[1]

```bash
USE_TF=0 .venv/bin/python -m snake_linux --headless --steps 120 --max-speed \
  --record resultados/minha-partida.jsonl
```

O arquivo JSONL contém metadados, cada estado **antes** do movimento, probabilidades reais da chamada ao Laya e resumo final. A gravação não substitui inferência por reprodução. A gravação e a prévia SVG já produzidas estão em `resultados/snake-120.jsonl` e `artefatos/snake-frame.svg`. Para gerar uma nova prévia de um frame real:

```bash
PYTHONPATH=. .venv/bin/python scripts/gerar_preview.py \
  resultados/snake-120.jsonl artefatos/minha-previa.svg --frame 10
```

## Como o Laya participa da jogada

```text
SnakeGame + ciclo de segurança determinístico
        ↓
Estado resumido: existe rota segura? a comida é alcançável?
Pergunta choice: qual direção entre UP, DOWN, LEFT, RIGHT?
Pergunta noul: existe uma rota segura?
Pergunta noul: comida alcançável pelas casas vazias atuais?
        ↓  uma chamada agent.predict(state, questions) por movimento
Laya: distribuição das quatro direções + duas probabilidades
        ↓
Escudo opcional: se a preferência do Laya é insegura,
executa a melhor direção admissível segundo as probabilidades originais
        ↓
SnakeGame.step() atualiza o tabuleiro
```

O planejador **já fornece ao modelo** as propriedades `Blocked`, `Unsafe`, `Safe`, `Best route` e `Eat food now` para cada direção. Isso demonstra uma decisão neural assistida por atributos, **não** que Laya aprendeu Snake ou lê o tabuleiro cru. O escudo usa um ciclo hamiltoniano para evitar atravessar o corpo e não ultrapassar a comida no percurso seguro. O jogo mantém separadas a direção `proposed` pelo modelo e a `executed` pelo escudo.[1]

`DEAD-END RISK` é `1 - P(existe rota segura)` retornado pelo modelo, **não** uma probabilidade calibrada de morte. `FOOD REACHABLE` avalia a conectividade das casas vazias **atuais**, não prevê o futuro. As barras são probabilidades fornecidas pelo Laya, e a UI mostra `SHIELD` quando intervém. `NETWORK OFFLINE` significa carregamento local com Hugging Face em modo offline durante a partida, **não** que a placa de rede foi desligada.[1]

### Resultado observado nesta máquina

O checkpoint `convaiinnovations/laya`, subpasta `multilingual`, foi carregado por PyTorch com CUDA na RTX 2050. Em uma execução headless de 120 passos com tabuleiro padrão e escudo, houve 120 chamadas reais ao modelo, 0 mortes e 0 intervenções; média de inferência de aproximadamente 26,0 ms por movimento e 37,8 passos por segundo sem pacing. Esses valores são **uma execução local**, não benchmark comparativo. O primeiro frame gravado atribuiu `UP: 0.0819`, `DOWN: 0.4016`, `LEFT: 0.1377` e `RIGHT: 0.3788` e executou `DOWN`. Na gravação menor, de 20 passos em tabuleiro 8 × 6, o resultado foi 2 alimentos, 0 mortes e 20 inferências. A demonstração não valida acurácia nem calibração em Snake; números mudam com hardware, tabuleiro, cargas e versão.[2]

## Testar a mesma cobra com Kev e SemIf

O jogo agora aceita `--backend laya` (padrão), `--backend kev` e `--backend semif`.
O tabuleiro, as três perguntas (`move`, `risk` e `food`) e o escudo de segurança são os mesmos. O Kev recebe uma requisição HTTP por jogada, com as três perguntas; o SemIf pontua três perguntas sobre o mesmo estado com a função `score_shared` do projeto original. As probabilidades do SemIf são condicionais às opções e **não calibradas** para este jogo. Os valores de risco e alcance são opiniões do modelo sobre atributos do planejador, não estatísticas de segurança.[5][6]

**Kev-0.8B:** o código do Kev, o ambiente virtual e os pesos são dependências locais e não são distribuídos neste repositório. A gravação incluída foi feita com o repositório oficial em `external/kev`; a revisão usada na instalação local foi `557598fced1dada75dfbf36ed144dce309ac6ceb`. Para repetir o teste, clone o projeto nesse caminho, confira a documentação upstream e prepare o ambiente e o checkpoint antes de iniciar o servidor. O servidor precisa estar ligado enquanto a partida roda. No primeiro terminal, a partir da raiz do projeto:

```bash
cd external/kev
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m kev.serve --run jaredpalmer/kev-0.8b --port 8009
```

Confirme `curl http://127.0.0.1:8009/v1/models` antes da partida. O jogo verifica o checkpoint servido antes de aquecer e **recusa com erro** outra variante no lugar de `jaredpalmer/kev-0.8b`. Num segundo terminal, na raiz do repositório:

```bash
USE_TF=0 .venv/bin/python -m snake_linux --backend kev --headless \
  --width 8 --height 6 --steps 3 --max-speed \
  --record resultados/minha-partida-kev.jsonl
```

Para a interface jogável, retire `--headless`, `--steps` e `--record` se preferir. Encerre o servidor com `Ctrl+C` ao terminar: ele ocupa VRAM e pode levar o Laya a cair para CPU. O acesso do jogo é **HTTP no localhost**, não uma consulta à nuvem durante as jogadas. O servidor usa PyTorch/CUDA e o checkpoint com LoRA e cabeça de decisão do Kev; **não** trata o backbone Qwen como substituto isolado.[5]

**SemIf-4B:** os pesos `Qwen_Qwen3.5-4B-Q4_K_M.gguf`, o tokenizer e o clone em `external/SemIf/` são dependências locais e não são distribuídos neste repositório. A instalação local usou o commit `1f2dea3e25379f9dfc98cb83c324f00ab5deda37` do projeto original. Para repetir o teste, prepare esses arquivos e instale o código upstream no ambiente `.venv`, conforme descrito abaixo. O SemIf roda na **CPU**, por `llama.cpp`; não é preciso ligar um serviço:

```bash
USE_TF=0 .venv/bin/python -m snake_linux --backend semif --headless \
  --width 8 --height 6 --steps 3 --max-speed \
  --record resultados/minha-partida-semif.jsonl
```

`--semif-threads 4` altera o número de threads (4 por padrão); `--model CAMINHO.gguf` altera o arquivo GGUF **somente para SemIf**, mas continua exigindo vocabulário compatível com o tokenizer fixado. O adaptador checa igualdade de IDs entre os tokenizadores antes de ler logits; não gera uma resposta textual nem usa o GGUF como se carregasse uma cabeça de decisão treinada. O código do SemIf veio do commit `1f2dea3e25379f9dfc98cb83c324f00ab5deda37`; seu manifesto fixa a revisão do tokenizer Qwen em `851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a` e o GGUF em `4168f45a16a1290d65a4ec0fa312ae917a4c15d6`.[6]

**Instalação nesta máquina:** o Kev está num venv Python 3.12 isolado; o SemIf usa o `.venv` Python 3.11 do jogo com instalação editável sem alterar as versões existentes de PyTorch e Transformers, mais `llama-cpp-python==0.3.35` e `diskcache`. Portanto, o SemIf foi **testado aqui**, mas não é uma reprodução exata das versões do ambiente pinado pelo autor. As cópias em `external/` preservam as licenças upstream. Não atualize as dependências do jogo para as do Kev: são ambientes diferentes.[5][6]

### Primeira comparação local observada

Partidas **curtas** de três passos, semente 7, tabuleiro 8 × 6, escudo ativo e mesmas perguntas. Cada processo fez até seis aquecimentos antes da partida; a média abaixo mede a chamada no laço da partida, não carga, download, aquecimento nem qualidade estatística. Não são medidas simultâneas: enquanto o servidor Kev ocupava a GPU, uma tentativa de baseline Laya caiu para CPU e foi descartada desta tabela.

| Backend | Execução | Média por decisão | Comida | Intervenções |
| --- | --- | ---: | ---: | ---: |
| Laya multilíngue, CUDA | [laya-cuda-3.jsonl](resultados/laya-cuda-3.jsonl) | 28,0 ms | 1 | 0 |
| Kev-0.8B, servidor local CUDA | [kev-3-rastreado.jsonl](resultados/kev-3-rastreado.jsonl) | 100,4 ms, incluindo HTTP | 1 | 0 |
| SemIf-4B, GGUF Q4_K_M na CPU | [semif-3-rastreado.jsonl](resultados/semif-3-rastreado.jsonl) | 9.375,1 ms | 1 | 0 |

Todos os três arquivos contêm **três frames com inferências reais**, metadados de modelo e resumo. As jogadas foram reproduzidas no motor `SnakeGame`: cada tabuleiro gravado corresponde ao estado anterior à ação, as probabilidades direcionais somam 1 dentro de tolerância e as ações executadas estavam entre as rotas consideradas seguras. Três passos não permitem declarar vencedor em inteligência, robustez ou jogo longo. Kev e SemIf propuseram `LEFT, DOWN, DOWN`; Laya propôs `LEFT, DOWN, RIGHT`. O Kev reportou o checkpoint servido e precisão; o SemIf registrou o SHA-256 do GGUF e revisão do tokenizer. A demo permanece protegida pelo planejador determinístico.[5][6]

## Instalar de novo em outro Linux

Requer Python 3.10+, espaço para as dependências e pesos, e terminal grande com cores. Esta instalação utilizou `uv`, Python 3.11, `laya==0.3.6`, `torch==2.14.0+cu130` e `rich==15.0.0`. CUDA funcional foi confirmada na RTX 2050; CPU funciona potencialmente mais devagar. No projeto, os pesos foram baixados antecipadamente, não no meio do jogo.[3]

```bash
uv venv .venv
uv pip install --python .venv/bin/python 'laya==0.3.6' 'rich==15.0.0' pytest
.venv/bin/hf download convaiinnovations/laya --include 'multilingual/*' \
  --local-dir models/laya
USE_TF=0 .venv/bin/python -m snake_linux
```

**Atenção:** a instalação de PyTorch CUDA pode baixar vários gigabytes. Para testar em outro hardware, escolha previamente um pacote PyTorch compatível ou CPU no ambiente virtual. O diretório `models/laya` deve conter `multilingual/rl_agent_config.json`, `multilingual/model.safetensors`, `multilingual/encoder/config.json` e arquivos do tokenizer. Não confunda o ID do modelo com o diretório local exigido pelo jogo. O código ativa `HF_HUB_OFFLINE` durante a partida.

## Aprenda modificando

1. Compare `python -m snake_linux --headless --steps 30 --max-speed` com a mesma opção e `--unassisted`. Sem proteção, o modelo pode escolher parede ou corpo. Não conclua que o escudo foi acionado se o contador permanecer em zero.
2. Abra `snake_linux/policy.py` e altere **somente** o texto das alternativas em `descriptions`. Verifique como mudam as quatro probabilidades, preservando semente e tamanho do tabuleiro. O texto em inglês acompanha o treino e o exemplo original; o checkpoint multilíngue também aceita outros idiomas, mas não há teste comparativo aqui.
3. Compare `--prompt compact` com `--prompt detailed`. A segunda opção inclui contagem de casas abertas e comprimento da cobra no estado. Registre partidas em caminhos diferentes, pois `--record` não sobrescreve arquivos.
4. Leia `snake_linux/game.py` para separar o cálculo seguro e a colisão determinística do julgamento neural em `policy.py`. Leia `snake_linux/ui.py` para ver como as probabilidades viram barras. Rode `.venv/bin/python -m pytest tests -q` para verificar os testes.

A sugestão compartilhada por Diego sobre até **8 mil tokens** se refere à capacidade de expansão do contexto do backbone multilíngue. O checkpoint usa **1.024 tokens por padrão** na configuração documentada, e é possível alterar `agent.cfg["max_len"]` e `agent.cfg["head_max_len"]` para casos longos. Esta demo usa entradas curtas e não ganha nada ao elevar o contexto; isso pode aumentar a latência e o consumo de memória. Não confunda capacidade arquitetural máxima com configuração padrão e desempenho validado.[4]

## Fontes

[1] [Demo original e explicação do planejamento](https://github.com/mizorewww/laya-mlx/blob/main/docs/SNAKE_DEMO.md), repositório [Laya-MLX](https://github.com/mizorewww/laya-mlx).

[2] `resultados/snake-120.jsonl` e `resultados/snake-20.jsonl`, gerados localmente com o checkpoint real.

[3] [Repositório oficial do Laya](https://github.com/NandhaKishorM/laya) e [checkpoint multilíngue](https://huggingface.co/convaiinnovations/laya/blob/main/README.md).

[4] [Card do modelo Laya](https://huggingface.co/convaiinnovations/laya/blob/main/README.md), seção sobre limites de contexto e orçamento de tokens.
[5] [Kev no GitHub](https://github.com/jaredpalmer/kev) e [checkpoint Kev-0.8B](https://huggingface.co/jaredpalmer/kev-0.8b).
[6] [SemIf no GitHub](https://github.com/TheoLeeCJ/SemIf), [manifesto de modelos](https://github.com/TheoLeeCJ/SemIf/blob/master/manifests/models.json) e [GGUF Q4_K_M](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF/tree/main).
