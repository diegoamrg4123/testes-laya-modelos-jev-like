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

## Reproduzir os testes no terminal

Esta seção descreve um procedimento completo para repetir uma partida visível ou headless, registrar as decisões e conferir a gravação. Execute tudo a partir da raiz do repositório. Os comandos não baixam pesos dentro da partida.

### 1. Conferir o ambiente

Antes de iniciar, confirme que o terminal tem pelo menos **104 colunas × 35 linhas**, que o ambiente principal existe e que o checkpoint escolhido está no caminho esperado:

```bash
pwd
.venv/bin/python --version
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
test -d models/laya && printf 'Laya encontrado\n'
```

Para partidas com Laya, use `.venv`. Para os dois checkpoints Mapika decider, use `.venv-decider` com o `PYTHONPATH` indicado na seção do decider. Não mantenha o servidor Kev ligado enquanto medir Laya ou outro modelo na RTX 2050.

### 2. Rodar com a interface visível

Este é o modo para observar o tabuleiro, as barras de probabilidade e os campos `PROPOSED`, `EXECUTED` e `SHIELD`:

```bash
USE_TF=0 .venv/bin/python -m snake_linux \
  --backend laya --width 8 --height 6 --seed 7 \
  --max-speed
```

O jogo continua até `Q`. As teclas disponíveis são:

| Tecla | Ação |
| --- | --- |
| `Espaço` | Pausar ou continuar |
| `↑` | Aumentar a velocidade alvo |
| `↓` | Reduzir a velocidade alvo |
| `R` | Reiniciar com a próxima semente |
| `Q` | Sair |

Para observar somente uma partida curta e deixar o resultado no terminal, acrescente `--steps 3 --no-alt-screen`. Para modelos lentos, `--max-speed` evita espera artificial entre inferências, mas não acelera o modelo.

### 3. Rodar headless com gravação

O modo headless não desenha o tabuleiro e é adequado para uma execução finita ou para automação. Cada nome passado a `--record` deve ser novo, pois o programa não sobrescreve gravações existentes.

```bash
USE_TF=0 .venv/bin/python -m snake_linux \
  --backend laya --headless --width 8 --height 6 --seed 7 \
  --steps 3 --max-speed \
  --record resultados/laya-terminal-3.jsonl
```

Ao final, o processo imprime um resumo JSON com `steps`, `inference_calls`, `score`, `deaths`, `interventions` e `mean_inference_ms`. O arquivo JSONL contém uma linha de metadados, uma linha `frame` por decisão e uma linha `end`.

### 4. Preparar e testar o Mapika decider

Os checkpoints usados nesta pesquisa são `Mapika/decider-2b` na versão `2b-v11` e `Mapika/decider-4b` na versão `4b-v2.1`. A instalação local usa um ambiente auxiliar que reaproveita o PyTorch do `.venv` principal:

```bash
uv venv --system-site-packages .venv-decider
uv pip install --python .venv-decider/bin/python --no-deps \
  'decider-ai==1.5.0'
```

O comando seguinte faz o processo encontrar o PyTorch e as bibliotecas do ambiente principal, mas carregar o pacote `decider-ai` do ambiente auxiliar:

```bash
export PYTHONPATH="$PWD/.venv/lib/python3.11/site-packages"
```

Baixe os pesos antes da partida. As revisões abaixo são as usadas nos registros deste projeto:

```bash
.venv/bin/hf download Mapika/decider-2b \
  --revision 533964dae8be954c5b5e19fa4948e48408094c1e \
  --local-dir models/decider-2b-v11

.venv/bin/hf download Mapika/decider-4b \
  --revision eb5fbdfc9448473ec25e399882912863afbdb70e \
  --local-dir models/decider-4b-v2.1
```

Na RTX 2050 de 4 GiB, a reprodução comparável deve usar CPU. O 2B ocupa aproximadamente 3,8 GB em BF16 e o 4B aproximadamente 8,4 GB. Os dois modelos usam fallback PyTorch se os kernels opcionais `causal_conv1d` e `flash-linear-attention` não estiverem ativos. Isso é compatível, mas deixa as decisões mais lentas.

Para ver o decider-2b v11 no terminal:

```bash
export PYTHONPATH="$PWD/.venv/lib/python3.11/site-packages"
USE_TF=0 .venv-decider/bin/python -m snake_linux \
  --backend decider --model models/decider-2b-v11 \
  --decider-device cpu --width 8 --height 6 --seed 7 \
  --steps 3 --max-speed --no-alt-screen \
  --record resultados/decider-2b-v11-terminal.jsonl
```

Para repetir o teste headless do mesmo checkpoint, acrescente `--headless`. Para o decider-4b v2.1, troque apenas o caminho do modelo e o nome do registro:

```bash
USE_TF=0 .venv-decider/bin/python -m snake_linux \
  --backend decider --model models/decider-4b-v2.1 \
  --decider-device cpu --headless --width 8 --height 6 \
  --seed 7 --steps 3 --max-speed \
  --record resultados/decider-4b-v2.1-terminal.jsonl
```

### 5. Repetir Kev e SemIf

O Kev exige dois terminais. No primeiro, ligue o servidor local:

```bash
cd external/kev
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m kev.serve \
  --run jaredpalmer/kev-0.8b --port 8009
```

No segundo terminal, volte para a raiz do projeto, confirme o servidor e execute a partida:

```bash
curl http://127.0.0.1:8009/v1/models
USE_TF=0 .venv/bin/python -m snake_linux \
  --backend kev --headless --width 8 --height 6 --seed 7 \
  --steps 3 --max-speed --record resultados/kev-terminal-3.jsonl
```

Encerre o servidor Kev com `Ctrl+C` assim que terminar. Para o SemIf, não é preciso servidor:

```bash
USE_TF=0 .venv/bin/python -m snake_linux \
  --backend semif --headless --width 8 --height 6 --seed 7 \
  --steps 3 --max-speed --semif-threads 4 \
  --record resultados/semif-terminal-3.jsonl
```

### 6. Validar uma gravação

Depois de uma partida, confira o resumo do arquivo e execute a suíte que reproduz as gravações incluídas:

```bash
.venv/bin/python -m pytest tests/ -q
```

A validação esperada deve confirmar que cada `frame.game` é o estado anterior à ação, que as probabilidades estão completas, que `executed` pertence às direções seguras quando o escudo está ativo e que o resumo final concorda com a quantidade de frames. Três passos confirmam integração e formato, mas não permitem comparar inteligência, calibração ou qualidade de jogo longo.

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

## Testar a mesma cobra com backends alternativos

O jogo agora aceita `--backend laya` (padrão), `--backend decider`, `--backend kev` e `--backend semif`.
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

**Mapika decider-2b v11 e decider-4b v2.1:** o adaptador usa a API `system_one` do pacote `decider-ai` e mantém as mesmas três perguntas, o mesmo planejador e o mesmo escudo. Como a RTX 2050 desta máquina tem 4 GiB, o teste comparável dos dois pesos foi feito na CPU. Os registros curtos e a medição estão em [`resultados/decider-v11-v2.1-snake-3.md`](resultados/decider-v11-v2.1-snake-3.md). Depois de preparar o ambiente isolado e baixar os pesos, os comandos são:

```bash
export PYTHONPATH="$PWD/.venv/lib/python3.11/site-packages"
USE_TF=0 .venv-decider/bin/python -m snake_linux --backend decider \
  --model models/decider-2b-v11 --decider-device cpu --headless \
  --width 8 --height 6 --seed 7 --steps 3 --max-speed \
  --record resultados/novo-decider-2b.jsonl

USE_TF=0 .venv-decider/bin/python -m snake_linux --backend decider \
  --model models/decider-4b-v2.1 --decider-device cpu --headless \
  --width 8 --height 6 --seed 7 --steps 3 --max-speed \
  --record resultados/novo-decider-4b.jsonl
```

O backend aceita `--decider-device cuda`, mas o 2B requer cerca de 4 GiB e o 4B cerca de 8,4 GiB em BF16. Não foi usado CUDA nesta comparação para evitar falta de memória e manter os dois modelos no mesmo dispositivo.

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
