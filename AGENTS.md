# Orientações para agentes: testes de modelos de decisão tipada

## Escopo e fontes de verdade

Este arquivo governa este repositório, um laboratório para aprender como modelos de decisão tipada funcionam no Linux, com foco prático na demo Snake. Leia `sobre.md` para a pesquisa sobre Laya e Jev, `DEMO.md` para instalação e uso, e `memory.md` para o histórico operacional recente. Para detalhes atuais das APIs, confirme a documentação e os repositórios upstream, pois os projetos evoluem rapidamente.

Não grave aqui resultados de partidas, números de benchmark ou tarefas pendentes. Use `memory.md` para decisões e descobertas operacionais relevantes, `resultados/` para execuções e relatórios, e `artefatos/` para prévias. Registre em cada experimento versão, checkpoint, hardware, perguntas, entradas, semente, método de medição e limitações.

## Arquitetura e limites

- `snake_linux/game.py` implementa regras e planejamento determinístico do Snake. `snake_linux/policy.py` transforma atributos calculados pelo planejador em três perguntas (`choice` para direção, `noul` para rota segura e `noul` para alcance da comida). `snake_linux/alternatives.py` converte essas perguntas para Kev, SemIf e Mapika decider sem mudar o planejador; `snake_linux/ui.py` apresenta as saídas; `cli.py` escolhe o backend e controla a execução. Não apresente o planejador nem o escudo como aprendizado autônomo do modelo.
- Uma decisão é uma chamada real a `agent.predict(state, questions)`. Para Kev, ela usa o servidor local; para SemIf, o adaptador pontua três perguntas a partir do mesmo estado com o código upstream; para Mapika decider, o adaptador chama `system_one` localmente. Preserve separadamente probabilidades originais, direção proposta pelo modelo, direção executada pelo escudo e intervenções. A apresentação de um estado gravado deve corresponder à predição feita **antes** da jogada seguinte. Não substitua inferência por valores aleatórios ou gravações sem identificar explicitamente uma simulação ou replay.
- O escudo do ciclo é a proteção padrão contra movimentos inadmissíveis; `--unassisted` é um experimento que pode resultar em morte. Mudanças no jogo ou no escudo devem manter as invariantes de legalidade e ordem do corpo. Para decisões fora do jogo, use revisão humana antes de ações irreversíveis.
- `DEAD-END RISK` e `FOOD REACHABLE` são estimativas do modelo escolhido sobre atributos já resumidos, não probabilidades validadas de morte futura. No SemIf, são leituras de logits condicionadas a sim/não, sem calibração para Snake. Resposta tipada não equivale a acerto. `score` é posição ordinal; `noul` é `P(true)` e não é sinônimo de `confidence`.
- O checkpoint usado na demo é o multilíngue, carregado localmente do diretório `models/laya` pela subpasta `multilingual`. O código deve detectar `multilingual/rl_agent_config.json`, não `multilingual/config.json`. Não baixe pesos dentro do loop do jogo nem presuma que a disponibilidade do arquivo implica que o modelo foi carregado corretamente.
- Aqui, Laya roda com PyTorch em Linux, não MLX nem llama.cpp. O modelo não gera texto livre, não faz busca por conta própria e não consome um documento inteiro sem limites. Um sistema RAG precisa recuperar e selecionar trechos antes de enviá-los ao Laya. O contexto padrão do multilíngue e a capacidade máxima configurável são coisas diferentes; avalie impacto de mudanças de contexto na latência e na precisão.
- Kev-0.8B usa um venv separado em `external/kev/.venv` e servidor somente em `127.0.0.1:8009`, PyTorch/CUDA. Não o deixe rodando quando testar Laya na mesma GPU de 4 GiB: o Laya pode cair para CPU. SemIf usa código de `external/SemIf`, tokenizer em `models/semif-tokenizer` e GGUF em `models/semif`, na CPU com `llama-cpp-python`. Não alegue que o GGUF sozinho é um modelo de decisão treinado.
- Mapika decider usa `decider-ai` e checkpoints locais em `models/decider-*`; o adaptador chama `system_one` e mantém separadas probabilidades, proposta, ação executada e intervenções. As temperaturas são ajustadas pelo checkpoint, mas isso não valida calibração para Snake. Os pesos 2B e 4B são grandes para a RTX 2050 de 4 GiB, portanto compare-os na CPU ou confirme memória antes de usar `--decider-device cuda`. Consulte `DEMO.md` e `memory.md` para comandos e resultados reais; não altere dependências upstream ou venvs sem verificar impacto no Laya.

## Como trabalhar

1. Antes de instalar ou alterar o ambiente, confira Python, venv, CUDA/CPU, memória disponível e arquivos de modelo. Preserve a configuração existente. Não altere drivers, serviços ou backends sem combinar escopo e impacto. O servidor Kev não é serviço permanente; inicie-o apenas durante a partida e desligue-o depois.
2. Para comportamento novo, escreva primeiro um teste que falhe. Teste regras do jogo e proteção com um agente injetado; depois confirme pelo menos uma inferência com o checkpoint real. Evite confundir mocks com validação de funcionamento do Laya.
3. Se adaptar componentes de `mizorewww/laya-mlx`, mantenha atribuição e as cópias `LICENSE.upstream` e `NOTICE.upstream`. A demo original é de macOS/MLX; remova rótulos ou comandos de MLX ao portar para Linux. O repositório do Laya é `NandhaKishorM/laya`. Não modifique os clones em `external/` para compatibilizar os adaptadores.
4. Em novas aplicações, defina estado e perguntas atômicas com critérios claros. Separe julgamento neural de regras determinísticas, recuperação de documentos e ação. Congele amostra rotulada para avaliar erros por categoria, idioma e distribuição antes de automatizar. Não trate confiança como garantia de calibração no domínio de Diego.
5. Para fine-tuning, compare primeiro checkpoint sem ajuste e baseline simples. O notebook oficial usa conjunto e checkpoint específicos em duas T4; não suponha que treinamento completo caiba na GPU apenas porque a inferência cabe. Preserve teste isolado e meça calibração.

## Execução e validação

A partir da raiz deste projeto:

```bash
USE_TF=0 .venv/bin/python -m snake_linux
.venv/bin/python -m pytest tests/ -q
USE_TF=0 .venv/bin/python -m snake_linux --headless --steps 20 --max-speed
USE_TF=0 .venv/bin/python -m snake_linux --backend semif --headless --steps 3 --max-speed
PYTHONPATH="$PWD/.venv/lib/python3.11/site-packages" USE_TF=0 .venv-decider/bin/python -m snake_linux --backend decider --model models/decider-2b-v11 --decider-device cpu --headless --steps 3 --max-speed
```

A UI interativa requer terminal de pelo menos 104 colunas por 35 linhas. A execução headless é finita e deve reportar chamadas reais ao modelo. `--record resultados/NOME.jsonl` cria uma gravação sem sobrescrever a existente; confronte cada frame com `game.snapshot()`, probabilidades, ação executada e resumo final. Para prévia SVG de uma gravação real, veja `scripts/gerar_preview.py` e seu comando em `DEMO.md`.

Antes de declarar uma alteração concluída, execute os testes, faça uma corrida headless com o checkpoint real quando a integração mudar, e relate resultados medidos e bloqueios sem inventar saídas. Não faça commit, push ou publique pesos/dados sem solicitação explícita.
