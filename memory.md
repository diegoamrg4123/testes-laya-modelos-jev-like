# Memória operacional do projeto

Este arquivo guarda histórico técnico e descobertas recentes que ajudam a retomar o laboratório. Ele pode ser atualizado com mais frequência que `AGENTS.md`. Não substitui o README, os relatórios em `resultados/` nem a documentação upstream.

## Papel do projeto

O repositório é um laboratório Linux para estudar modelos de decisão tipada que recebem um estado e perguntas delimitadas, como `choice`, `score` e `noul`. A aplicação principal é uma demo Snake com inferência local, planejamento determinístico e escudo de segurança. O projeto não é o Jev, não é um SDK oficial da TypeSafe e não mede sozinho a qualidade geral dos modelos.

A separação central é:

1. `snake_linux/game.py` aplica as regras do Snake e o planejamento do ciclo.
2. `snake_linux/policy.py` transforma os atributos do estado em perguntas e separa probabilidades, proposta do modelo, ação executada e intervenção.
3. `snake_linux/alternatives.py` adapta backends reais sem trocar o planejador.
4. `snake_linux/ui.py` apresenta o estado e as decisões.
5. `snake_linux/cli.py` escolhe o backend e controla aquecimento, partida, gravação e resumo.

O modelo não recebe o tabuleiro cru. O estado já contém atributos resumidos, como rota segura, alcance atual da comida e descrições das alternativas.

## Histórico recente

### 2026-09-25: integração do Mapika decider

Foi adicionado o backend `decider` para os checkpoints:

* `Mapika/decider-2b`, versão `2b-v11`
* `Mapika/decider-4b`, versão `4b-v2.1`

O adaptador `DeciderAgent` chama a API `system_one` do pacote `decider-ai`. A interface da demo continua usando as perguntas `move`, `risk` e `food`, o mesmo estado do Snake e o mesmo escudo. O código não converte o modelo em gerador de texto nem troca a cabeça de decisão por um Qwen genérico.

A CLI ganhou:

* `--backend decider`
* `--model models/decider-*`
* `--decider-device cpu|cuda`

O teste do adaptador foi escrito antes da implementação e cobre a preservação das respostas tipadas. A suíte completa passou com 32 testes.

### Checkpoints baixados

Os diretórios locais são ignorados pelo Git e ficam em `models/`:

* `models/decider-2b-v11`, revisão do Hub `533964dae8be954c5b5e19fa4948e48408094c1e`
* `models/decider-4b-v2.1`, revisão do Hub `eb5fbdfc9448473ec25e399882912863afbdb70e`

O arquivo `decider_config.json` confirma as versões `2b-v11` e `4b-v2.1`, com data de lançamento `2026-09-24`. Os hashes LFS e tamanhos dos pesos estão no relatório `resultados/decider-v11-v2.1-snake-3.md`.

### Ambiente usado para o decider

A RTX 2050 tem 4 GiB de VRAM. Para comparar os dois checkpoints sem disputa de memória, os testes foram executados na CPU.

* Python: 3.11.16
* PyTorch: 2.14.0+cu130
* Transformers: 5.17.0
* `decider-ai`: 1.5.0
* Ambiente auxiliar: `.venv-decider`
* Dispositivo dos testes: CPU

O ambiente auxiliar foi criado com `uv` e os comandos usaram:

```bash
export PYTHONPATH="$PWD/.venv/lib/python3.11/site-packages"
```

Isso reutiliza o PyTorch e as bibliotecas do `.venv` principal. O pacote `decider-ai` declara `numpy<2`, mas essa combinação de `PYTHONPATH` faz o processo enxergar o NumPy instalado no `.venv` principal. O modelo funcionou, porém esse arranjo deve ser revisto antes de transformar o backend em instalação reproduzível para terceiros.

Durante a execução, o Transformers avisou que `causal_conv1d` e `flash-linear-attention` não estavam ativos. O fallback PyTorch é válido, mas muito mais lento. Os avisos são de desempenho, não de carregamento incompleto do checkpoint.

### Execuções do decider no Snake

Configuração comum dos testes comparáveis:

* Tabuleiro 8 × 6
* Semente 7
* Comprimento inicial 6
* Seis decisões de aquecimento
* Três passos medidos
* `--max-speed`
* Escudo ativo
* Estado compacto e três perguntas tipadas

Resultados registrados:

| Modelo | Inferências medidas | Comida | Mortes | Intervenções | Média observada |
| --- | ---: | ---: | ---: | ---: | ---: |
| decider-2b v11 | 3 | 1 | 0 | 0 | 14.209,3 ms no primeiro registro; 17.430,1 ms no teste visual posterior |
| decider-4b v2.1 | 3 | 1 | 0 | 0 | 37.113,1 ms |

Nos registros comparáveis, os dois modelos propuseram e executaram `LEFT`, `DOWN`, `DOWN`. Os estados dos frames foram reproduzidos pelo `SnakeGame`, as distribuições direcionais somaram 1 dentro do arredondamento e as ações executadas eram seguras.

Arquivos:

* `resultados/decider-2b-v11-snake-3.jsonl`
* `resultados/decider-4b-v2.1-snake-3.jsonl`
* `resultados/decider-2b-v11-snake-3-view.jsonl`
* `resultados/decider-2b-v11-snake-live.jsonl`
* `resultados/decider-v11-v2.1-snake-3.md`

O teste visual foi inicialmente executado com `--headless`, portanto mostrou somente logs e resumo. Para ver o tabuleiro, o comando precisa omitir `--headless`, usar terminal com pelo menos 104 colunas por 35 linhas e, de preferência, usar `--max-speed` em partidas curtas.

Essas partidas são smoke tests de integração. Não sustentam uma conclusão sobre qual modelo é melhor, inteligência geral, calibração ou desempenho de jogo longo. As métricas publicadas pelo autor, como 0,752 no regression set para o 2B e 0,784 para o 4B, não foram reproduzidas neste Snake.

## Registros anteriores

Antes do decider, o projeto já tinha execuções reais com:

* Laya multilíngue em PyTorch/CUDA, incluindo `resultados/snake-120.jsonl` e `resultados/snake-20.jsonl`.
* Kev-0.8B por servidor local em `127.0.0.1:8009`.
* SemIf-4B com GGUF Q4_K_M na CPU.

Os detalhes, comandos e limitações estão em `DEMO.md`. O Kev e o Laya não devem ser mantidos simultaneamente na RTX 2050 durante comparações, porque podem disputar a VRAM.

## Testes de modelos adicionais em 2026-09-25

### Together Tev1-0.8B-experimental

Foi testado localmente no Snake sem integrar backend ao CLI. É um fine-tune experimental de Qwen3.5-0.8B com saída autoregressiva por letra, não um runtime Jev nem uma API de probabilidades tipadas. Revisão `6bb2dff14b38fea90ddb14d870166ccaf77374e9`; detalhes, SHA-256, medidas e limites em `resultados/tev1-0.8b-experimental-snake-3.md`; harness em `scripts/test_tev1_snake.py`.

Três passos reais em tabuleiro 8 × 6, semente 7, RTX 2050, FP16; propostas LEFT, DOWN, DOWN, cobra viva e 1 ponto. Média observada de 889,37 ms por decisão, cada uma com três chamadas ao modelo. O harness normaliza logits das letras válidas para encaixar no `LayaPolicy`; isso é preferência relativa construída pelo harness, não probabilidade oficial ou calibrada do Tev1. O card não declara avaliação multilíngue abrangente e informa que a licença dos pesos está sendo finalizada; não redistribuir.

### GLiNER2.5 multilingual

Checkpoint `fastino/gliner2.5-multi-v1`, revisão `12fc40399dae672ce840c5e3c50a92340bff3c8c`, carregado e inferido em CPU com GLiNER2 2.0.0 e Transformers 4.53.3. O campo legado `extra_special_tokens` em forma de lista causa exceção no carregamento padrão. Workaround temporário, sem editar os arquivos do checkpoint: substituir em runtime `gliner2.models.base.load_extractor_tokenizer` por `AutoTokenizer.from_pretrained(path, extra_special_tokens={})`. Uma inferência de classificação e uma partida interativa curta foram executadas com `Classifier`, usando probabilidades retornadas pela API de classificação. O harness temporário foi `/home/diego/.hermes/cache/scratch/run_gliner_live.py`; isso não adicionou backend à CLI. A janela real do GNOME Terminal foi fechada depois, e os processos de jogo foram encerrados. O teste usou estado em inglês, portanto não avaliou capacidade multilíngue. A solução de compatibilidade é provisória; não alterar em silêncio a cópia do checkpoint.

## Regras operacionais consolidadas

* Estados gravados representam o momento anterior à ação seguinte.
* Probabilidades originais, direção proposta e direção executada precisam continuar separadas.
* O escudo não é acerto do modelo. Ele é uma proteção determinística contra movimentos inadmissíveis.
* `DEAD-END RISK` e `FOOD REACHABLE` são opiniões tipadas sobre atributos atuais do planejador, não probabilidades calibradas de morte ou vitória.
* Não baixar pesos dentro do laço do jogo.
* Não substituir inferência real por valores aleatórios ou replay sem identificar claramente a simulação.
* Usar `resultados/` para JSONL e relatórios e `artefatos/` para prévias.
* Não fazer commit, push ou publicar pesos e dados sem solicitação explícita.

## Como retomar

Para uma execução headless curta do 2B com o ambiente atual:

```bash
PYTHONPATH="$PWD/.venv/lib/python3.11/site-packages" \
USE_TF=0 .venv-decider/bin/python -m snake_linux \
  --backend decider --model models/decider-2b-v11 \
  --decider-device cpu --headless --width 8 --height 6 \
  --seed 7 --steps 3 --max-speed \
  --record resultados/novo-decider-2b.jsonl
```

Para ver a interface, retire `--headless`. Antes de concluir mudanças no código, rode:

```bash
USE_TF=0 .venv/bin/python -m pytest tests/ -q
```

Quando uma alteração tocar o adaptador ou o carregamento de um modelo, faça também uma inferência real com o checkpoint afetado e registre o resultado em um arquivo novo.
