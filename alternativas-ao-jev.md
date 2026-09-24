# Alternativas ao Jev: modelos, projetos e evidências

> Pesquisa documental consultada em 23/09/2026. Este arquivo distingue documentação e resultados publicados por seus autores de observações neste notebook. **Nenhuma das alternativas listadas abaixo foi instalada, carregada ou avaliada localmente nesta pesquisa.** O único modelo de decisão cuja inferência foi demonstrada aqui é o Laya, descrito em [sobre.md](sobre.md) e [DEMO.md](DEMO.md). Projetos e pesos podem mudar: confirme as revisões antes de reproduzir um teste.

## Primeiro: o que significa ser alternativa?

O Jev, da TypeSafe, recebe um estado e perguntas de resposta limitada, como escolha, nível ordinal e probabilidade de verdadeiro.[1]
Uma alternativa pode reproduzir a **interface** de perguntas, o **tipo de saída**, parte da **arquitetura presumida**, a **capacidade de execução local** ou a **qualidade em alguma avaliação**.[1][5]
Uma dessas semelhanças não implica as outras. Jev é proprietário; nem a arquitetura nem seus dados de treinamento podem ser presumidos a partir desses projetos abertos.[1][5]

| Projeto | O que de fato se executa | Perguntas e integração | Condição das evidências |
| --- | --- | --- | --- |
| Laya | Encoder e cabeça própria de decisão; checkpoint multilíngue já usado neste laboratório | `state` + `choice`, `score`, `noul` em Python | Inferência local real documentada em `DEMO.md`; não é benchmark comparativo.[11] |
| Kev | Qwen3.5-Base + adaptador LoRA + cabeça que pontua alternativas; pesos publicados em 0,8B, 4B e 9B | Servidor `/v1/systemone` compatível com o SDK da TypeSafe; três tipos de pergunta | Código, modelos, dados e testes publicados; números abaixo são dos autores, não medidos aqui.[1] |
| SemIf | Qwen congelado e leitura de logits das opções, sem checkpoint de decisão próprio na configuração principal | Ferramenta `semif-score` e entradas JSONL; não é um substituto imediato da API `agent.predict` | Resultados, artefatos e limites publicados pelos autores; sem execução local nesta pesquisa.[5][7] |
| Jev-Omni | Gemma 4 12B multimodal com backbone textual ajustado e cabeça de opções | API Python com estado, **uma** pergunta, opções e mídia opcional; não é substituto direto do SDK TypeSafe | Pesos e código publicados; resultados e latências divulgados pelo autor, não medidos aqui.[22][23][24] |
| `harshatheg/Qwen-2.5-1B-RLCD` | Código de inferência estruturada que carrega outro Qwen2.5-Instruct | Campos `enum` e `boolean` em esquema JSON | Não foram encontrados pesos próprios nem demonstração, no código consultado, de treino RLCD.[8][9][10] |

## Kev: o substituto mais próximo no formato de uso

**Funcionamento.** Os checkpoints atuais adaptam Qwen3.5-Base com LoRA de posto 16 e uma pequena cabeça de decisão.[1][2]
O modelo recebe um texto de estado e cada pergunta com suas opções; a cabeça pontua as opções e normaliza seus escores.[1]
Ele não precisa gerar uma resposta em prosa nem serializar um JSON token por token.[1]
Na base Qwen3.5, cada pergunta percorre sua própria linha de processamento, com cache do prefixo de estado reaproveitado; as perguntas não leem umas às outras.[1]
O projeto também mantém pesos de uma geração anterior sobre Qwen3.[1]
Não confunda o número de parâmetros da base com a quantidade de parâmetros treináveis do adaptador.[2]

**Uso.** Há `noul` para probabilidade de verdadeiro, `choice` para escolher entre opções com distribuição e `score` para uma posição numa escala.[1]
O servidor aceita `state` e `questions` em `/v1/systemone`, oferece respostas por pergunta e funciona com o SDK da TypeSafe apontado para um servidor local.[1]
Isso o torna um candidato à mesma tarefa de triagem do Laya, mas não um substituto de pesos ou de API Python por simples troca de nome: seria necessário escrever e testar um adaptador para nosso programa.[1][11]

**Qualidade publicada, não medida neste laboratório:**

| Checkpoint atual | Acurácia em fontes novas, teste retido | Brier em fontes novas, teste retido | Leitura prudente |
| --- | ---: | ---: | --- |
| Kev-0.8B | 0,684 | 0,460 | Primeiro candidato de GPU pequena; qualidade menor no conjunto do autor.[1][3] |
| Kev-4B | 0,837 | 0,255 | Salto expressivo sobre 0,8B nesse conjunto, com custo de memória maior.[1][2] |
| Kev-9B | 0,852 | 0,237 | Ganho menor que o salto anterior e custo de memória maior.[1][4] |

Menor Brier é melhor.[1]
Os autores usaram conjuntos de desenvolvimento e teste distintos; publicaram comparações com Jev, mas reconhecem que não conhecem os dados em que Jev foi treinado.[1]
**Não existe aqui um teste pareado com o Laya em textos de Diego, nem comprovação de superioridade em compreensão semântica em português.** Métricas obtidas com bases, prompts, idiomas ou distribuições diferentes não podem ser ordenadas como se fossem uma liga universal.[1][2]

**Contexto:** o servidor Kev aceita até **8.192 tokens por estado mais uma pergunta**, enquanto o treinamento descrito usou no máximo **384 tokens de estado e 1.024 tokens para estado mais pergunta**.[1] A base Qwen3.5-4B informa contexto arquitetural nativo muito maior, **262.144 tokens**, mas esse valor não é contexto validado para a tarefa e implementação do Kev.[1][12] Antes de classificar reportagens inteiras, selecione trechos e avalie se a evidência decisiva foi mantida.

**Probabilidades:** os checkpoints aplicam por padrão uma temperatura ajustada em dados de desenvolvimento, o que melhora algumas métricas de calibração relatadas.[1][2]
Para `choice`, o campo `confidence` segue fórmula própria baseada na distribuição e no número de alternativas; **não é a taxa medida de acerto daquela previsão**.[1]
Ordem das opções e mudança de domínio podem alterar resultados.[1]
Decisões sensíveis precisam de avaliação e limiares específicos com revisão humana.[1][2]

## SemIf: decisões diretas de um Qwen congelado

**Funcionamento.** Em vez de ajustar uma cabeça especializada, o SemIf prepara estado, pergunta e alternativas e lê os escores das opções diretamente do modelo.[5]
O caminho principal estudado usa Qwen3.5-4B sem fine-tuning para a tarefa; há comparação com modelos menores e maiores.[5]
É possível reutilizar o prefixo do mesmo estado entre perguntas.[5]
O SemIf também documenta um backend de **CPU com `llama.cpp` e arquivo GGUF**: a razão da compatibilidade é a base Qwen decoder que o projeto sabe pontuar, não uma conversão do Laya para GGUF.[5][7]

**Interface e qualidade.** O fluxo documentado é `semif-score` sobre JSONL com `state`, `question` e `options`, salvando opções, escores e metadados.[5][7]
Não fornece, sem adaptação, o contrato Python do Laya ou os três tipos `choice`/`score`/`noul` com a mesma semântica.[5][11]
Os autores relatam **0,813 de acurácia balanceada** em 144 decisões próprias com Qwen3.5-4B BF16, além de **0,637** numa avaliação WANLI de 256 exemplos.[5]
O subconjunto comparado com previsões públicas do Jev tem 102 linhas, não a avaliação agregada completa do fornecedor.[5]
Essas métricas **não são comparáveis numericamente** à tabela de acurácia do Kev.[1][5]

**Quantização e calibração.** Um artefato Q4_K_M do Qwen3.5-4B usado na demonstração em navegador é anunciado com download de cerca de **3,01 GB**; isso não garante caber integralmente em 4 GB de VRAM durante inferência.[5]
O backend GGUF pela CPU é uma alternativa, com velocidade e probabilidades potencialmente diferentes das do BF16.[5][7]
Os escores diretos das alternativas começam **sem calibração garantida**.[6]
O projeto acrescentou ajuste de temperatura **por conjunto de trabalho** usando exemplos rotulados e relata melhora fora da amostra em algumas tarefas, especialmente WANLI; não transfira essa calibração automaticamente para notícias, golpes ou português.[5][6]

## Jev-Omni: decisões tipadas com texto, imagem, áudio e vídeo

O [Jev-Omni](https://huggingface.co/akhilaaa3/Jev-Omni) é um projeto independente baseado em **Gemma 4 12B IT**, com pesos de backbone textual ajustado e uma cabeça que devolve probabilidades para alternativas delimitadas.[22][23][26]
O repositório declara licença **Apache-2.0** para o modelo; direitos de uso dos datasets são separados.[22]
A implementação pública recebe `state`, `question` e uma lista de `options`; com `media` e `modality`, aceita uma imagem, áudio ou vídeo além do texto da pergunta.[24]
Para vídeo, o código **amostra 16 quadros** por padrão; para áudio, converte o arquivo com `ffmpeg` e limita a **30 segundos**. Isso demonstra entradas dessas modalidades na implementação, não compreensão irrestrita de qualquer vídeo ou gravação.[22][24]
A afirmação de ser o **primeiro** modelo aberto a reunir as quatro modalidades é uma reivindicação da divulgação do projeto, **não uma prioridade histórica verificada aqui**.

**Interface:** o helper `predict` avalia **uma pergunta por chamada**, retorna alternativa vencedora, índice, `confidence` e probabilidades.[24]
Aceita de 2 a 256 opções, mas o cartão recomenda **até 20** como faixa sustentada pela avaliação.[22][24]
Embora o cartão anuncie `noul`, `choice` e `score`, o código examinado recebe opções genéricas; booleanos e níveis ordinais exigem formatação e interpretação pela aplicação. Não encontrei nesse helper um servidor `/v1/systemone` ou uma saída `score` contínua equivalente à API TypeSafe. Não o conecte à demo Snake trocando apenas o checkpoint.[22][24]

**Resultados publicados pelo autor, com protocolos diferentes:**

| Avaliação | Resultado divulgado | O que o número significa |
| --- | ---: | --- |
| DecisionBench Medium, 80 cenários e 293 perguntas | 87,57% de média por cenário; **86,01% por pergunta** | No mesmo conjunto, o cartão do dataset reporta **88,05% por pergunta** para Jev 1.13. Esses valores são próximos nesse recorte, não provam equivalência geral.[22][25] |
| JevBench, **195 grupos / 231 decisões alinhadas** | 86,15% por grupo; 87,45% por decisão | Subconjunto divulgado pelo autor, **não** o JevBench v1.3.0 integral de 534 decisões nem seu score composto. Não comparar com 74,4 ou 73,1 do ranking.[13][22] |
| MMAU, 1.000 questões de áudio | 63,10% | Resultado do cartão do modelo; falta reproduzir em hardware e condições independentes.[22] |
| MVBench, 14 tarefas e 2.786 questões de vídeo | 53,10% por tarefa; 53,09% por questão | Resultado do cartão, com limite de quadros definido pelo pipeline publicado.[22][24] |

O cartão **não apresenta uma avaliação específica para imagens** nem métricas por idioma. A aceitação de imagens pelo código não demonstra, sozinha, acerto em tarefas visuais de interesse.[22][24]
O cartão reporta ECE de **0,0400** no conjunto Medium, o que **não calibra automaticamente** saídas em português, outras modalidades ou aplicações sensíveis.[22]
O dataset DecisionBench é disponibilizado pelo autor, mas seu cartão informa que os rótulos são respostas pretendidas pelo gerador, **sem revisão cega independente**; ambiguidades podem afetar sobretudo os casos difíceis.[25]
A divulgação menciona 30 mil perguntas e treino com oito H200; já `decision_config.json` descreve **a etapa final** com 24 mil exemplos e `world_size: 4`. Isso não basta para reconstruir o custo e a sequência completos do treinamento: registre a diferença, sem presumir contradição ou validar os números da publicação.[22][26]

**Hardware e velocidade:** o cartão exige CUDA e estima cerca de **50 GB** para o checkpoint FP32 antes do overhead. Nos arquivos há também um checkpoint unificado BF16 de aproximadamente **22,28 GiB**, e o loader de referência carrega o backbone textual e os componentes multimodais do Gemma 4 separadamente.[22][23][24]
Mesmo o arquivo BF16 isolado é maior que a VRAM total de **4 GiB** desta RTX 2050; não é candidato para execução local neste notebook sem uma adaptação substancial, cuja qualidade e funcionamento não foram avaliados.[23][24]
A alegação de **menos de 100 ms** se refere a inferências aquecidas em **H200** para texto, imagem e áudio: o autor relata 83 ms para texto de cerca de 2 mil tokens, 26 ms para imagem e 31 ms para áudio de 13 segundos. Vídeo com 16 quadros levou **504 ms** na medição publicada; pré-processamento e rede ficam fora desses tempos.[22]

**Onde vale pesquisar:** triagem de imagens de produtos, classificação de trechos audiovisuais e seleção de evidências multimodais, sempre com critérios explícitos e revisão humana para decisões sensíveis. Para testar nesta instalação, seria preciso primeiro dispor de hardware compatível externo ou verificar uma versão realmente menor/quantizada que preserve a cabeça de decisão; a mera existência de pesos BF16 não resolve o limite de VRAM.[22][24]

## `Qwen-2.5-1B-RLCD`: nome sugestivo, evidência insuficiente

A listagem do repositório Hugging Face consultada não contém pesos próprios, como `*.safetensors`, `*.bin` ou adaptador treinado.[8]
O próprio cartão chama a implementação de **Qwen2.5-1.5B-Instruct com Parallel Constrained Decoding**, apesar de o identificador mencionar `1B-RLCD`.[10]
No Linux, `core/engine_torch.py` carrega por padrão `Qwen/Qwen2.5-1.5B-Instruct`; no Mac, o exemplo privilegia uma versão MLX de 4 bits.[9][10]
Trata-se de reutilizar um modelo existente para preencher campos de um esquema limitado.[9][10]

O código compartilha um prefixo e avalia sufixos de campos em lote, seleciona candidatos por logits e monta JSON programaticamente.[9]
Isso ajuda a limitar **valores e sintaxe**, não garante classificação correta.[9]
A temperatura padrão `1.0` com `softmax` não comprova calibração estatística: seria necessário confrontar probabilidades com rótulos independentes.[9]
No backend PyTorch, `run_rlcd_generation` é apenas um alias para a função de inferência paralela.[9]
**Não encontrei evidência, nesses artefatos, de treinamento RLCD de um novo checkpoint.** Os números de velocidade anunciados são do cartão do autor para hardware Apple, não medições neste Linux.[10]

**Conclusão:** estudar o mecanismo de saídas limitadas pode ser útil; classificá-lo como um modelo RLCD concorrente de Kev ou Laya não é sustentado pelos arquivos examinados. O código inclui caminho PyTorch para Linux, mas não foi executado nem validado aqui.[8][9]

## JevBench v1.3.0: comparação externa e suas ressalvas

O [JevBench](https://benchmarkheaven.com/jev-models) divulga 534 decisões por sistema, incluindo 220 casos difíceis, em quatro categorias de tarefas.[13]
O **JevBench Score não é porcentagem de acerto**: é uma média geométrica de quatro eixos, Inteligência corrigida pelo acaso, Calibração, Velocidade e Custo, cada um com peso de 25%.[13]
O custo atribuído a modelos sem tarifa pública é **estimado** a partir de preços de provedores semelhantes; a latência de servidores locais e de demonstração recebe um ajuste assumido para simular carga.[13]

| Configuração avaliada | Posição | Score composto | Inteligência | Acertos nos 220 difíceis | Onde foi executada |
| --- | ---: | ---: | ---: | ---: | --- |
| Jev 1.13.0 | 1ª | 74,4 | 85,7 | 74,1% | API de produção.[13] |
| SemIf sobre Qwen3.5-4B | 2ª | 73,1 | 79,0 | 59,5% | GPU RunPod do avaliador.[13] |
| Laya, checkpoint raiz em inglês | 33ª | 54,4 | 45,8 | 34,1% | CPU do avaliador.[13] |
| Kev-4B, **prévia anterior** | 27ª | 59,7 | 64,8 | 42,3% | RTX 3090 do avaliador, commit `20fa626`.[13] |

**Leitura correta:** naquele protocolo, **SemIf superou claramente a configuração testada do Laya** inclusive no eixo de Inteligência, que não depende do preço; não foi só efeito da média que combina custo e velocidade.[13]
Por outro lado, o avaliador executou o **Laya inglês na CPU, com orçamento de 512 tokens por pergunta**; relata que o pacote cortou estados longos dos casos difíceis.[13]
Este laboratório usa **outro checkpoint, o multilíngue, na RTX 2050**, e a demo Snake não é um teste de acurácia semântica.[11]
Portanto, o resultado não demonstra que SemIf supera **nosso Laya multilíngue em português**, mas eleva SemIf a candidato prioritário para um experimento pareado.
A comparação de velocidade da tabela mistura CPU e GPU, de modo que tampouco mede qual arquitetura é intrinsecamente mais rápida.[13]

**Atenção ao Kev:** a linha `kev 4B` da tabela foi executada a partir de uma **prévia no commit `20fa626`**, enquanto as métricas de Kev-0.8B, Kev-4B e Kev-9B acima se referem aos checkpoints **mais recentes, sobre Qwen3.5**, descritos pelo autor. Não use o 59,7 do JevBench para classificar a família Kev atual sem reavaliar o mesmo checkpoint.[1][2][13]

### Por que `classifier.dev` aparece com 83,6, acima de Jev?

O **fast tier do `classifier.dev` usa Jev**, não pesos próprios; o **smart tier** submete respostas de baixa confiança a outro modelo, mas **não foi o tier testado nessa linha do JevBench**.[13][14]
Por isso o benchmark mostra 83,6 como **menção honrosa, sem posição no ranking**.[13]
A embalagem em lotes e as condições de serviço podem produzir pequenas diferenças nas respostas e na latência; essa linha não demonstra um modelo mais inteligente que Jev.[13][14]

**Preço desatualizado no JevBench:** o texto do benchmark calcula cerca de **US$ 0,0033 por mil decisões** supondo uma modalidade Pro antiga de **US$ 20 por mês e 200 mil classificações rápidas por dia**, usada integralmente.[13]
A [página atual de preços do `classifier.dev`](https://classifier.dev/pricing) descreve **US$ 20 mensais em créditos incluídos**, cobrança por **tokens de entrada** e cobranças adicionais por escalonamento no smart tier; 200 mil por dia aparece como **limite de taxa Pro**, não como volume gratuito ilimitado.[15]
Assim, aquela estimativa de custo **não deve ser aplicada aos preços atuais sem recálculo**.[13][15]
O tier grátis do serviço é sujeito a limites e a um orçamento compartilhado; o texto classificado é enviado ao provedor do modelo, o que também difere de uma execução local.[14][15]

## O ranking como mapa de projetos, não apenas como placar

Além de confrontar SemIf com Laya, o JevBench é útil para **descobrir implementações, famílias de modelos, modos de execução e perguntas de pesquisa**. As 52 linhas misturam modelos próprios, adaptações de bases congeladas, rerankers, APIs proprietárias, demonstrações e execuções parciais; número de linhas não é número de novos checkpoints independentes. Antes de experimentar um nome, confira o repositório do autor, a disponibilidade dos pesos, a licença aplicável, a revisão efetivamente avaliada, o hardware e a origem dos dados de treinamento.[13]

| Projeto descoberto | Por que merece atenção | Obstáculo ou próximo filtro neste notebook |
| --- | --- | --- |
| **SemIf-4B** | 2º no JevBench e caminho GGUF/`llama.cpp` documentado; exemplo de leitura direta dos logits sem treinar uma cabeça.[5][13] | Executar quantizado pela CPU é uma hipótese de teste; não extrapolar a classificação BF16 do servidor GPU para esta configuração.[5][7] |
| **Jev-Omni** | Integra classificação tipada a **texto, imagem, áudio e vídeo**; publica código, pesos e avaliações por modalidade.[22][23][24] | Seus números de JevBench vêm de **subconjunto diferente** da tabela v1.3.0; checkpoint e loader de referência excedem amplamente a VRAM local. Observar evolução ou avaliar em hardware externo, não baixar pesos para este notebook como primeiro passo.[13][22][23] |
| **decider** | Família treinada com decisões tipadas sobre Qwen3.5, incluindo pesos **0,8B, 2B e 4B**, código, `choice`/`score`/`noul` e exemplos de API. O `decider-2b` medido pelo JevBench ficou em 23º, com 61,7 no score composto.[13][16] | O **0,8B tem pesos publicados**, mas não é a linha 2B do benchmark; pode ser um segundo candidato pequeno para testar em CUDA. O autor estima **cerca de 4 GB só para o 2B em BF16**, limite apertado para a RTX 2050, antes de comprovar carregamento real.[16][17] |
| **jeff / GLiFormer 400M** | API compatível com o SDK TypeSafe e `choice`/`score`/`noul`; usa modelo menor, com suporte anunciado a CPU e CUDA. O JevBench o colocou em 32º, com 54,4.[13][18] | Interessante como baseline local compacto; no protocolo do JevBench teve desempenho bem menor nos casos difíceis e não deve ser escolhido só pela compatibilidade da API.[13] |
| **reflex-4B** | Outra implementação sobre Qwen3.5-4B, com servidor compatível e alternativas com probabilidades; 5º no JevBench, com 70,3.[13][19] | A linha avaliada usa configuração própria e GPU; 4B BF16 não cabe inteiro nesta RTX 2050. O projeto informa mudanças posteriores, então fixar revisão e configuração antes de comparar.[13][19] |
| **djev-dev** | Abordagem diferente com **DiffusionGemma**, vLLM e escolhas tipadas inclusive a partir de imagens; o endpoint hospedado da Maisa ficou em 3º, com 73,0, **não uma execução local deste repositório**.[13][20] | O projeto reutiliza pesos DiffusionGemma, não oferece um novo checkpoint djev; os pesos e o runtime são grandes demais para supor execução viável nesta GPU.[13][20] |
| **system-one-open** | Experimento sobre Gemma 4 E2B, com servidor e exemplos variados; apareceu em 10º, com 66,6.[13][21] | A documentação consultada dizia que o upload dos pesos ao Hugging Face ainda estava pendente. Confirmar acesso aos pesos e licença da base antes de chamá-lo de alternativa local pronta.[21] |
| **Winnow-12B Q8** | 4º no JevBench, com 71,2; pode orientar estudo de qualidade e arquitetura de servidores.[13] | Pesos Q8 e contexto de 8.192 tokens foram usados em GPU com offload integral; o corpus privado de treino não permite auditoria independente completa. Não é candidato inicial para 4 GB de VRAM.[13] |

**Seleção prática:** para rodar algo diferente do Laya nesta máquina, começar por **Kev-0.8B ou decider-0.8b em um ambiente separado**, após conferir espaço e compatibilidade; explorar **SemIf-4B em GGUF pela CPU** como teste maior, sem prometer equivalência à linha de GPU do ranking.[1][5][16]
Os projetos grandes e os serviços hospedados servem para pesquisar mecanismos e tarefas, mas não devem substituir a comparação local do mesmo corpus em português. O benchmark é uma **lista de candidatos e hipóteses**, não uma resposta pronta sobre qual modelo resolve a tarefa de Diego.[13]

## O que cabe no notebook e o que ainda é hipótese

Nesta consulta, `nvidia-smi` informou **RTX 2050 com 4.096 MiB de VRAM**, dos quais **3.757 MiB estavam livres**; `free -h` informou **15 GiB de RAM total** e **6,8 GiB disponíveis**. São um retrato do momento, não reserva de memória. Só pesos de aproximadamente 4 bilhões de parâmetros em BF16 exigem cerca de **7,45 GiB**, sem cache e ativações; 9 bilhões exigiriam cerca de **16,76 GiB**. Assim, Kev-4B ou SemIf-4B BF16 **não cabem inteiros nessa GPU**. Kev-0.8B é **um dos** testes CUDA plausíveis, junto com decider-0.8b, mas carregar qualquer checkpoint completo ainda precisa ser verificado.[1][3][16]
SemIf-4B GGUF na CPU é um segundo teste plausível; não pressupor velocidade confortável nem equivalência com BF16.[5][7]

| Se você quer... | Primeiro experimento sensato | Por quê |
| --- | --- | --- |
| Algo já comprovado neste projeto | Laya multilíngue | Inferência real local registrada em `DEMO.md`; ponto de partida para novas tarefas. |
| API quase igual à proposta do Jev | Kev-0.8B ou decider-0.8b | Checkpoints pequenos publicados; capacidade na VRAM e qualidade em português a confirmar.[1][16][17] |
| Investigar Qwen 4B com contexto selecionado | SemIf + GGUF/CPU | Caminho documentado sem depender de 8 GB de VRAM; desempenho a medir. |
| Comparar qualidade sem enviesar | Laya, Kev-0.8B, decider-0.8b e SemIf na mesma amostra rotulada | Evita confundir tamanho, formato, idioma, distribuição e métricas.[1][5][16] |

**Não confundir janela máxima de contexto com compreensão de documentos longos.** Se o texto for uma reportagem ou conjunto de fontes, recupere ou selecione trechos pertinentes, mantenha a pergunta atômica e verifique se o trecho necessário cabe no orçamento efetivo. Modelos de decisão não fazem busca e não redigem por conta própria uma checagem de fatos.[1][5][11]

## Como comparar honestamente no próximo experimento

1. Fixar uma amostra pequena de mensagens, notícias ou trechos em português, com respostas humanas e exemplos ambíguos. Manter um teste isolado antes de ajustar prompts, regras ou temperatura.
2. Definir o mesmo `state`, texto da pergunta, alternativas, ordem e critério de acerto para Laya, Kev, decider e SemIf. Onde uma primitiva não coincidir, comparar apenas a parte equivalente; não converter `score` ordinal em `noul` sem declarar mudança de tarefa.
3. Registrar revisão dos modelos, tokenizer, quantização, runtime, hardware, semente quando aplicável, tokens efetivamente consumidos, tempo de carga e tempo de decisão. Manter saída bruta, ação proposta e eventual intervenção de regras separadas.
4. Avaliar acurácia por tipo de caso e idioma; conferir probabilidades por faixa de confiança, erros graves, mudança na ordem de alternativas e impacto da seleção/truncamento de trechos. Não estabelecer limiar de atuação automática sem dados da própria tarefa.
5. Não instalar nem substituir o ambiente Laya existente só para comparar. Usar ambiente e cache separados após decidir escopo, downloads e espaço em disco; manter Snake e seus testes intactos.

**Estado desta pesquisa:** verificação documental e inventário de hardware apenas. Os números externos acima são resultados publicados por seus autores ou pelo JevBench, com configurações diferentes. Não houve download ou inferência das alternativas listadas neste notebook.

## Sources

[1] https://github.com/jaredpalmer/kev
[2] https://huggingface.co/jaredpalmer/kev-4b/raw/main/README.md
[3] https://huggingface.co/jaredpalmer/kev-0.8b/raw/main/README.md
[4] https://huggingface.co/jaredpalmer/kev-9b/raw/main/README.md
[5] https://github.com/TheoLeeCJ/SemIf
[6] https://raw.githubusercontent.com/TheoLeeCJ/SemIf/master/docs/CALIBRATION.md
[7] https://raw.githubusercontent.com/TheoLeeCJ/SemIf/master/docs/REPRODUCE.md
[8] https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/tree/main
[9] https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/raw/main/core/engine_torch.py
[10] https://huggingface.co/harshatheg/Qwen-2.5-1B-RLCD/raw/main/MODEL_CARD.md
[11] https://github.com/NandhaKishorM/laya
[12] https://huggingface.co/Qwen/Qwen3.5-4B/raw/main/README.md
[13] https://benchmarkheaven.com/jev-models
[14] https://classifier.dev/benchmark
[15] https://classifier.dev/pricing
[16] https://github.com/Mapika/decider
[17] https://huggingface.co/Mapika/decider-0.8b/tree/main
[18] https://github.com/logan-markewich/jeff
[19] https://github.com/kshetrajna12/reflex
[20] https://github.com/Davipar/djev-dev
[21] https://github.com/mithalouni/system-one-open
[22] https://huggingface.co/akhilaaa3/Jev-Omni
[23] https://huggingface.co/akhilaaa3/Jev-Omni/tree/main
[24] https://huggingface.co/akhilaaa3/Jev-Omni/blob/main/jev_omni.py
[25] https://huggingface.co/datasets/akhilaaa3/decision-bench
[26] https://huggingface.co/akhilaaa3/Jev-Omni/blob/main/decision_config.json
