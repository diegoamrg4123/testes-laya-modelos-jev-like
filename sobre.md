# Laya: alternativa de código aberto ao Jev

Para comparar Kev, SemIf e o repositório `Qwen-2.5-1B-RLCD` com o Laya, consulte [alternativas-ao-jev.md](alternativas-ao-jev.md).

Links indicados por Diego:

- [Apresentação do Jev pela TypeSafe](https://typesafe.ai/blog/introducing-system-one-models-and-jev)
- [Repositório do Laya](https://github.com/NandhaKishorM/laya)

> Pesquisa de setembro de 2026. Os exemplos abaixo seguem a documentação, mas **não foram executados neste notebook**. Benchmarks citados são publicados pelos próprios projetos, não testes independentes feitos aqui.

## Em poucas palavras

**Laya** é uma família de modelos de decisão da Convai Innovations, com pesos e código disponibilizados sob Apache 2.0. Recebe um `state` (texto ou dicionário) e perguntas delimitadas em `questions`; retorna decisões tipadas e probabilidades. É possível executá-lo localmente em Python.[1][2]

**Jev** é o modelo System One da TypeSafe AI, oferecido por API em acesso antecipado nas fontes consultadas. Usa o mesmo paradigma de estado mais perguntas tipadas, mas **não é o mesmo produto nem o mesmo modelo**. Laya é uma alternativa aberta inspirada nessa proposta, não um SDK oficial do Jev.[1][4][5]

Ambos são úteis para classificar, pontuar e encaminhar casos. Não geram notícias, explicações ou resumos; para isso, combine-os com código convencional, ferramentas de pesquisa e, quando necessário, um modelo generativo. Uma saída com formato válido ainda pode ser uma decisão errada.[1][4]

## Como o Laya funciona

1. **Estado:** o objeto a avaliar, por exemplo uma mensagem, um ticket ou um texto com título e corpo.
2. **Perguntas:** cada uma contém instruções e, quando necessário, alternativas ou níveis definidos pelo desenvolvedor.
3. **Inferência:** um encoder lê o conteúdo; a cabeça de decisão pontua as opções. As perguntas são processadas em lote, sem redigir a resposta token por token.[2]
4. **Resultado:** seu programa consulta as probabilidades e decide se classifica, encaminha para revisão humana ou pede outra análise.[1][2]

| Tipo | O que devolve | Exemplo de uso |
| --- | --- | --- |
| `choice` | Uma opção entre critérios definidos, distribuição e confiança | Escolher editoria |
| `score` | Posição numa escala ordinal, distribuição e confiança | Estimar prioridade |
| `noul` | Probabilidade de uma afirmação ser verdadeira, entre 0 e 1 | Detectar indício de urgência |

O checkpoint inglês usa ModernBERT-large e uma cabeça de decisão treinada para pontuar marcadores das opções. O projeto também oferece checkpoint multilíngue com mmBERT-base e outro ajustado para tarefas específicas de decisões tipadas. A documentação descreve treinamento denominado RLCD (*Reinforcement Learning for Calibrated Decisions*) e ajuste posterior de temperatura para calibração.[1][2] Calibração é uma propriedade estatística de conjuntos de previsões, não garantia de que uma previsão individual esteja correta.

### Qual checkpoint escolher?

| Checkpoint | Uso inicial recomendado | Contexto divulgado |
| --- | --- | --- |
| `convaiinnovations/laya` | Texto em inglês | 512 tokens por pergunta |
| `multilingual` | Português e outros idiomas | 1.024 tokens por pergunta |
| `typed-decisions` | Explorar tarefas semelhantes às de seu ajuste fino | 1.024 tokens por pergunta |

O `Router` pode identificar o idioma e selecionar o modelo. Para um teste **só em português**, prefira carregar diretamente `multilingual` em vez de pré-carregar todos. Se houver tráfego alternado entre idiomas, investigue `Router(preload=True)`, ciente de que manter vários checkpoints residentes consome mais memória.[1][2]

## Primeiro experimento no Zorin OS

O projeto pede Python 3.10 ou superior na documentação do repositório. Instalar o pacote não é o mesmo que baixar os pesos, que são obtidos ao carregar o modelo pela primeira vez. Use ambiente virtual para não mexer no Python global.[1][3]

```bash
# Execute na raiz do repositório
python3 --version
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install laya
```

O pacote instala dependências como PyTorch e Transformers; o download e a execução podem consumir memória, armazenamento e tempo. Estes comandos **não foram executados** nesta pesquisa. Uma alternativa sem instalação é a [demo no Hugging Face](https://huggingface.co/spaces/convaiinnovations/laya-demo). Não envie a uma demo pública textos confidenciais.[1][3]

Exemplo de `teste_laya.py` para um texto fictício em português:

```python
import laya

# Primeiro carregamento: baixa o checkpoint multilíngue.
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")

state = {
    "titulo": "Mensagem falsa sobre atualização do banco",
    "texto": "A mensagem pede acesso a um link e informa que a senha deve ser digitada."
}
questions = {
    "editoria": {
        "type": "choice",
        "instructions": "Qual editoria descreve melhor este texto?",
        "criteria": {
            "seguranca": "fraudes, golpes e proteção digital",
            "games": "jogos e consoles",
            "hardware": "componentes e dispositivos",
            "outros": "assuntos fora dessas editorias"
        }
    },
    "alerta": {
        "type": "noul",
        "instructions": "O texto descreve uma tentativa de golpe?"
    },
    "prioridade": {
        "type": "score",
        "instructions": "Qual é a prioridade aparente de revisão editorial?",
        "criteria": ["baixa", "média", "alta"]
    }
}

result = agent.predict(state, questions)
print(result["answers"])
```

Depois de salvar o exemplo, rode `python teste_laya.py` com o ambiente ativo. **Nenhuma saída foi simulada aqui.** Confira a estrutura real de `answers`, incluindo `choice`, `confidence`, `score` e `noul`. `score` representa posição na escala definida, não porcentagem.[1][2]

**Experimento útil:** separe uma amostra de textos previamente rotulados por você, rode as três perguntas e compare as saídas com os rótulos humanos. Examine erros por editoria e por confiança. Só então estabeleça limiares de revisão humana. Classificação de assunto e detecção de indício de golpe **não equivalem à checagem de fatos**. Não publique acusações nem execute decisões irreversíveis com base apenas no modelo.

## Limitações que importam

- Entradas podem ser truncadas pelo orçamento de tokens e pela divisão interna do espaço entre pergunta/opções e estado.[2]
- O checkpoint raiz é voltado ao inglês. O multilíngue é a escolha inicial para português, mas também precisa de avaliação no seu corpus.[2]
- Muitas opções de `choice` podem comprimir a descrição de cada alternativa e prejudicar a precisão; `score` foi uma primitiva fraca em uma avaliação publicada pelo projeto.[1][2]
- A calibração divulgada reflete conjuntos de avaliação específicos. O próprio README informa que checkpoints básicos foram fracos, sem ajuste fino, num benchmark de decisões tipadas; a vantagem do checkpoint especializado não se transfere automaticamente a novos domínios.[1][2]
- Velocidade anunciada em GPU não equivale à latência completa no notebook, especialmente no primeiro carregamento. Executar localmente evita enviar o texto à API do Jev, mas o primeiro download do modelo ainda requer conexão.[1][2]

## Jev: o que é e como se compara

A TypeSafe define Jev como modelo para avaliar perguntas `Choice`, `Score` e `Noul` em paralelo contra um estado. `Choice` e `Score` incluem confiança; `Noul` devolve a probabilidade de verdadeiro. O código da aplicação pode usar esses valores para ramificar, ordenar e encaminhar casos.[4] A empresa divulga latências de 70 a 500 ms e preço de US$ 0,042 por milhão de tokens de entrada em sua apresentação inicial, mas esses são valores anunciados pelo fornecedor e podem mudar.[5]

| Aspecto | Laya | Jev |
| --- | --- | --- |
| Responsável | Convai Innovations | TypeSafe AI |
| Acesso | Pesos e código abertos, uso local possível | Serviço proprietário por API |
| Português | Checkpoint multilíngue divulgado | É preciso verificar suporte e desempenho na documentação e em testes |
| Custos | Sem tarifa por inferência local, com custo de hardware/energia | Tarifação de API anunciada |
| Uso típico | Experimentação, execução local e especialização | Integração hospedada para decisões tipadas |

Não tome comparações de velocidade e acurácia feitas com conjuntos, versões ou máquinas diferentes como prova de superioridade. O próprio repositório do Laya aponta limites de generalização e casos em que Jev leva vantagem, especialmente escolhas com muitas categorias. Compare os dois com **a mesma amostra, perguntas e métricas** antes de escolher para produção.[1][2]

## Fontes

## Sources

[1] https://github.com/NandhaKishorM/laya
[2] https://huggingface.co/convaiinnovations/laya/blob/main/README.md
[3] https://pypi.org/project/laya/0.3.5
[4] https://docs.typesafe.ai/introduction.md
[5] https://typesafe.ai/blog/introducing-system-one-models-and-jev
