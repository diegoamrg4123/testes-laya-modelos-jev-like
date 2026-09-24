# Gravações selecionadas

Os arquivos deste diretório são capturas JSONL de inferências reais feitas durante os experimentos locais descritos em [`../DEMO.md`](../DEMO.md). Não são dados simulados nem um benchmark estatístico. Cada gravação inclui metadados do backend e checkpoint, configurações, estados antes das ações, decisões e resumo final.

| Arquivo | Experimento |
| --- | --- |
| `snake-120.jsonl` | Laya multilíngue, 120 decisões no tabuleiro padrão. |
| `snake-20.jsonl` | Laya multilíngue, 20 decisões em tabuleiro 8 × 6. |
| `laya-cuda-3.jsonl` | Laya multilíngue, 3 decisões para a comparação curta. |
| `kev-3-rastreado.jsonl` | Kev-0.8B, 3 decisões pelo servidor local. |
| `semif-3-rastreado.jsonl` | SemIf-4B quantizado, 3 decisões na CPU. |

Os experimentos de três decisões servem para confirmar que os backends e adaptadores funcionaram nessa execução. O tamanho da amostra não permite concluir qual modelo é melhor. Veja `DEMO.md` para hardware, versões, método de medição e limitações.

Ao gerar novos registros, escolha um nome de arquivo novo. A opção `--record` cria o arquivo sem sobrescrever um registro existente. Pesos e arquivos de log não são mantidos aqui.
