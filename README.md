# bbe-phase-correction
Sistema Python de correção de fase de áudio (estilo BBE/BMX). Utiliza filtros passa-tudo (All-Pass) IIR biquad em cascata para atrasar os graves, alinhando seu impacto temporal com o ataque dos agudos sem alterar a magnitude. Inclui resolução numérica e controles shelf de equalização.

Aqui está a descrição completa e pronta para colar no seu README.md, incluindo a informação de que o sistema funciona bem em mono ou estéreo e a formatação correta para a imagem:
# BBE / BMX Phase Correction System (Python)

Este projeto implementa um sistema de correção de fase e aprimoramento harmônico de áudio em Python. O objetivo principal é alinhar no tempo o impacto dos graves com o ataque dos agudos, melhorando a percepção, a clareza e a definição sonora. O sistema funciona perfeitamente tanto com áudio mono quanto estéreo.

## Visualização do Alinhamento de Fase


![Diagrama de Alinhamento de Fase](https://github.com/user-attachments/assets/2a0ad239-0a63-4b2d-9751-e425bc6d725b)


## Como Funciona

O sistema utiliza uma rede de filtros passa-tudo (All-Pass) em cascata para manipular o tempo do sinal. Diferente do atraso de bandas separadas comum em saídas de crossovers—que cria cancelamentos de fase e vales de até 10 dB próximos às frequências de corte—a rede passa-tudo mantém a magnitude da frequência exatamente plana. O atraso de grupo gerado é dependente da frequência, aplicando um atraso maior nas frequências graves.

## Funcionalidades Principais

* **Alinhamento Numérico de Fase:** O algoritmo resolve iterativamente as frequências e o fator Q das seções passa-tudo para garantir que o atraso de grupo medido corresponda ao atraso configurado (`low_ms` em 80 Hz e `mid_ms` em 600 Hz, ambos relativos à referência de agudos em 6 kHz).
* **Filtros Biquad IIR (SOS):** O motor de áudio utiliza as receitas "Audio EQ Cookbook" de RBJ, estruturado em cascata de seções de 2ª ordem (SOS). O estado do filtro é preservado entre os blocos para evitar estalos no processamento contínuo em `float64`.
* **Definição e Ar:** Controle de *high-shelf* (0.0 a 1.0) que adiciona até +6 dB de ganho em 7 kHz.
* **Reforço de Graves:** Controle de *low-shelf* (0.0 a 1.0) que adiciona até +5 dB de ganho em 90 Hz.

## Estrutura e Dependências

O módulo é composto por dois arquivos interligados:
* `bbe.py`: Classe principal responsável pelo design e roteamento da cascata passa-tudo e calibração de atrasos.
* `biquad.py`: Funções DSP essenciais (peaking, shelves, lowpass, highpass, allpass) e processamento do sinal `SosFilter`.

O processamento matemático exige as seguintes bibliotecas científicas:
```bash
pip install numpy scipy
```





