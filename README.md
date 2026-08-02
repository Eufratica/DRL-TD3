# 🤖 Navegação Autônoma de Robôs Móveis com TD3 (Twin Delayed DDPG) em ROS & Gazebo

Este repositório contém a implementação completa em **PyTorch**, **ROS (Robot Operating System)** e **Gazebo** do algoritmo de Aprendizado por Reforço Profundo (**Deep Reinforcement Learning - DRL**) **TD3 (Twin Delayed Deep Deterministic Policy Gradient)** aplicado à navegação autônoma de robôs móveis utilizando sensores **LiDAR 3D (Velodyne)**.

---

## 📋 Sumário

- [1. Visão Geral da Arquitetura do Sistema](#1-visão-geral-da-arquitetura-do-sistema)
- [2. Fundamentos Matemáticos do DRL](#2-fundamentos-matemáticos-do-drl)
  - [2.1 O Processo de Decisão de Markov (MDP)](#21-o-processo-de-decisão-de-markov-mdp)
  - [2.2 Retorno Futuro Descontado](#22-retorno-futuro-descontado-g_t)
  - [2.3 Funções de Valor $V(s)$ e $Q(s, a)$](#23-funções-de-valor-vs-e-qsa)
  - [2.4 Equação de Otimização de Bellman](#24-equação-de-otimização-de-bellman)
  - [2.5 Gradiente de Política Determinística (DPG)](#25-gradiente-de-política-determinística-dpg)
- [3. Os 3 Pilares do Algoritmo TD3](#3-os-3-pilares-do-algoritmo-td3)
  - [Pilar 1: Clipped Double Q-Learning](#pilar-1-clipped-double-q-learning-combate-à-superestimação)
  - [Pilar 2: Target Policy Smoothing](#pilar-2-target-policy-smoothing-suavização-da-política-alvo)
  - [Pilar 3: Delayed Policy Updates e Soft Updates](#pilar-3-delayed-policy-updates-e-soft-updates-polyak-averaging)
- [4. Mapeamento Físico e Função de Recompensa](#4-mapeamento-físico-e-função-de-recompensa)
  - [4.1 Mapeamento de Ações](#41-mapeamento-de-ações)
  - [4.2 Função de Recompensa Matemático-Programática](#42-função-de-recompensa-matemático-programática)
  - [4.3 Decaimento do Ruído de Exploração](#43-decaimento-do-ruído-de-exploração)
- [5. Análise Detalhada dos Arquivos de Código](#5-análise-detalhada-dos-arquivos-de-código)
  - [5.1 `velodyne_env.py` (Ambiente ROS / Gazebo)](#51-velodyne_envpy-ambiente-ros--gazebo)
  - [5.2 `replay_buffer.py` (Memória Off-Policy)](#52-replay_bufferpy-memória-off-policy)
  - [5.3 `train.py` (Treinamento Completo do TD3)](#53-trainpy-treinamento-completo-do-td3)
  - [5.4 `test.py` (Inferência e Avaliação)](#54-testpy-inferência-e-avaliação)
- [6. Tabela Tática de Referência de Equações e Símbolos](#6-tabela-tática-de-referência-de-equações-e-símbolos)

---

## 1. Visão Geral da Arquitetura do Sistema

O sistema opera em um ciclo de controle fechado distribuído em 4 arquivos principais:

---

## 2. Fundamentos Matemáticos do DRL

### 2.1 O Processo de Decisão de Markov (MDP)

O problema de navegação do robô é formulado matematicamente como um MDP contínuo definido pela tupla $\mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)$:

* $\mathcal{S} \subset \mathbb{R}^{24}$: Espaço contínuo de estados de 24 dimensões.
* $\mathcal{A} \subset [-1, 1]^2$: Espaço contínuo de ações de 2 dimensões.
* $\mathcal{P}(s_{t+1} \mid s_t, a_t)$: Função de probabilidade de transição do ambiente simulado.
* $\mathcal{R}(s_t, a_t, s_{t+1})$: Função de recompensa escalar do agente.
* $\gamma \in [0, 1)$: Fator de desconto para recompensas futuras.

---

### 2.2 Retorno Futuro Descontado ($G_t$)

O objetivo primário do agente é maximizar o retorno futuro acumulado a partir do instante de tempo $t$:

$$G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1} = r_{t+1} + \gamma r_{t+2} + \gamma^2 r_{t+3} + \dots$$

#### Explicação dos Símbolos:
* $G_t$: Retorno acumulado total a partir do tempo $t$.
* $\sum_{k=0}^{\infty}$: Somatório infinito sobre todos os passos futuros do episódio.
* $r_{t+k+1}$: Recompensa escalar recebida no passo futuro $t+k+1$.
* $\gamma$ (*gamma*): Fator de desconto no intervalo $[0, 1)$. Determina o impacto das recompensas futuras. Se $\gamma = 0$, o agente é imediatista; se $\gamma \to 1$, valoriza metas de longo prazo.

---

### 2.3 Funções de Valor $V(s)$ e $Q(s, a)$

#### A. Função de Valor de Estado $V^\pi(s)$
Mede o retorno esperado estando no estado $s$ sob uma política $\pi$:
$$V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right]$$

#### B. Função de Valor de Ação $Q^\pi(s, a)$
Mede o retorno esperado ao executar a ação $a$ no estado $s$ e continuar seguindo a política $\pi$:
$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]$$

#### Explicação dos Símbolos:
* $Q^\pi(s, a)$: Valor escalar estimado para o par estado-ação sob a política $\pi$.
* $\mathbb{E}_\pi [\cdot]$: Valor esperado estatístico (esperança matemática) sob a política $\pi$.
* $s_t = s$: Condição do estado no tempo $t$ ser $s$.
* $a_t = a$: Condição da ação no tempo $t$ ser $a$.

---

### 2.4 Equação de Otimização de Bellman

A política ótima $\pi^*$ é aquela que atinge o maior valor $Q$ em todos os estados. A Equação de Otimização de Bellman decompõe recursivamente a função $Q^*$:

$$Q^*(s, a) = r + \gamma \max_{a'} Q^*(s', a')$$

#### Explicação dos Símbolos:
* $Q^*(s, a)$: O valor $Q$ ótimo (máximo retorno teórico possível).
* $r$: Recompensa imediata obtida ao transitar do estado $s$ para $s'$ via ação $a$.
* $\gamma$: Fator de desconto temporal.
* $\max_{a'}$: Operador que escolhe a ação $a'$ no próximo estado $s'$ que maximiza o valor $Q^*$.
* $s'$: Próximo estado ($s_{t+1}$).
* $a'$: Próxima ação ($a_{t+1}$).

---

### 2.5 Gradiente de Política Determinística (DPG)

Em espaços de ação contínuos, usar $\max_{a'}$ a cada passo de simulação é computacionalmente inviável. O DPG utiliza uma rede neural **Ator determinístico** $\mu_\phi(s)$ que produz a ação direta $a = \mu_\phi(s)$, atualizando os parâmetros $\phi$ do Ator através do gradiente de desempenho $J(\phi)$:

$$\nabla_\phi J(\phi) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \left. \nabla_a Q_\theta(s, a) \right\vert{}_{a=\mu_\phi(s)} \cdot \nabla_\phi \mu_\phi(s) \right]$$

#### Explicação dos Símbolos:
* $\nabla_\phi J(\phi)$: Gradiente do desempenho $J$ em relação aos parâmetros $\phi$ do Ator.
* $\mathbb{E}_{s \sim \mathcal{D}}$: Média esperada sobre estados $s$ amostrados do Replay Buffer $\mathcal{D}$.
* $\nabla_a Q_\theta(s, a) \mid_{a=\mu_\phi(s)}$: Derivada parcial do valor $Q$ (do Crítico $\theta$) em relação à ação $a$, avaliada na ação escolhida pelo Ator $\mu_\phi(s)$.
* $\nabla_\phi \mu_\phi(s)$: Derivada parcial da ação produzida em relação aos pesos $\phi$ da rede Ator.

---

## 3. Os 3 Pilares do Algoritmo TD3

---

### Pilar 1: Clipped Double Q-Learning (Combate à Superestimação)

#### Teoria e Equação
Devido ao uso de redes neurais, ocorre a superestimação sistemática do valor Q. O TD3 mantém dois Críticos ($Q_{\theta_1}$ e $Q_{\theta_2}$) e duas redes Alvo ($Q_{\theta_{\text{target}, 1}}$ e $Q_{\theta_{\text{target}, 2}}$), definindo o Alvo Temporal $y$ através do **mínimo** entre eles:

$$y = r + \gamma (1 - d) \min_{i \in \{1, 2\}} Q_{\theta_{\text{target}, i}}(s', a')$$

A função de perda do erro quadrático médio (MSE) para atualizar os dois Críticos Principais é dada por:

$$L(\theta_1, \theta_2) = \frac{1}{\vert{}B\vert{}} \sum_{(s, a, r, d, s') \in B} \left[ \left( Q_{\theta_1}(s, a) - y \right)^2 + \left( Q_{\theta_2}(s, a) - y \right)^2 \right]$$

#### Explicação dos Símbolos:
* $y$: Alvo de Diferença Temporal (TD Target).
* $d \in \{0, 1\}$: Indicador de término do episódio (`done`). Se $d=1$, $(1-d)=0$ e cancela os retornos futuros.
* $\min_{i \in \{1,2\}}$: Função de seleção do menor valor escalar entre os dois Críticos Alvo.
* $Q_{\theta_{\text{target}, i}}(s', a')$: Previsão do valor $Q$ do $i$-ésimo Crítico Alvo.
* $\vert{}B\vert{}$: Tamanho do mini-batch extraído do Replay Buffer (ex: $40$).

#### Analogia do Mundo Real: O Conselho de Dois Engenheiros Conservadores
Se você quer saber a capacidade de carga de uma ponte e consulta dois engenheiros:
* O **Engenheiro 1** (Crítico 1) diz que a ponte aguenta 100 toneladas.
* O **Engenheiro 2** (Crítico 2) é mais conservador e diz que aguenta 70 toneladas.
* O TD3 adota a estimativa mínima de **70 toneladas** ($y = \min(Q_1, Q_2)$), evitando que a estrutura colapse por excesso de otimismo.

#### No Código (`train.py`)
```python
target_Q1, target_Q2 = self.critic_target(next_state, next_action)
target_Q = torch.min(target_Q1, target_Q2) # Pega o mínimo
target_Q = reward + ((1 - done) * discount * target_Q).detach()

current_Q1, current_Q2 = self.critic(state, action)
loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
