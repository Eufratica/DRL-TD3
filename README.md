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

## 3. As Funções de Valor $V(s)$ e $Q(s, a)$

### A. Função de Valor de Estado $V^\pi(s)$

Mede o retorno esperado estando no estado $s$ sob uma política $\pi$:

$$
V^\pi(s) = \mathbb{E}_\pi \left[ G_t \mid s_t = s \right]
$$

### B. Função de Valor de Ação $Q^\pi(s, a)$ (Função Q)

Mede o retorno esperado ao executar a ação $a$ no estado $s$ e continuar seguindo a política $\pi$:

$$
Q^\pi(s, a) = \mathbb{E}_\pi \left[ G_t \mid s_t = s, a_t = a \right]
$$

### Explicação de cada símbolo

- **$Q^\pi(s, a)$**: Valor numérico (escalar) estimado para o par estado-ação sob a política $\pi$.  
- **$\mathbb{E}_\pi [\cdot]$**: Valor esperado estatístico (esperança matemática) seguindo as decisões da política $\pi$.  
- **$s_t = s$**: Condição do estado no tempo $t$ ser $s$.  
- **$a_t = a$**: Condição da ação no tempo $t$ ser $a$.
---

### 2.4. A Equação de Otimização de Bellman para Q*(s, a)

A política ótima π* é aquela que atinge o maior valor Q em todos os estados. A Equação de Otimização de Bellman decompõe recursivamente a função Q*:

<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbg%7Btransparent%7D%5Ccolor%7Bwhite%7D%5Clarge%20Q%5E*%28s%2Ca%29%3D%5Cmathcal%7BR%7D%28s%2Ca%29%2B%5Cgamma%5Cint_%7B%5Cmathcal%7BS%7D%7D%5Cmathcal%7BP%7D%28s%27%5Cmid%20s%2Ca%29%5Cmax_%7Ba%27%7DQ%5E*%28s%27%2Ca%27%29%5C%2Cds%27" alt="Equação de Bellman 1">
</div>

Em ambientes discretos ou amostrados do Replay Buffer, escreve-se:

<div align="center">
  <img src="https://latex.codecogs.com/svg.latex?%5Cbg%7Btransparent%7D%5Ccolor%7Bwhite%7D%5Clarge%20Q%5E*%28s%2Ca%29%3Dr%2B%5Cgamma%5Cmax_%7Ba%27%7DQ%5E*%28s%27%2Ca%27%29" alt="Equação de Bellman 2">
</div>

#### Explicação de cada símbolo

- **Q*(s, a)**: O valor Q ótimo (máximo retorno teórico possível).  
- **r**: Recompensa imediata obtida ao transitar do estado s para s' via ação a.  
- **γ**: Fator de desconto temporal.  
- **max a'**: Operador que escolhe a ação a' no próximo estado s' que maximiza o valor Q*.  
- **s'**: Próximo estado (s_{t+1}).  
- **a'**: Próxima ação (a_{t+1}).
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

```
## Pilar 2: Target Policy Smoothing (Suavização da Política Alvo)

### Teoria e Equação

Para regularizar a superfície de valor $Q$ e evitar picos de recompensa em ações isoladas, adiciona-se um ruído gaussiano cortado (*clipped noise*) à próxima ação $a'$ calculada pelo Ator Alvo:

$$a' = \text{clip}\left( \mu_{\phi_{\text{target}}}(s') + \tilde{\epsilon}, \, a_{\min}, \, a_{\max} \right)$$

Onde a perturbação $\tilde{\epsilon}$ é definida por:

$$\tilde{\epsilon} = \text{clip}\left( \epsilon, \, -c, \, c \right), \quad \epsilon \sim \mathcal{N}(0, \sigma^2)$$

### Explicação dos Símbolos

* **$a'$**: Próxima ação suavizada enviada aos Críticos Alvo.
* **$\text{clip}(x, a_{\min}, a_{\max})$**: Função que limita $x$ dentro do intervalo $[a_{\min}, a_{\max}]$.
* **$\tilde{\epsilon}$**: Ruído gaussiano truncado.
* **$\epsilon$**: Amostra de uma distribuição normal com média zero e desvio padrão $\sigma^2$ (policy_noise = 0.2).
* **$-c, c$**: Limites de corte do ruído (noise_clip = 0.5).

---

### Analogia do Mundo Real: Testar o Veículo em Piso com Trepidação

Ajustar a direção do carro apenas em um asfalto plano torna-o vulnerável a qualquer pequena imperfeição na pista real. Adicionar vibração durante o teste ($\tilde{\epsilon}$) força o sistema a ser estável não apenas na trajetória ideal, mas em toda a vizinhança do movimento.


No Código (train.py)
```python
next_action = self.actor_target(next_state)

# Gera ruído gaussiano N(0, policy_noise)
noise = torch.Tensor(batch_actions).data.normal_(0, policy_noise).to(device)

# Trunca o ruído no intervalo [-noise_clip, noise_clip]
noise = noise.clamp(-noise_clip, noise_clip)

# Injeta o ruído e trunca nos limites [-max_action, max_action]
next_action = (next_action + noise).clamp(-self.max_action, self.max_action)
```





## Pilar 3: Delayed Policy Updates e Soft Updates (Polyak Averaging)

### Teoria e Equações

**Atraso na Atualização (Delayed Updates):** O Ator $\phi$ e as redes Alvo ($\phi_{\text{target}}, \theta_{\text{target}}$) são atualizados apenas a cada $d$ iterações dos Críticos ($d = 2$).

**Atualização Suave (Polyak Averaging):** As redes Alvo atualizam seus parâmetros via interpolação convexa com taxa $\tau \ll 1$ ($\tau = 0.005$):

$$\theta_{\text{target}, i} \leftarrow \tau \theta_i + (1 - \tau) \theta_{\text{target}, i}, \quad \text{para } i \in \{1, 2\}$$

$$\phi_{\text{target}} \leftarrow \tau \phi + (1 - \tau) \phi_{\text{target}}$$

### Explicação dos Símbolos

* **$\theta_{\text{target}, i}$**: Pesos da rede neural do $i$-ésimo Crítico Alvo.
* **$\theta_i$**: Pesos da rede neural do $i$-ésimo Crítico Principal.
* **$\phi_{\text{target}}$**: Pesos da rede neural do Ator Alvo.
* **$\phi$**: Pesos da rede neural do Ator Principal.
* **$\tau$ (tau)**: Taxa de atualização suave ($0.005$).

---

### Analogia do Mundo Real: Absorver Matéria aos Poucos Antes da Prova

O Crítico é o estudo diário de exercícios. O Ator é a mudança de estratégia na prova. O aluno não altera sua estratégia a cada exercício resolvido; ele estuda 2 capítulos inteiros primeiro (Atraso $d=2$). Ao absorver conhecimento novo, incorpora 0.5% ($\tau = 0.005$) por dia na sua memória de longo prazo (redes Alvo).


No Código (train.py)

```python
if it % policy_freq == 0:
    # 1. Atualização do Ator Principal
    actor_loss = -self.critic(state, self.actor(state))[0].mean()
    self.actor_optimizer.zero_grad()
    actor_loss.backward()
    self.actor_optimizer.step()

    # 2. Soft Update do Ator Alvo
    for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)

    # 3. Soft Update dos Críticos Alvo
    for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
        target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
```

## 4. Mapeamento Físico e Função de Recompensa

### 4.1 Mapeamento de Ações

As saídas da rede Ator $a_0, a_1 \in [-1, 1]$ geradas pela função $	anh$ são mapeadas para os comandos reais de velocidade linear ($v \in [0, 1]$ m/s) e angular ($\omega \in [-1, 1]$ rad/s):

$$v = \frac{a_0 + 1}{2}, \quad \omega = a_1$$

**Explicação dos Símbolos:**
* **$a_0 \in [-1, 1]$**: Saída do primeiro neurônio do Ator.
* **$a_1 \in [-1, 1]$**: Saída do segundo neurônio do Ator.
* **$v \in [0, 1]$**: Velocidade linear em m/s (garante avanço sem ré).
* **$\omega \in [-1, 1]$**: Velocidade angular em rad/s.

---

### 4.2 Função de Recompensa Matemático-Programática

A recompensa instantânea $\mathcal{R}(s, a, s')$ é dada por:

$$\mathcal{R}(s, a, s') = \begin{cases} +100.0, & \text{se } d_{\text{goal}} < D_{\text{target}} \quad (\text{chegou ao objetivo}) \\ -100.0, & \text{se } \min(\mathbf{z}_{\text{laser}}) < D_{\text{collision}} \quad (\text{colidiu}) \\ \frac{a_0}{2} - \frac{|a_1|}{2} - \frac{f(\min(\mathbf{z}_{\text{laser}}))}{2}, & \text{caso contrário} \end{cases}$$

Onde $f(x)$ penaliza a proximidade de obstáculos:

$$f(x) = \begin{cases} 1 - x, & \text{se } x < 1.0 \\ 0.0, & \text{se } x \ge 1.0 \end{cases}$$

**Explicação dos Símbolos:**
* **$d_{\text{goal}}$**: Distância euclidiana atual do robô até o alvo.
* **$D_{\text{target}} = 0.3$m**: Tolerância para objetivo atingido.
* **$\mathbf{z}_{\text{laser}} \in \mathbb{R}^{20}$**: Vetor de 20 leituras do LiDAR Velodyne.
* **$D_{\text{collision}} = 0.35$m**: Limiar para detecção de colisão.
* **$a_0$**: Recompensa proporcional à velocidade linear.
* **$|a_1|$**: Penalidade proporcional ao módulo da rotação.

---

### 4.3 Decaimento do Ruído de Exploração

O desvio padrão do ruído de exploração $\sigma_t$ decai linearmente ao longo do tempo:

$$\sigma_t = \max\left( \sigma_{\min}, \, \sigma_{t-1} - \frac{1 - \sigma_{\min}}{N_{\text{decay}}} \right)$$

$$a_{\text{exploração}} = \text{clip}\left( \mu_\phi(s) + \mathcal{N}(0, \sigma_t^2), \, -1, \, 1 \right)$$

**Explicação dos Símbolos:**
* **$\sigma_t$**: Desvio padrão do ruído de exploração no passo $t$.
* **$\sigma_{\min} = 0.1$**: Ruído mínimo mantido ao fim do decaimento (expl_min).
* **$N_{\text{decay}} = 500.000$**: Número total de passos para decaimento (expl_decay_steps).

---

## 5. Análise Detalhada dos Arquivos de Código

### 5.1 velodyne_env.py (Ambiente ROS / Gazebo)

* **check_pos(x, y)**: Verifica se uma coordenada $(x, y)$ está em zona livre no mapa.
* **GazeboEnv.__init__()**: Lança roscore e roslaunch, inicializa o nó ROS, cria o publisher `/r1/cmd_vel`, conecta-se aos leitores do Velodyne e Odometria, e cria proxies para congelar/descongelar a física do Gazebo.
* **velodyne_callback(v)**: Processa a nuvem 3D (`pc2.read_points`), calcula o ângulo polar $\beta$ e a distância 3D, mapeando os pontos em 20 fatias angulares (environment_dim = 20) e salvando a menor distância por fatia.
* **step(action)**: Envia comando Twist ao robô; executa `unpause()`; aguarda TIME_DELTA = 0.1s e executa `pause()`; lê Odometria e calcula posição $(x, y)$ e ângulo Yaw; calcula distância e ângulo relativo $\theta$ até o objetivo; monta o vetor de estado $s_{t+1}$ de 24 dimensões; chama `get_reward()` e retorna `(state, reward, done, target)`.
* **reset()**: Executa `/gazebo/reset_world`, sorteia uma posição inicial livre para o robô, chama `change_goal()` para sortear um novo alvo e `random_box()` para reordenar 4 caixas pelo mapa.

---

### 5.2 replay_buffer.py (Memória Off-Policy)

* **ReplayBuffer.__init__()**: Cria a fila circular `deque()` com capacidade para 1.000.000 transições.
* **add(s, a, r, t, s2)**: Armazena $(s, a, r, \text{done}, s')$. Remove a memória mais antiga via `popleft()` ao atingir a capacidade máxima.
* **sample_batch(batch_size)**: Sorteia $N=40$ transições aleatórias (`random.sample`) e desmembra a lista de tuplas em 5 matrizes NumPy isoladas.

---

### 5.3 train.py (Treinamento Completo do TD3)

* **Actor**: Rede neural da política $\mu_\phi(s)$: 24 $\to$ 800 (ReLU) $\to$ 600 (ReLU) $\to$ 2 (Tanh).
* **Critic**: Classe contendo os Críticos Gêmeos $Q_{\theta_1}(s, a)$ e $Q_{\theta_2}(s, a)$ em um único módulo.
* **TD3.train()**: Executa a amostragem do buffer, injeção de ruído cortado (Pilar 2), seleção do menor valor Q (Pilar 1), cálculo do erro MSE e atualização atrasada do Ator com soft update ($\tau = 0.005$) (Pilar 3).
* **evaluate()**: Roda 10 episódios limpos a cada 5.000 passos para medir a recompensa média e taxa de colisão, salvando os modelos `.pth`.
* **Trava Antiparede (random_near_obstacle)**: Se a menor leitura do laser for $< 0.6$m, força rotações aleatórias por alguns passos para tirar o robô de cantos difíceis.

---

### 5.4 test.py (Inferência e Avaliação)

* **TD3 Modo Inferência**: Instancia apenas a rede Actor e desativa Críticos e Replay Buffer.
* **network.load()**: Carrega os pesos de `TD3_velodyne_actor.pth`.
* **Loop Principal**: Lê o estado $s_t$, calcula a ação determinística $a = \mu_\phi(s_t)$, executa a conversão $a_{\text{in}} = [(a_0+1)/2, a_1]$ e navega o robô no Gazebo sem atualização de parâmetros.


## 6. Tabela Tática de Referência de Equações e Símbolos

| Conceito / Equação | Fórmula Matemática | Símbolos Principais | Onde fica no Código |
| :--- | :--- | :--- | :--- |
| Retorno Descontado ($G_t$) | $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$ | $G_t$: Retorno total<br>$\gamma$: Desconto ($0.99999$)<br>$r$: Recompensa | `train.py` (discount = 0.99999) |
| Bellman Optimality | $Q^*(s, a) = r + \gamma \max_{a'} Q^*(s', a')$ | $Q^*$: Valor $Q$ ótimo<br>$r$: Recompensa<br>$s'$: Próximo estado | `train.py` (target_Q = reward + ...) |
| DPG (Gradiente do Ator) | $\nabla_\phi J(\phi) = \mathbb{E}[\nabla_a Q \cdot \nabla_\phi \mu]$ | $\phi$: Pesos do Ator<br>$J$: Desempenho<br>$\nabla_a Q$: Gradiente do Crítico | `train.py` (actor_loss = -Q.mean()) |
| Clipped Double Q (Pilar 1) | $y = r + \gamma(1-d) \min_{i=1,2} Q_{\text{target}, i}$ | $y$: Alvo TD<br>$d$: Terminação (done)<br>$\min$: Seleção do menor $Q$ | `train.py` (torch.min(target_Q1, target_Q2)) |
| Policy Smoothing (Pilar 2) | $a' = \text{clip}(\mu_{\text{target}}(s') + \tilde{\epsilon})$ | $\tilde{\epsilon}$: Ruído cortado<br>$\sigma$: policy_noise ($0.2$)<br>$c$: noise_clip ($0.5$) | `train.py` (noise.clamp(-0.5, 0.5)) |
| Delayed & Soft Updates (Pilar 3) | $\theta_{\text{target}} \leftarrow \tau \theta + (1-\tau)\theta_{\text{target}}$ | $\tau$: Taxa suave ($0.005$)<br>$d$: Frequência (policy_freq = 2) | `train.py` (if it % policy_freq == 0:) |
| Mapeamento de Ações | $v = \frac{a_0 + 1}{2}, \quad \omega = a_1$ | $a_0, a_1 \in [-1, 1]$<br>$v \in [0, 1]\text{ m/s}$<br>$\omega \in [-1, 1]\text{ rad/s}$ | `velodyne_env.py`, `train.py`, `test.py` (a_in) |
