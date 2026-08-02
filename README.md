# DRL-TD3
Aplicando o aprendizado por reforço profundo para o controle contínuo avançado com duas redes em navegação robótica.
# Explicação Completa do Projeto TD3 para Navegação com LiDAR

Este documento explica detalhadamente os quatro códigos fornecidos, partindo da teoria dos Processos de Decisão de Markov (MDP), passando pelo algoritmo TD3 e culminando na implementação linha a linha. O foco especial está no processamento do LiDAR (como 20 distâncias são obtidas) e nas equações que governam cada etapa.

---

## 1. O Problema como um Processo de Decisão de Markov (MDP)

O robô precisa navegar de um ponto inicial aleatório a um alvo, desviando de obstáculos. Modelamos isso como um MDP definido pela tupla \(\langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma \rangle\):

- **Estado \(s \in \mathcal{S}\)** : vetor de 24 dimensões:
  \[
  s = [\underbrace{d_1, d_2, \dots, d_{20}}_{\text{LiDAR}}, \underbrace{\rho, \theta, v_{t-1}, \omega_{t-1}}_{\text{informações do robô}}]
  \]
  - \(d_i\): distância mínima ao obstáculo na i-ésima fatia angular do LiDAR (20 fatias, cobrindo ~180° frontais).
  - \(\rho\): distância euclidiana do robô ao alvo.
  - \(\theta\): ângulo relativo entre a orientação do robô e a direção do alvo.
  - \(v_{t-1}, \omega_{t-1}\): velocidades linear e angular aplicadas no passo anterior (ações passadas).

- **Ação \(a \in \mathcal{A}\)** : par contínuo \((v, \omega)\), com \(v \in [-1,1]\) (mapeado para \([0,1]\) antes de enviar ao robô) e \(\omega \in [-1,1]\).

- **Transição \(\mathcal{P}(s'|s,a)\)** : determinada pela física do simulador Gazebo. Dado um estado e ação, o ambiente retorna o próximo estado.

- **Recompensa \(r = \mathcal{R}(s,a)\)** :
  \[
  r(s,a) = \begin{cases}
  +100, & \text{se } \rho < 0.3 \text{ (alvo alcançado)}\\
  -100, & \text{se } \min_i d_i < 0.35 \text{ (colisão)}\\
  \dfrac{v}{2} - \dfrac{|\omega|}{2} - \dfrac{1 - \min_i d_i}{2}, & \text{caso contrário (apenas se } \min_i d_i < 1)
  \end{cases}
  \]

- **Fator de desconto \(\gamma = 0.99999\)** : quase 1, para valorizar recompensas futuras mesmo em longos horizontes.

**Objetivo**: encontrar uma política determinística \(\mu: \mathcal{S} \to \mathcal{A}\) que maximize o retorno acumulado esperado:
\[
J(\mu) = \mathbb{E}\left[ \sum_{t=0}^{\infty} \gamma^t r_t \right]
\]

---

## 2. Do DDPG ao TD3 – Melhorias com Equações

### 2.1 DDPG (ponto de partida)

O DDPG é um algoritmo actor‑critic para ações contínuas:

- **Ator**: \(\mu(s|\theta^\mu)\) → ação.
- **Crítico**: \(Q(s,a|\theta^Q)\) → valor da ação.
- Atualização do crítico minimizando o erro quadrático com o alvo de Bellman:
  \[
  y = r + \gamma Q'(s', \mu'(s'|\theta^{\mu'}) | \theta^{Q'})
  \]
  \[
  L(\theta^Q) = \mathbb{E}\left[ (Q(s,a|\theta^Q) - y)^2 \right]
  \]
- Atualização do ator pelo gradiente determinístico da política:
  \[
  \nabla_{\theta^\mu} J \approx \mathbb{E}\left[ \nabla_a Q(s,a|\theta^Q)\big|_{a=\mu(s)} \nabla_{\theta^\mu} \mu(s|\theta^\mu) \right]
  \]
- Redes alvo atualizadas suavemente (soft update) com fator \(\tau \ll 1\):
  \[
  \theta' \leftarrow \tau \theta + (1-\tau) \theta'
  \]

### 2.2 TD3 (Twin Delayed DDPG)

O TD3 introduz três melhorias para estabilizar o aprendizado e evitar superestimação:

1. **Clipped Double Q‑learning**  
   Mantém dois críticos \(Q_1, Q_2\) e usa o valor mínimo para o alvo:
   \[
   y = r + \gamma \min_{i=1,2} Q_i'(s', \tilde{a})
   \]

2. **Target Policy Smoothing**  
   Adiciona ruído gaussiano truncado à ação alvo para suavizar a estimativa Q:
   \[
   \tilde{a} = \mu'(s'|\theta^{\mu'}) + \epsilon, \quad \epsilon \sim \text{clip}(\mathcal{N}(0,\sigma), -c, c)
   \]
   No código: \(\sigma = 0.2\), \(c = 0.5\).

3. **Delayed Policy Updates**  
   O ator (e as redes alvo) são atualizados com menor frequência que os críticos. A cada `policy_freq` iterações do crítico (aqui 2), o ator é atualizado uma vez. Isso permite que o crítico esteja mais estável antes de guiar a política.

---

## 3. Explicação Detalhada dos Códigos

### 3.1 `velodyne_env.py` – O Ambiente (modelo do MDP)

Este arquivo implementa a interface com o Gazebo/ROS, gerando as transições do MDP.

#### 3.1.1 Inicialização e conexão ROS

```python
rospy.init_node("gym", anonymous=True)
subprocess.Popen(["roslaunch", "-p", port, fullpath])
# ...
self.vel_pub = rospy.Publisher("/r1/cmd_vel", Twist, queue_size=1)
self.velodyne = rospy.Subscriber("/velodyne_points", PointCloud2, self.velodyne_callback, queue_size=1)
self.odom = rospy.Subscriber("/r1/odom", Odometry, self.odom_callback, queue_size=1)
