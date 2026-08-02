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
