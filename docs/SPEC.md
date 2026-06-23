# SPEC.md — Especificação Técnica: Câmara Secadora + Esfera Vitrocerâmica

> **Etapa:** Specify (tlc-spec-driven). **Regra restritiva desta etapa:** nenhum código é gerado. Este documento contém exclusivamente (1) a árvore de arquivos Python proposta e (2) o equacionamento analítico exato que essa árvore deverá implementar nas fases posteriores (Design/Tasks/Execute).
> **Fonte dos requisitos:** `requisitos.md` (raiz do projeto).

---

## 0. Escopo

O sistema possui dois subdomínios físicos acoplados apenas em regime (a passagem da esfera é desacoplada, conforme `requisitos.md §3`):

1. **Câmara em regime permanente** — recinto fechado de 6 superfícies cinzas-difusas-opacas, com radiação interna (método das radiosidades), convecção natural interna (ar↔superfícies), perda global para o ambiente externo (coeficiente $U$) e perturbação de massa por infiltração.
2. **Esfera vitrocerâmica em regime transitório** — atravessa a câmara desacoplada das temperaturas de regime permanente; necessita do número de Biot para decidir entre modelo de capacitância concentrada e modelo distribuído (núcleo × superfície).

---

## 1. Convenções, Notação e Constantes

| Símbolo | Significado | Valor / Unidade |
|---|---|---|
| $W, L, H$ | Comprimento, largura, altura da câmara | 8, 4, 3 m |
| $T_1$ | Temperatura da base (superfície ativa) | 350 °C = 623,15 K |
| $T_\infty$ | Temperatura do ambiente externo | 25 °C = 298,15 K |
| $U_i$ | Coef. global superfície interna → ambiente externo | 1 W/m².°C (uniforme, $i=1..6$) |
| $\varepsilon_i$ | Emissividade das superfícies internas | 0,9 (uniforme) |
| $\dot V_{inf}$ | Vazão volumétrica de infiltração | 3 m³/min = 0,05 m³/s |
| $D, k_{esf}, \rho_{esf}, c_{p,esf}$ | Diâmetro, condutividade, massa específica, calor específico da esfera | 0,5 m; 1,4 W/m.°C; 2520 kg/m³; 790 J/kg.°C |
| $\sigma$ | Constante de Stefan-Boltzmann | $5,670\times10^{-8}$ W/m².K⁴ |
| $g$ | Aceleração da gravidade | 9,81 m/s² |
| $T[K]$ | Conversão de temperatura | $T[K] = T[°C] + 273,15$ — **toda equação com $T^4$ usa Kelvin** |

Numeração das 6 superfícies (sistema de coordenadas $x\in[0,W]$, $y\in[0,L]$, $z\in[0,H]$):

| Superfície | Posição | Dimensões | Área | Natureza |
|---|---|---|---|---|
| $S_1$ | base, $z=0$ | $W\times L$ | $A_1=32\ m^2$ | **Ativa** ($T_1$ fixo) |
| $S_2$ | topo, $z=H$ | $W\times L$ | $A_2=32\ m^2$ | Passiva |
| $S_3$ | parede $x=0$ | $L\times H$ | $A_3=12\ m^2$ | Passiva |
| $S_4$ | parede $x=W$ | $L\times H$ | $A_4=12\ m^2$ | Passiva |
| $S_5$ | parede $y=0$ | $W\times H$ | $A_5=24\ m^2$ | Passiva |
| $S_6$ | parede $y=L$ | $W\times H$ | $A_6=24\ m^2$ | Passiva |

---

## 2. Arquitetura Modular Proposta (Árvore de Arquivos Python)

```text
projeto-transcal/
├── docs/
│   └── SPEC.md
├── requisitos.md
├── pyproject.toml
├── src/
│   └── transcal/
│       ├── __init__.py
│       ├── config.py                   # Constantes geométricas/materiais/BC (fonte única, §1)
│       ├── domain/
│       │   ├── __init__.py
│       │   ├── chamber.py              # Definição das 6 superfícies: áreas, adjacências, normais (§1)
│       │   └── sphere.py               # Geometria/termofísica/cinemática x(t) da esfera (§6.1 — Eq. 7.1)
│       ├── radiation/
│       │   ├── __init__.py
│       │   ├── view_factors.py         # Casos A e B + montagem da matriz F 6×6 (§3 — Eqs. 4.A/4.B)
│       │   └── radiosity.py            # Sistema [I-(I-ε)F]J = εσT⁴ e q_i (Eqs. 6.1–6.2)
│       ├── convection/
│       │   ├── __init__.py
│       │   ├── air_properties.py       # k, ν, α, Pr, β do ar em função de T_filme (§4.1 — Eq. 5.0)
│       │   └── correlations.py         # Nu piso/teto (condicional) e paredes verticais; Nu esfera (Eqs. 5.1–5.3, 7.2)
│       ├── balance/
│       │   ├── __init__.py
│       │   ├── steady_state_system.py  # Resíduo R(u) — 5 superfícies passivas + nó de ar (Eqs. 6.3–6.6)
│       │   └── newton_solver.py        # Wrapper de scipy.optimize.root (hybr/lm) + bootstrap de u0 (Eq. 6.7 — ver §7)
│       ├── transient/
│       │   ├── __init__.py
│       │   ├── biot.py                 # Bi(t), critério Gr/Re², decisão lumped×distribuído (Eqs. 7.1–7.8)
│       │   └── radial_fdm.py           # Condução transiente 1D radial (núcleo×superfície), se Bi≥0,1
│       ├── reporting/
│       │   ├── __init__.py
│       │   ├── outputs.py              # Monta as 7 entregas de requisitos.md §4 no schema canônico (§7.4)
│       │   ├── export.py               # Serializa o schema canônico em JSON/CSV/planilha (§7.4)
│       │   └── plots.py                # Histórico T(t) da esfera, 0–120 min (§7.4)
│       └── run.py                      # Orquestração: config→geometria→F_ij→convecção→balanço→esfera→relatório
└── tests/
    ├── __init__.py
    ├── test_view_factors.py            # Reciprocidade, soma=1, simetria (§10)
    ├── test_convection_correlations.py # Limites assintóticos, troca de ramo piso/teto
    ├── test_steady_state_balance.py    # Fechamento global de energia (§10)
    ├── test_biot_transient.py          # Consistência L_c=D/6, Bi com h conhecido
    └── test_solver_convergence.py      # Convergência hybr/lm a partir de u0 do §7.2, condicionamento (§10)
```

**Nenhum arquivo acima contém implementação nesta etapa** — a árvore define apenas responsabilidades e a rastreabilidade com as equações das Seções 3–6 (consolidada na matriz da Seção 8).

---

## 3. Fatores de Forma Espaciais $F_{ij}$ (Câmara 3D)

### 3.1 Relações de fechamento (válidas para qualquer enclosure convexo de superfícies planas)

$$
F_{i\to i}=0 \quad\text{(superfícies planas não se autoirradiam)}
$$
$$
A_i F_{i\to j} = A_j F_{j\to i} \quad\text{(reciprocidade)}
$$
$$
\sum_{j=1}^{6} F_{i\to j} = 1, \quad \forall i \quad\text{(soma/fechamento)}
$$

### 3.2 Caso A — Retângulos paralelos, alinhados e diretamente opostos

Para dois retângulos de lados $a, b$, paralelos, alinhados e separados por distância $c$, com $\bar X = a/c$, $\bar Y = b/c$ (Hamilton & Morgan, 1952; tabulado em Incropera et al., *Fundamentals of Heat and Mass Transfer*, Tabela 13.2; configuração C-13 do catálogo de Howell):

$$
F_{i\to j} = \frac{2}{\pi \bar X \bar Y}\left\{\ln\!\left[\sqrt{\frac{(1+\bar X^2)(1+\bar Y^2)}{1+\bar X^2+\bar Y^2}}\right] + \bar X\sqrt{1+\bar Y^2}\,\arctan\frac{\bar X}{\sqrt{1+\bar Y^2}} + \bar Y\sqrt{1+\bar X^2}\,\arctan\frac{\bar Y}{\sqrt{1+\bar X^2}} - \bar X\arctan\bar X - \bar Y\arctan\bar Y\right\}
\tag{4.A}
$$

### 3.3 Caso B — Retângulos perpendiculares com aresta comum

Retângulo horizontal de largura $W_h$ e retângulo vertical de altura $H_v$, compartilhando uma aresta comum de comprimento $L_c$. Com $\bar h = H_v/L_c$, $\bar w = W_h/L_c$ (Hamilton & Morgan, 1952; Incropera Tabela 13.2; configuração C-14 de Howell):

$$
F_{i\to j} = \frac{1}{\pi \bar w}\left[\bar h\arctan\frac{1}{\bar h} + \bar w\arctan\frac{1}{\bar w} - \sqrt{\bar h^2+\bar w^2}\,\arctan\frac{1}{\sqrt{\bar h^2+\bar w^2}}\right] + \frac{1}{4\pi \bar w}\ln\!\left[a\,b^{\bar w^2}\,c^{\bar h^2}\right]
\tag{4.B}
$$
$$
a=\frac{(1+\bar h^2)(1+\bar w^2)}{1+\bar h^2+\bar w^2}, \qquad b=\frac{\bar w^2(1+\bar h^2+\bar w^2)}{(1+\bar w^2)(\bar h^2+\bar w^2)}, \qquad c=\frac{\bar h^2(1+\bar h^2+\bar w^2)}{(1+\bar h^2)(\bar h^2+\bar w^2)}
$$

### 3.4 Mapeamento geométrico exato para a câmara $8\times4\times3$ m

A câmara possui $\binom{6}{2}=15$ pares de superfícies, redutíveis a **6 avaliações** das Eqs. (4.A)/(4.B) por simetria especular do paralelepípedo:

| Par(es) | Tipo | Parâmetros substituídos | Observação |
|---|---|---|---|
| $(1,2)$ | A | $\bar X=W/H=8/3$, $\bar Y=L/H=4/3$ | Piso–Teto |
| $(3,4)$ | A | $\bar X=L/W=4/8=0{,}5$, $\bar Y=H/W=3/8=0{,}375$ | Paredes opostas (topo/base do comprimento) |
| $(5,6)$ | A | $\bar X=W/L=8/4=2{,}0$, $\bar Y=H/L=3/4=0{,}75$ | Paredes opostas (laterais) |
| $(1,3)=(1,4)=(2,3)=(2,4)$ | B | aresta comum $=L=4$; $\bar w = W/L = 2{,}0$; $\bar h = H/L = 0{,}75$ | Piso/Teto ↔ parede $x=0$ ou $x=W$ |
| $(1,5)=(1,6)=(2,5)=(2,6)$ | B | aresta comum $=W=8$; $\bar w = L/W = 0{,}5$; $\bar h = H/W = 0{,}375$ | Piso/Teto ↔ parede $y=0$ ou $y=L$ |
| $(3,5)=(3,6)=(4,5)=(4,6)$ | B | aresta comum $=H=3$; $\bar w = L/H = 1{,}333$; $\bar h = W/H = 2{,}667$ | Paredes adjacentes (canto vertical) |

A simetria $(2,k)=(1,k)$ para $k=3..6$ decorre de o teto ser geometricamente equivalente ao piso em relação às paredes (mesma extensão, apenas refletido em $z$). A avaliação numérica final (3ª casa decimal) é tarefa de `radiation/view_factors.py` na fase de Execute — **não realizada nesta etapa**.

**Matriz simbólica $F$** (estrutura; reciprocidade aplicada onde $A_i\neq A_j$):

$$
F=\begin{bmatrix}
0 & F_{12} & F_{13} & F_{13} & F_{15} & F_{15}\\
F_{12}\!\cdot\!\frac{A_1}{A_2} & 0 & F_{13} & F_{13} & F_{15} & F_{15}\\
\frac{A_1}{A_3}F_{13} & \frac{A_1}{A_3}F_{13} & 0 & F_{34} & F_{35} & F_{35}\\
\frac{A_1}{A_3}F_{13} & \frac{A_1}{A_3}F_{13} & F_{34} & 0 & F_{35} & F_{35}\\
\frac{A_1}{A_5}F_{15} & \frac{A_1}{A_5}F_{15} & \frac{A_3}{A_5}F_{35} & \frac{A_3}{A_5}F_{35} & 0 & F_{56}\\
\frac{A_1}{A_5}F_{15} & \frac{A_1}{A_5}F_{15} & \frac{A_3}{A_5}F_{35} & \frac{A_3}{A_5}F_{35} & F_{56} & 0
\end{bmatrix}
$$

Como $A_1=A_2$, $F_{21}=F_{12}$ (caso particular notável, útil como verificação — Seção 10).

---

## 4. Convecção Natural Interna — Coeficientes $h_{c,i}$ (6 faces)

### 4.1 Número de Rayleigh e temperatura de filme

Para cada superfície $i$, com ar interno no nó único $T_{int}$:

$$
T_{f,i} = \frac{T_i + T_{int}}{2}\ (\text{K}), \qquad \beta_i = \frac{1}{T_{f,i}}\ (\text{gás ideal})
$$
$$
Ra_{L,i} = \frac{g\,\beta_i\,|T_i-T_{int}|\,L_{c,i}^3}{\nu_i\,\alpha_i}, \qquad \nu_i,\alpha_i,Pr_i,k_{ar,i}\ \text{avaliadas em}\ T_{f,i}
\tag{5.0}
$$

### 4.2 Piso $S_1$ — placa horizontal, face superior aquecida ($T_1 \gg T_{int}$, regime confirmado pelo enunciado)

Comprimento característico (McAdams, em Incropera Tab. 9.2/9.3): $L_{c,1}=L_{c,2}=A_s/P=\dfrac{WL}{2(W+L)}=\dfrac{32}{24}=1{,}333\,m$.

$$
Nu_{L,1} =
\begin{cases}
0{,}54\,Ra_{L,1}^{1/4}, & 10^4 \le Ra_{L,1} \le 10^7\\
0{,}15\,Ra_{L,1}^{1/3}, & 10^7 < Ra_{L,1} \le 10^{11}
\end{cases}
\tag{5.1}
$$
(McAdams, 1954; face superior de placa quente — fluxo instável/intensificado.)

### 4.3 Teto $S_2$ — placa horizontal, ramo condicional ao sinal de $(T_2-T_{int})$

$$
Nu_{L,2} =
\begin{cases}
0{,}54\,Ra_{L,2}^{1/4}\ \text{ou}\ 0{,}15\,Ra_{L,2}^{1/3}\ (\text{conforme faixa de}\ Ra), & \text{se } T_2 < T_{int}\ \text{(face inferior de placa fria — fluxo instável)}\\[4pt]
0{,}27\,Ra_{L,2}^{1/4}, \quad 10^5\le Ra_{L,2}\le10^{10}, & \text{se } T_2 > T_{int}\ \text{(face inferior de placa quente — fluxo estável)}
\end{cases}
\tag{5.2}
$$

> O sinal de $(T_2-T_{int})$ é desconhecido *a priori* (depende da solução do sistema não linear da Seção 5) — a seleção de ramo deve ser reavaliada a cada iteração de Newton, e é precisamente o tipo de não-suavidade que motiva a estratégia de palpite inicial da Seção 7.2.

### 4.4 Paredes laterais $S_3,S_4,S_5,S_6$ — placas verticais

Comprimento característico: $L_{c,k}=H=3\,m$ (altura da câmara). Correlação de Churchill–Chu (1975), válida para todo $Ra_L$:

$$
Nu_{L,k} = \left\{0{,}825 + \frac{0{,}387\,Ra_{L,k}^{1/6}}{\left[1+(0{,}492/Pr_k)^{9/16}\right]^{8/27}}\right\}^2, \qquad k=3,4,5,6
\tag{5.3}
$$

### 4.5 Síntese

$$
h_{c,i} = \frac{Nu_{L,i}\,k_{ar}(T_{f,i})}{L_{c,i}}, \qquad i=1,\dots,6
\tag{5.4}
$$

---

## 5. Radiosidade e Balanço Não-Linear de Energia dos Nós Termodinâmicos Passivos

### 5.1 Sistema matricial de radiosidades

Para as 6 superfícies cinzas-difusas-opacas ($\varepsilon_i=\varepsilon=0{,}9$), a radiosidade $J_i$ satisfaz $J_i=\varepsilon_i\sigma T_i^4+(1-\varepsilon_i)\sum_j F_{i\to j}J_j$. Em forma matricial, com $F=[F_{i\to j}]$ (§3.4), $\boldsymbol\varepsilon=\mathrm{diag}(\varepsilon_1,\dots,\varepsilon_6)$, $I$ identidade $6\times6$, $\mathbf J=[J_1,\dots,J_6]^\top$, $\mathbf T^4=[T_1^4,\dots,T_6^4]^\top$ (potência elemento a elemento, $T$ em Kelvin):

$$
\boxed{\left[I-(I-\boldsymbol\varepsilon)F\right]\mathbf J = \boldsymbol\varepsilon\,\sigma\,\mathbf T^4}
\tag{6.1}
$$

Sistema linear $6\times6$ em $\mathbf J$, resolvido a cada avaliação de $\mathbf T$ (não linear no laço externo, pois $T_2,\dots,T_6$ são incógnitas).

### 5.2 Fluxo radiativo líquido por superfície

$$
q_i = \frac{\varepsilon_i A_i}{1-\varepsilon_i}\left(\sigma T_i^4 - J_i\right), \qquad i=1,\dots,6 \qquad \left(\textstyle\sum_{i=1}^6 q_i = 0\right)
\tag{6.2}
$$

A identidade $\sum_i q_i=0$ decorre da conservação de energia radiativa em um invólucro fechado e é usada como verificação (Seção 10).

### 5.3 Balanço de energia das superfícies passivas $S_2,\dots,S_6$

Regime permanente, sem geração nem armazenamento. Para cada $i\in\{2,3,4,5,6\}$:

$$
\boxed{h_{c,i}\,A_i\,(T_{int}-T_i) \;=\; q_i(\mathbf T,\mathbf J) \;+\; U_i\,A_i\,(T_i-T_\infty)}
\tag{6.3}
$$

(Ganho convectivo do ar = perda radiativa líquida para o invólucro + perda para o ambiente externo via $U_i$.)

### 5.4 Balanço de energia do nó de ar interno com infiltração

$$
\boxed{\dot m_{inf}\,c_{p,ar}\,(T_{int}-T_\infty) \;=\; \sum_{k=1}^{6} h_{c,k}\,A_k\,(T_k-T_{int})}
\tag{6.4}
$$
$$
\dot m_{inf} = \rho_{ar}(T_\infty)\,\dot V_{inf}, \qquad \dot V_{inf}=0{,}05\ m^3/s
$$

### 5.5 Temperatura de saída do ar de infiltração

Sob a hipótese de mistura perfeita do nó único de ar (implícita ao se pedir um único $T_{int}$ em `requisitos.md §4.3`):

$$
\boxed{T_{ar,sai} \equiv T_{int}}
\tag{6.5}
$$

### 5.6 Potência da resistência elétrica (superfície ativa $S_1$)

$$
\boxed{Q_{resist} = h_{c,1}\,A_1\,(T_1-T_{int}) \;+\; q_1(\mathbf T,\mathbf J) \;+\; U_1\,A_1\,(T_1-T_\infty)}
\tag{6.6}
$$

(Potência que o aquecedor deve repor: convecção para o ar + radiação líquida para o invólucro + condução/convecção para o ambiente externo.)

### 5.7 Sistema não-linear consolidado $R(\mathbf u)=0$

Vetor de incógnitas $\mathbf u=[T_2,T_3,T_4,T_5,T_6,T_{int}]^\top\in\mathbb R^6$ ($T_1\equiv623{,}15\,K$ fixo):

$$
R_i(\mathbf u) = h_{c,i}(T_i,T_{int})\,A_i\,(T_{int}-T_i) - q_i(\mathbf T(\mathbf u),\mathbf J(\mathbf T(\mathbf u))) - U_i A_i (T_i-T_\infty) = 0,\quad i=2,\dots,6
$$
$$
R_{ar}(\mathbf u) = \sum_{k=1}^{6} h_{c,k}(T_k,T_{int})\,A_k\,(T_k-T_{int}) - \dot m_{inf}\,c_{p,ar}\,(T_{int}-T_\infty) = 0
$$

$$
\boxed{R(\mathbf u)=\mathbf 0 \in \mathbb R^6}
\tag{6.7}
$$

onde $\mathbf J(\mathbf T(\mathbf u))$ é obtido resolvendo (6.1) a cada avaliação. Não-linearidade: termos $T^4$ em $q_i$ e dependência de $h_{c,i}$ em $T_i,T_{int}$ via $Ra_{L,i}$ (Eq. 5.0) e via troca de ramo (Eq. 5.2). Método de solução, palpite inicial e fluxo de saída: especificados na Seção 7 (`scipy.optimize.root`, métodos `'hybr'`/`'lm'`).

---

## 6. Número de Biot da Esfera Vitrocerâmica

### 6.1 Cinemática e critério de regime convectivo

$$
V_{esf} = \frac{W}{t_{total}} = \frac{8\ m}{7200\ s} = 1{,}111\times10^{-3}\ m/s
$$
$$
\frac{Gr_D}{Re_D^2} = \frac{g\,\beta\,|T_{s,esf}(t)-T_{int}|\,D}{V_{esf}^2}
\tag{7.1}
$$

Se $Gr_D/Re_D^2 \gg 1$, a convecção forçada (induzida pelo deslocamento) é desprezível frente à natural — a confirmar numericamente na fase de execução (Seção 9, [A5]).

### 6.2 Coeficiente convectivo natural (esfera) — Churchill, 1975

Válida para $Ra_D\le10^{11}$, $Pr\ge0{,}7$:

$$
Nu_D = 2 + \frac{0{,}589\,Ra_D^{1/4}}{\left[1+(0{,}469/Pr)^{9/16}\right]^{4/9}}
\tag{7.2}
$$
$$
Ra_D = \frac{g\,\beta\,|T_{s,esf}(t)-T_{int}|\,D^3}{\nu\,\alpha}, \quad \text{propriedades em } T_f=\frac{T_{s,esf}(t)+T_{int}}{2}
$$
$$
h_{conv}(t) = \frac{Nu_D\,k_{ar}(T_f)}{D}
\tag{7.3}
$$

### 6.3 Coeficiente radiativo linearizado

Esfera pintada de preto ($\varepsilon_{esf}\approx1$), trocando radiação com uma temperatura de vizinhança efetiva $T_{surr}$, aproximada pela média de quarta potência ponderada por $\varepsilon_k A_k$ das 6 superfícies da câmara (esfera pequena frente ao invólucro):

$$
T_{surr} = \left[\frac{\sum_{k=1}^{6}\varepsilon_k A_k T_k^4}{\sum_{k=1}^{6}\varepsilon_k A_k}\right]^{1/4}
\tag{7.4}
$$
$$
h_{rad}(t) = \varepsilon_{esf}\,\sigma\,\left(T_{s,esf}(t)+T_{surr}\right)\left(T_{s,esf}(t)^2+T_{surr}^2\right)
\tag{7.5}
$$

### 6.4 Coeficiente combinado e comprimento característico

$$
h_{total}(t) = h_{conv}(t) + h_{rad}(t)
\tag{7.6}
$$
$$
L_{c,esf} = \frac{V_{esf}}{A_{s,esf}} = \frac{\tfrac{4}{3}\pi r^3}{4\pi r^2} = \frac{r}{3} = \frac{D}{6} = \frac{0{,}5}{6} = 0{,}08333\ m
\tag{7.7}
$$

> **Atenção de notação:** $D$ (diâmetro) é a escala usada em $Nu_D$/$Ra_D$ (Eqs. 7.2–7.3); $L_{c,esf}=D/6$ é a escala usada exclusivamente no número de Biot (Eq. 7.8). Não confundir as duas.

### 6.5 Número de Biot

$$
\boxed{Bi(t) = \frac{h_{total}(t)\cdot L_{c,esf}}{k_{esf}} = \frac{h_{total}(t)\cdot (D/6)}{k_{esf}}}
\tag{7.8}
$$

com $D=0{,}5\,m$, $k_{esf}=1{,}4\ W/m.°C \Rightarrow L_{c,esf}/k_{esf}=0{,}08333/1{,}4=0{,}05952\ m^2.°C/W$.

### 6.6 Critério de decisão do modelo térmico

$$
Bi(t) < 0{,}1 \;\Rightarrow\; \text{capacitância concentrada (lumped)}, \quad T_{esf}\ \text{uniforme}
$$
$$
Bi(t) \ge 0{,}1 \;\Rightarrow\; \text{gradiente interno relevante} \;\Rightarrow\; \text{solução distribuída (condução transiente radial 1D, núcleo} \times \text{superfície)}
$$

A entrega #7 de `requisitos.md §4` exige explicitamente temperatura de superfície **e** de núcleo separadas, o que é compatível com $Bi(t)\ge0{,}1$ ao longo do trajeto — a confirmar numericamente (Seção 9, [A8]).

---

## 7. Arquitetura do Solver Numérico

> Esta seção especifica **como** os sistemas das Seções 5 e 6 devem ser resolvidos — método obrigatório, estratégia de inicialização e fluxo de exportação. Permanece descritiva (nenhum código): define contratos e critérios que `balance/newton_solver.py` e `reporting/{outputs,export}.py` deverão satisfazer no Design/Execute.

### 7.1 Método de solução obrigatório — `scipy.optimize.root`

A resolução do sistema não linear $R(\mathbf u)=\mathbf 0$ (Eq. 6.7, $\mathbf u\in\mathbb R^6$) **deve** usar `scipy.optimize.root`, não `scipy.optimize.fsolve` nem um laço de Newton-Raphson manual. Justificativa: `root` expõe diretamente os dois métodos baseados em MINPACK exigidos e devolve diagnósticos de convergência (`OptimizeResult.success`, `.fjac`, `.r`, `.nfev`) necessários para os critérios de verificação (§10).

| Método (`method=`) | Algoritmo MINPACK | Papel nesta especificação |
|---|---|---|
| `'hybr'` (padrão) | HYBRD — Powell, dogleg/região de confiança | **Método primário.** Adequado ao sistema exatamente determinado (6 equações, 6 incógnitas) e suave por trechos. |
| `'lm'` | LMDIF/LMDER — Levenberg-Marquardt | **Método de contingência (§7.3).** O parâmetro de amortecimento $\lambda$ interpola entre Gauss-Newton e máxima descida, tornando-o mais tolerante a Jacobianos mal-condicionados do que `'hybr'` — propriedade buscada explicitamente para mitigar a instabilidade descrita em §7.2. |

Regras adicionais de implementação obrigatórias para a função de resíduo `fun=R`:

1. **Resolução aninhada da radiosidade.** A cada chamada de `R(\mathbf u)$, o subsistema linear (Eq. 5.1) deve ser resolvido por solver **direto** (ex. `numpy.linalg.solve`), nunca iterativo — a matriz $[I-(I-\boldsymbol\varepsilon)F]$ é estritamente diagonal-dominante (cada linha tem $1$ na diagonal contra $\sum_j(1-\varepsilon)F_{ij}\le(1-\varepsilon)\le0{,}1$ fora da diagonal, por Gershgorin), portanto sempre não singular e bem condicionada — o solve aninhado não é fonte de instabilidade.
2. **Guarda de domínio físico.** Antes de avaliar $Ra$, $Nu$ e propriedades do ar, a função de resíduo deve impor $T_i > 0\,K$ (ex. `max(T_i, 1.0)`) — o passo de diferenças finitas interno de `hybr`/`lm` pode propor iterados exploratórios não físicos, e `sqrt`/`log`/expoentes fracionários nas Eqs. 4.A–7.8 não são definidos para argumentos negativos.
3. **Adimensionalização do resíduo.** $R_i$ tem unidade de potência (W) e as 6 componentes (5 superfícies + nó de ar) diferem em ordens de grandeza (áreas de 12–32 m², $h_c$ de poucos W/m².K, $\Delta T$ de dezenas a centenas de °C). Antes de expor o vetor a `root`, cada componente deve ser normalizada por uma escala de referência da ordem da maior perda possível daquele nó: $Q_{ref,i}=U_iA_i(T_1-T_\infty)$ para $i=2,\dots,6$ (superfícies passivas) e $Q_{ref,ar}=\dot m_{inf}c_{p,ar}(T_1-T_\infty)$ para o nó de ar — de modo que `xtol`/`tol` tenham significado comparável nas 6 componentes.
4. **Jacobiano:** não fornecido analiticamente nesta etapa — `hybr`/`lm` o aproximam por diferenças finitas internamente. É exatamente por essa dependência de diferenças finitas que a qualidade do palpite inicial (§7.2) e a adimensionalização (regra 3) são requisitos de primeira ordem, não otimizações opcionais.

### 7.2 Estratégia de determinação dos palpites iniciais (mitigação de instabilidade jacobiana)

**Problema a evitar:** um palpite inicial com todos os $T_i^{(0)}$ iguais entre si e iguais a $T_{int}^{(0)}$ coloca a avaliação inicial exatamente sobre a fronteira de troca de ramo da Eq. 4.3 (teto, $T_2=T_{int}\Rightarrow Ra=0$, função não suave) **e** anula $|T_i-T_{int}|$ em todas as superfícies simultaneamente. Diferenças finitas calculadas por `hybr`/`lm` nesse ponto produzem colunas do Jacobiano numericamente degeneradas (derivadas de $h_{c,i}$ mal definidas ou nulas), o sintoma clássico de "instabilidade jacobiana" nesta classe de problema.

**Construção do palpite ($\mathbf u^{(0)}$), em 3 passos, fundamentada numa rede de resistências térmicas linearizada (usada *apenas* para inicializar — não substitui a solução não linear das Seções 5–6):**

**Passo 1 — Condutâncias de bootstrap.** Linearizar a radiação em torno de $T_{ref}=(T_1+T_\infty)/2$ e arbitrar um $\Delta T$ grosseiro para uma avaliação única (não iterada) das correlações de convecção:
$$
h_{rad}^{(0)} \approx 4\varepsilon\sigma T_{ref}^3, \qquad h_{c,1}^{(0)} = h_c\big(Ra_{L,1}(\Delta T=(T_1-T_\infty)/2)\big)
\tag{8.1}
$$
$$
G_{A,P} = \left(h_{c,1}^{(0)}+h_{rad}^{(0)}\right)A_1, \qquad G_{A,\infty}=U_1A_1, \qquad G_{P,\infty}=\sum_{i=2}^{6}U_iA_i + \dot m_{inf}c_{p,ar}
\tag{8.2}
$$

**Passo 2 — Estimativa escalar do nó interior agrupado** (rede de 2 nós: piso $T_1$ fixo $\to$ "cluster" interior $T_p^{(0)}$ $\to$ ambiente $T_\infty$ fixo, divisor de tensão térmico clássico):
$$
\boxed{T_p^{(0)} = T_\infty + (T_1-T_\infty)\,\frac{G_{A,P}}{G_{A,P}+G_{P,\infty}}}
\tag{8.3}
$$
Como $U_i=1\ W/m^2.°C \ll h_{c,1}^{(0)}+h_{rad}^{(0)}$ (radiação linearizada a $T_{ref}\sim450\,°C$ é de ordem $10^1\ W/m^2.K$), espera-se $G_{A,P}\gg G_{P,\infty}$, logo $T_p^{(0)}$ próximo de $T_1$ — fisicamente consistente com um envoltório externo muito mais resistivo (isolante) do que o acoplamento interno radiação+convecção.

**Passo 3 — Quebra de simetria por exposição radiativa** (evita o palpite degenerado descrito acima):
$$
\boxed{T_i^{(0)} = T_\infty + \left(T_p^{(0)}-T_\infty\right)\left[1+\delta\left(F_{1\to i}-\overline{F_{1\to\bullet}}\right)\right], \quad i=2,\dots,6, \qquad T_{int}^{(0)} = T_p^{(0)}}
\tag{8.4}
$$
onde $\overline{F_{1\to\bullet}}=\tfrac{1}{5}\sum_{i=2}^{6}F_{1\to i}$ e $\delta\approx0{,}1$ é um fator de dispersão pequeno — usa os $F_{1\to i}$ já calculados na Seção 3 (disponíveis antes do solver não linear, sem custo adicional) para diferenciar superfícies mais "vistas" pelo piso (devem partir mais aquecidas) das menos vistas, sem pretender exatidão física.

Esta construção garante: **(i)** $T_i^{(0)}\neq T_{int}^{(0)}$ para todo $i$ (longe da fronteira não suave da Eq. 4.3); **(ii)** os seis $T_i^{(0)}$ são distintos entre si (colunas do Jacobiano numérico linearmente independentes desde a primeira iteração); **(iii)** ordenação física plausível $T_\infty < T_i^{(0)} < T_1$, reduzindo o número de iterações e o risco de propriedades do ar avaliadas fora da faixa tabulada.

### 7.3 Política de escalonamento (fallback)

1. **Tentativa 1:** `method='hybr'`, $\mathbf u^{(0)}$ da Eq. 8.4.
2. **Tentativa 2** (se `OptimizeResult.success=False`): `method='lm'`, mesmo $\mathbf u^{(0)}$ — o amortecimento de Levenberg-Marquardt tende a recuperar convergência onde o passo dogleg de `hybr` falhou.
3. **Tentativa 3** (se ainda falhar): **continuação/homotopia em $\varepsilon$** — resolver uma sequência $\varepsilon_k=0{,}05,0{,}10,\dots,0{,}9$ (passo $\sim$0,05), usando a solução convergida em $\varepsilon_k$ como palpite inicial de $\varepsilon_{k+1}$. Isso introduz a não-linearidade radiativa (o termo $T^4$ mais agressivo) gradualmente em vez de exigi-la de uma só vez a partir de (8.4).
4. **Critério final de aceitação:** $\|R(\mathbf u^\*)\|_\infty$ (resíduo adimensionalizado, regra 3 de §7.1) abaixo da tolerância **e** condicionamento do Jacobiano numérico em $\mathbf u^\*$ dentro de faixa aceitável (ligado ao novo critério de verificação, §10, item 7).

### 7.4 Fluxo de I/O — exportação estruturada dos resultados (itens 1–7)

**Esquema canônico** (montado por `reporting/outputs.py` após convergência de §7.1–7.3 e da solução transiente da esfera):

| Campo | Item (`requisitos.md §4`) | Forma | Unidade | Arredondamento exigido | Equação fonte |
|---|---|---|---|---|---|
| `F_ij` | 1 | matriz $6\times6$ | adimensional | 3 decimais | 4.A, 4.B |
| `h_c` | 2 | vetor (6) | W/m².K | — | 5.1–5.3 |
| `T_surfaces`, `T_int` | 3 | vetor (5) + escalar | °C | — | 6.3, 6.4, 6.7 |
| `T_ar_sai` | 4 | escalar | °C | — | 6.5 |
| `J` | 5 | vetor (6) | W/m² | — | 6.1 |
| `Q_resist` | 6 | escalar | W | — | 6.6 |
| `sphere_profile` | 7 (perfil) | série temporal, 0–120 min, colunas `t,T_superficie,T_nucleo` | °C | — | 7.1–7.8 + FDM radial |
| `sphere_plot` | 7 (gráfico) | figura | — | — | `reporting/plots.py` |

**Pipeline de exportação** (`reporting/export.py`, consumindo o esquema acima):

1. `outputs.json` — registro completo de auditoria, precisão total (inclui também $Ra$, $Nu$, propriedades do ar, $\|R(\mathbf u^\*)\|_\infty$, número de condição do Jacobiano — rastreabilidade do §7.3 item 4).
2. `outputs_planilha.csv` — uma linha/bloco por item 1–7, **já arredondado** conforme a coluna acima, na ordem e nomenclatura de `requisitos.md §4`, formatado para colagem direta na planilha de resposta final.
3. `sphere_profile.csv` — série temporal completa do núcleo/superfície da esfera (item 7).
4. `figures/sphere_temperature_history.png` — gráfico exigido pelo item 7.

> **Dependência aberta [A9 — ver §9]:** o layout exato da planilha de resposta final (nomes de aba, endereços de célula) **não está presente no diretório do projeto** (confirmado: apenas `requisitos.md`, `venv/` e `docs/`). O esquema acima usa nomes de campo neutros; uma vez fornecido o arquivo-modelo, basta acrescentar um módulo fino de mapeamento campo→célula (ex. via `openpyxl`) sem alterar nenhum módulo de física — separação de responsabilidades deliberada entre cálculo (Seções 3–6) e apresentação (esta seção).

---

## 8. Matriz de Rastreabilidade (Entregas × Equações × Módulo)

| # | Entrega (`requisitos.md §4`) | Equação(ões) governantes | Módulo responsável |
|---|---|---|---|
| 1 | $F_{ij}$, $i,j=1..6$, 3 decimais | (4.A), (4.B) + tabela §3.4 + fechamento §3.1 | `radiation/view_factors.py` |
| 2 | $h_c$ locais das 6 faces (W/m².K) | (5.0)–(5.4) | `convection/correlations.py`, `convection/air_properties.py` |
| 3 | Vetor $T_2,\dots,T_6$ e $T_{int}$ (°C) | (6.3), (6.4), (6.7) + §7.1–7.3 (solver) | `balance/steady_state_system.py`, `balance/newton_solver.py` |
| 4 | $T_{ar,sai}$ (°C) | (6.5) | `reporting/outputs.py`, `reporting/export.py` |
| 5 | Vetor de radiosidades $\mathbf J$ (W/m²) | (6.1) | `radiation/radiosity.py` |
| 6 | Potência da resistência (W) | (6.6) | `balance/steady_state_system.py`, `reporting/export.py` |
| 7 | Perfil térmico da esfera (superfície/núcleo) + gráfico 0–120 min | (7.1)–(7.8) + FDM radial (fase futura) | `transient/biot.py`, `transient/radial_fdm.py`, `reporting/plots.py`, `reporting/export.py` |

Esquema de campos e formatos de exportação dos 7 itens: §7.4.

---

## 9. Premissas de Modelagem (a validar/discutir antes do Design)

| ID | Premissa | Justificativa |
|---|---|---|
| A1 | $U_i$ aplica-se às 6 superfícies, incluindo a base ($S_1$) | `requisitos.md §2` descreve $U$ para "superfícies internas" no plural, sem indicar isolamento traseiro do aquecedor |
| A2 | $\dot m_{inf}=\rho_{ar}(T_\infty)\dot V_{inf}$; $c_{p,ar}$ avaliado em condição de referência externa | Ar de infiltração entra nas condições ambiente; convenção usual em balanços de infiltração |
| A3 | $T_{ar,sai}\equiv T_{int}$ (nó de ar único, mistura perfeita) | `requisitos.md §4.3` pede um único $T_{int}$, implicando ar interno isotérmico |
| A4 | Jacobiano de (6.7) não especificado nesta etapa (analítico vs. diferenças finitas) | Decisão de implementação, não de modelagem física — pertence à fase de Design |
| A5 | Convecção forçada por deslocamento da esfera é desprezada frente à natural | $V_{esf}\approx1{,}1\,mm/s$; a confirmar via (7.1) na execução |
| A6 | $\varepsilon_{esf}\approx1$ (tinta preta) | `requisitos.md §3` |
| A7 | $T_{surr}$ da esfera = média de 4ª potência ponderada por $\varepsilon_k A_k$ das 6 superfícies, independente da posição $x(t)$ | Simplificação para "objeto pequeno em invólucro grande"; ignora variação angular dos fatores de forma esfera→superfície ao longo da trajetória |
| A8 | Espera-se $Bi(t)\ge0{,}1$ ao longo de todo o percurso | Compatível com a exigência de perfil núcleo×superfície separado na entrega #7 |
| A9 | Layout exato da planilha de resposta final (abas/células) não fornecido no diretório do projeto | Busca em `projeto-transcal/` retornou apenas `requisitos.md`, `venv/` e `docs/`; exportação especificada por campos nominais (§7.4), pendente de mapeamento campo→célula quando o modelo for disponibilizado |
| A10 | O palpite inicial do §7.2 (Eqs. 8.1–8.4) é uma heurística de rede linearizada para inicialização, não uma solução aproximada do sistema | Deve ser validada empiricamente na execução (sensibilidade do número de iterações de `hybr`/`lm` a $\delta$ e a $\Delta T$ de referência em (8.1)) |

---

## 10. Critérios de Verificação e Validação Analítica

1. **Reciprocidade e soma:** para cada $i$, $\sum_j F_{i\to j}=1$ (tolerância sugerida $10^{-3}$, compatível com o arredondamento pedido). Para o piso: $F_{12}+2F_{13}+2F_{15}=1$.
2. **Simetria geométrica:** $F_{21}=F_{12}$ (pois $A_1=A_2$); $F_{2k}=F_{1k}$ para $k=3..6$.
3. **Conservação radiativa interna:** $\sum_{i=1}^{6} q_i = 0$ (Eq. 6.2) — verificação independente do solver de radiosidade.
4. **Fechamento energético global do sistema** (derivado somando as Eqs. 6.3 para $i=2..6$, a Eq. 6.6 e a Eq. 6.4, usando $\sum_i q_i=0$):
$$
\boxed{Q_{resist} = \sum_{i=1}^{6} U_i A_i (T_i-T_\infty) + \dot m_{inf}\,c_{p,ar}\,(T_{int}-T_\infty)}
$$
Esta identidade **não é uma equação adicional do sistema** — é uma consequência algébrica de (6.1)–(6.6) e deve ser usada como teste de consistência numérica pós-solução (toda a potência elétrica injetada é, no agregado, dissipada por perdas de envoltório + entalpia do ar de infiltração).
5. **Limites assintóticos das correlações de convecção:** $Nu\to2$ quando $Ra\to0$ (esfera, Eq. 7.2); continuidade de (5.1) em $Ra_{L,1}=10^7$.
6. **Consistência dimensional do Biot:** $L_{c,esf}=V/A_s=D/6$ deve ser conferido contra a definição geral $Bi=hL_c/k$ (Incropera) e não com $L_c=D$ (erro comum).
7. **Diagnóstico de condicionamento do solver (§7):** inspecionar `OptimizeResult.fjac`/`.r` retornados por `scipy.optimize.root` no ponto convergido $\mathbf u^\*$ (ou recompor o Jacobiano numérico) e calcular seu número de condição; valores muito elevados (heurística inicial: $>10^3$, a calibrar empiricamente) indicam que a adimensionalização do resíduo (§7.1, regra 3) ou o palpite inicial (§7.2) precisam de revisão antes de aceitar a solução.

---

## Referências (consulta/verificação cruzada desta etapa)

- Hamilton, D.C.; Morgan, D.R. (1952). *Radiant-Interchange Configuration Factors*. NACA TN-2836.
- Howell, J.R. *A Catalog of Radiation Heat Transfer Configuration Factors*, configurações C-13 e C-14 — [thermalradiation.net](https://www.thermalradiation.net/indexCat.html)
- Incropera, F.P. et al. *Fundamentals of Heat and Mass Transfer*, Tabela 13.2 (fatores de forma 3D), Cap. 9 (convecção natural).
- Martinez, I. (ETSIAE-UPM). *Radiative View Factors* (notas de curso, usadas para verificação cruzada das Eqs. 4.A/4.B) — http://imartinez.etsiae.upm.es/~isidoro/tc3/Radiation%20View%20factors.pdf
- Churchill, S.W.; Chu, H.H.S. (1975). *Correlating equations for laminar and turbulent free convection from a vertical plate / from a sphere*. Int. J. Heat Mass Transfer, 18.
- McAdams, W.H. (1954). *Heat Transmission*, 3rd ed. — correlações de placas horizontais.
- Virtanen, P. et al. (2020). *SciPy 1.0: Fundamental Algorithms for Scientific Computing in Python*. Nature Methods, 17 — base de `scipy.optimize.root` (§7.1).
- Moré, J.J.; Garbow, B.S.; Hillstrom, K.E. (1980). *User Guide for MINPACK-1*. Argonne National Laboratory, ANL-80-74 — algoritmos HYBRD (`'hybr'`) e LMDIF/LMDER (`'lm'`) subjacentes ao §7.1.
