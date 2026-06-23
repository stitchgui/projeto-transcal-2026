# Documento de Requisitos - Projeto de Transferência de Calor (Câmara Secadora)

## 1. Parâmetros Geométricos
* [cite_start]Geometria principal: Câmara retangular[cite: 3, 26].
* [cite_start]Comprimento ($W$): 8 m[cite: 5].
* [cite_start]Largura ($L$): 4 m[cite: 5].
* [cite_start]Altura ($H$): 3 m[cite: 5].

## 2. Condições de Contorno e Propriedades Térmicas da Câmara
* [cite_start]Temperatura da superfície base ($T_1$): 350°C, mantida por resistência elétrica[cite: 4].
* [cite_start]Temperatura do meio externo ($T_{\infty}$): 25°C[cite: 5].
* [cite_start]Coeficiente global de transferência de calor (das superfícies internas para o meio externo): $U_{sup,int\rightarrow\infty}$ = 1 W/m².°C[cite: 5].
* [cite_start]Emissividade das superfícies internas ($\epsilon$): 0,9[cite: 5].
* [cite_start]Características ópticas: Superfícies opacas e cinzas[cite: 5].
* [cite_start]Regime térmico da câmara: Permanente[cite: 5].
* [cite_start]Mecanismo de troca térmica interna: Convecção natural entre as superfícies internas e o ar interno[cite: 4].
* [cite_start]Perturbação de massa: Infiltração de ar externo com vazão de 3 m³/min[cite: 6].

## 3. Dados do Objeto em Regime Transitório (Esfera)
* [cite_start]Material: Vitrocerâmico[cite: 13].
* [cite_start]Diâmetro ($D$): 50 cm[cite: 13].
* [cite_start]Condutividade térmica ($k$): 1,4 W/m.°C[cite: 13].
* [cite_start]Massa específica ($\rho$): 2520 kg/m³[cite: 13].
* [cite_start]Calor específico ($c_p$): 790 J/kg.°C[cite: 13].
* [cite_start]Condição de contorno radiativa: Pintada de tinta preta[cite: 13].
* [cite_start]Trajetória e Cinemática: Atravessa a câmara ao longo da linha de centro no comprimento $W$ em um tempo total de 120 minutos[cite: 13].
* [cite_start]Condição inicial ($t=0$ s): Temperatura de 25°C[cite: 15].
* [cite_start]Acoplamento: Desacoplado das matrizes da câmara (a passagem da esfera não altera as temperaturas internas em regime permanente)[cite: 15].

## 4. Escopo de Entregas (Outputs do Solver Numérico)
1. [cite_start]Fatores de forma de radiação espacial ($F_{ij}$) para índices de 1 a 6, arredondados para a terceira casa decimal[cite: 7].
2. [cite_start]Coeficientes convectivos locais para as 6 faces internas ($h_c$) assumindo convecção natural, expressos em W/m².K[cite: 8].
3. [cite_start]Vetor de temperaturas das superfícies passivas ($T_2$ a $T_6$) e a temperatura em regime do ar interno ($T_{int}$), expressas em °C[cite: 9].
4. [cite_start]Temperatura de saída do ar de infiltração termalizado ($T_{ar.sai}$) em °C[cite: 10].
5. [cite_start]Vetor de radiosidades internas ($J$) para as 6 superfícies, em W/m²[cite: 11].
6. [cite_start]Demanda de potência (carga térmica) da resistência elétrica na base para manter $T_1$ a 350°C, expressa em Watts[cite: 12].
7. [cite_start]Perfil térmico espacial da esfera na saída (temperatura da superfície e do núcleo), acompanhado do plot do histórico de temperatura ao longo do tempo de residência (0 a 120 min)[cite: 14].