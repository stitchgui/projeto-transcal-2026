"""Fatores de forma de radiação F_i->j para um recinto retangular 3D de 6 superfícies.

Implementa as relações analíticas exatas (Hamilton & Morgan, 1952) para retângulos
paralelos opostos (Caso A, Eq. 4.A) e retângulos perpendiculares com aresta comum
(Caso B, Eq. 4.B), conforme docs/SPEC.md Seção 3. Apenas 6 avaliações são feitas
por simetria especular do paralelepípedo (§3.4); o restante da matriz 6x6 é
obtido por álgebra de fatores de forma (equivalência piso<->teto) e pela regra
da reciprocidade A_i F_i->j = A_j F_j->i.
"""

import math

Matrix6x6 = list[list[float]]

SURFACE_LABELS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6")


def f_parallel_rectangles(a: float, b: float, c: float) -> float:
    """Eq. 4.A — F_i->j entre retângulos a x b paralelos, alinhados e diretamente
    opostos, separados pela distância c."""
    x_bar = a / c
    y_bar = b / c
    log_term = math.log(
        math.sqrt((1 + x_bar**2) * (1 + y_bar**2) / (1 + x_bar**2 + y_bar**2))
    )
    bracket = (
        log_term
        + x_bar * math.sqrt(1 + y_bar**2) * math.atan(x_bar / math.sqrt(1 + y_bar**2))
        + y_bar * math.sqrt(1 + x_bar**2) * math.atan(y_bar / math.sqrt(1 + x_bar**2))
        - x_bar * math.atan(x_bar)
        - y_bar * math.atan(y_bar)
    )
    return (2.0 / (math.pi * x_bar * y_bar)) * bracket


def f_perpendicular_rectangles(w_h: float, h_v: float, l_c: float) -> float:
    """Eq. 4.B — F_i->j entre retângulo horizontal de largura w_h e retângulo
    vertical de altura h_v, compartilhando uma aresta comum de comprimento l_c."""
    w_bar = w_h / l_c
    h_bar = h_v / l_c
    diag = math.sqrt(h_bar**2 + w_bar**2)
    term1 = (
        h_bar * math.atan(1 / h_bar)
        + w_bar * math.atan(1 / w_bar)
        - diag * math.atan(1 / diag)
    )
    a = (1 + h_bar**2) * (1 + w_bar**2) / (1 + h_bar**2 + w_bar**2)
    b = w_bar**2 * (1 + h_bar**2 + w_bar**2) / ((1 + w_bar**2) * (h_bar**2 + w_bar**2))
    c = h_bar**2 * (1 + h_bar**2 + w_bar**2) / ((1 + h_bar**2) * (h_bar**2 + w_bar**2))
    log_term = math.log(a * b ** (w_bar**2) * c ** (h_bar**2))
    return term1 / (math.pi * w_bar) + log_term / (4 * math.pi * w_bar)


def chamber_surface_areas(
    W: float, L: float, H: float
) -> tuple[float, float, float, float, float, float]:
    """Áreas A1..A6 das 6 superfícies do recinto W x L x H (SPEC §1)."""
    a_floor_ceiling = W * L
    a_x_walls = L * H
    a_y_walls = W * H
    return (
        a_floor_ceiling,  # A1 - base (z=0)
        a_floor_ceiling,  # A2 - topo (z=H)
        a_x_walls,        # A3 - parede x=0
        a_x_walls,        # A4 - parede x=W
        a_y_walls,        # A5 - parede y=0
        a_y_walls,        # A6 - parede y=L
    )


def reciprocal_view_factor(f_pq: float, a_p: float, a_q: float) -> float:
    """Regra da reciprocidade: A_p F_p->q = A_q F_q->p  =>  retorna F_q->p."""
    return a_p * f_pq / a_q


def build_view_factor_matrix(W: float, L: float, H: float) -> Matrix6x6:
    """Monta a matriz 6x6 completa F_i->j do recinto W x L x H (SPEC §3.4).

    Calcula apenas as 6 relações geometricamente independentes (Eqs. 4.A/4.B);
    o restante é derivado por simetria especular piso<->teto e pela regra da
    reciprocidade.
    """
    areas = chamber_surface_areas(W, L, H)

    f12 = f_parallel_rectangles(W, L, H)         # piso <-> teto
    f34 = f_parallel_rectangles(L, H, W)         # parede x=0 <-> parede x=W
    f56 = f_parallel_rectangles(W, H, L)         # parede y=0 <-> parede y=L
    f13 = f_perpendicular_rectangles(W, H, L)    # piso/teto <-> parede x=0 ou x=W
    f15 = f_perpendicular_rectangles(L, H, W)    # piso/teto <-> parede y=0 ou y=L
    f35 = f_perpendicular_rectangles(L, W, H)    # parede x=.. <-> parede y=.. (canto)

    F: Matrix6x6 = [[0.0] * 6 for _ in range(6)]

    # Triângulo superior (i<j). O teto (índice 1) é geometricamente equivalente
    # ao piso (índice 0) frente às paredes (reflexão especular em z) — herda F13/F15.
    F[0][1] = f12
    F[0][2] = F[0][3] = f13
    F[0][4] = F[0][5] = f15
    F[1][2] = F[1][3] = f13
    F[1][4] = F[1][5] = f15
    F[2][3] = f34
    F[2][4] = F[2][5] = f35
    F[3][4] = F[3][5] = f35
    F[4][5] = f56

    # Triângulo inferior (i>j) via regra da reciprocidade.
    for i in range(6):
        for j in range(i):
            F[i][j] = reciprocal_view_factor(F[j][i], areas[j], areas[i])

    return F


def validate_view_factor_matrix(
    F: Matrix6x6, areas: tuple[float, ...], tol: float = 1e-3
) -> list[str]:
    """Critérios de verificação da SPEC §10: fechamento (soma=1 por linha),
    reciprocidade A_i F_i->j = A_j F_j->i e simetria F21=F12 (pois A1=A2)."""
    messages: list[str] = []

    for i in range(6):
        row_sum = sum(F[i])
        status = "OK" if abs(row_sum - 1.0) < tol else "FALHA"
        messages.append(f"  soma(F[{SURFACE_LABELS[i]},:]) = {row_sum:.6f}  [{status}]")

    recip_ok = all(
        abs(areas[i] * F[i][j] - areas[j] * F[j][i]) < 1e-9
        for i in range(6)
        for j in range(6)
    )
    messages.append(f"  reciprocidade A_i F_ij = A_j F_ji: {'OK' if recip_ok else 'FALHA'}")

    sym_ok = abs(F[1][0] - F[0][1]) < 1e-12
    messages.append(f"  simetria F21 = F12 (A1=A2): {'OK' if sym_ok else 'FALHA'}")

    return messages


if __name__ == "__main__":
    # Dimensões da câmara secadora (docs/SPEC.md §1).
    W, L, H = 8.0, 4.0, 3.0

    F = build_view_factor_matrix(W, L, H)
    areas = chamber_surface_areas(W, L, H)

    print(f"Fatores de forma F_i->j -- recinto {W:.0f}x{L:.0f}x{H:.0f} m\n")
    header = "      " + "".join(f"{label:>10}" for label in SURFACE_LABELS)
    print(header)
    for i, row in enumerate(F):
        formatted_row = "".join(f"{value:10.3f}" for value in row)
        print(f"{SURFACE_LABELS[i]:<6}{formatted_row}")

    print("\nValidação (SPEC §10):")
    for line in validate_view_factor_matrix(F, areas):
        print(line)
