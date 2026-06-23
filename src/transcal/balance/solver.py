"""Balanço de energia em regime permanente da câmara secadora (docs/SPEC.md Seções 4-7).

Resolve o sistema não linear R(u)=0 (Eq. 6.7) acoplando radiação (radiosidade,
Eq. 6.1-6.2), convecção natural interna (Eqs. 5.0-5.4) e infiltração de ar
(Eq. 6.4), extraindo T2..T6, T_int, T_ar_sai, J e a potência da resistência (Q_resist).

SPEC_DEVIATION (1): docs/SPEC.md §7.1/§7.3 define 'hybr' como método primário e
'lm' como contingência, dentro de uma política de 3 tentativas (hybr -> lm ->
homotopia em epsilon). Esta implementação chama scipy.optimize.root diretamente
com method='lm', por instrução explícita desta task; a política de
escalonamento completa de §7.3 não foi implementada.

SPEC_DEVIATION (2): o palpite inicial usa uma heurística simplificada (peso fixo
T_p0 = T_inf + 0.85*(T1-T_inf) + quebra de simetria via F_(1->i)), em vez da rede
de resistências linearizada completa das Eqs. 8.1-8.3 de §7.2. A quebra de
simetria via F_(1->i) (Eq. 8.4) foi mantida.

SPEC_DEVIATION (3): por instrução desta task, convecção, radiosidade e o solver
não-linear foram colapsados neste único arquivo, em vez dos módulos separados
convection/{air_properties,correlations}.py, radiation/radiosity.py e
balance/{steady_state_system,newton_solver}.py previstos na árvore do SPEC §2.

Fonte das propriedades do ar (k, mu, nu, alpha, Pr, rho, cp a 1 atm): Cengel &
Ghajar, "Heat and Mass Transfer", Table A-9 — tabela equivalente à Incropera
Tabela A.4 referenciada genericamente em SPEC §4.1 (nenhuma fórmula fechada é
dada na SPEC para essas propriedades; valores tabelados, interpolados
linearmente, evitam fabricar uma correlação não documentada).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import optimize

from transcal.radiation.view_factors import build_view_factor_matrix, chamber_surface_areas

logger = logging.getLogger("transcal.balance.solver")

# --------------------------------------------------------------------------- #
# Propriedades do ar a 1 atm (Cengel & Ghajar, Table A-9 / ~ Incropera Tab. A.4)
# --------------------------------------------------------------------------- #

_AIR_T_C = np.array(
    [-50, -40, -30, -20, -10, 0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50,
     60, 70, 80, 90, 100, 120, 140, 160, 180, 200, 250, 300, 350, 400],
    dtype=float,
)
_AIR_RHO = np.array(
    [1.582, 1.514, 1.451, 1.394, 1.341, 1.292, 1.269, 1.246, 1.225, 1.204,
     1.184, 1.164, 1.145, 1.127, 1.109, 1.092, 1.059, 1.028, 0.9994, 0.9718,
     0.9458, 0.8977, 0.8542, 0.8148, 0.7788, 0.7459, 0.6746, 0.6158, 0.5664, 0.5243],
)
_AIR_CP = np.array(
    [999, 1002, 1004, 1005, 1006, 1006, 1006, 1006, 1007, 1007,
     1007, 1007, 1007, 1007, 1007, 1007, 1007, 1007, 1008, 1008,
     1009, 1011, 1013, 1016, 1019, 1023, 1033, 1044, 1056, 1069],
)
_AIR_K = np.array(
    [0.01979, 0.02057, 0.02134, 0.02211, 0.02288, 0.02364, 0.02401, 0.02439, 0.02476, 0.02514,
     0.02551, 0.02588, 0.02625, 0.02662, 0.02699, 0.02735, 0.02808, 0.02881, 0.02953, 0.03024,
     0.03095, 0.03235, 0.03374, 0.03511, 0.03646, 0.03779, 0.04104, 0.04418, 0.04721, 0.05015],
)
_AIR_ALPHA = np.array(
    [1.252e-5, 1.356e-5, 1.465e-5, 1.578e-5, 1.696e-5, 1.818e-5, 1.880e-5, 1.944e-5, 2.009e-5, 2.074e-5,
     2.141e-5, 2.208e-5, 2.277e-5, 2.346e-5, 2.416e-5, 2.487e-5, 2.632e-5, 2.780e-5, 2.931e-5, 3.086e-5,
     3.243e-5, 3.565e-5, 3.898e-5, 4.241e-5, 4.593e-5, 4.954e-5, 5.890e-5, 6.871e-5, 7.892e-5, 8.951e-5],
)
_AIR_MU = np.array(
    [1.474e-5, 1.527e-5, 1.579e-5, 1.630e-5, 1.680e-5, 1.729e-5, 1.754e-5, 1.778e-5, 1.802e-5, 1.825e-5,
     1.849e-5, 1.872e-5, 1.895e-5, 1.918e-5, 1.941e-5, 1.963e-5, 2.008e-5, 2.052e-5, 2.096e-5, 2.139e-5,
     2.181e-5, 2.264e-5, 2.345e-5, 2.420e-5, 2.504e-5, 2.577e-5, 2.760e-5, 2.934e-5, 3.101e-5, 3.261e-5],
)
_AIR_NU = np.array(
    [9.319e-6, 1.008e-5, 1.087e-5, 1.169e-5, 1.252e-5, 1.338e-5, 1.382e-5, 1.426e-5, 1.470e-5, 1.516e-5,
     1.562e-5, 1.608e-5, 1.655e-5, 1.702e-5, 1.750e-5, 1.798e-5, 1.896e-5, 1.995e-5, 2.097e-5, 2.201e-5,
     2.306e-5, 2.522e-5, 2.745e-5, 2.975e-5, 3.212e-5, 3.455e-5, 4.091e-5, 4.765e-5, 5.475e-5, 6.219e-5],
)
_AIR_PR = np.array(
    [0.7440, 0.7436, 0.7425, 0.7408, 0.7387, 0.7362, 0.7350, 0.7336, 0.7323, 0.7309,
     0.7296, 0.7282, 0.7268, 0.7255, 0.7241, 0.7228, 0.7202, 0.7177, 0.7154, 0.7132,
     0.7111, 0.7073, 0.7041, 0.7014, 0.6992, 0.6974, 0.6946, 0.6935, 0.6937, 0.6948],
)


@dataclass(frozen=True)
class AirProperties:
    rho: float
    cp: float
    k: float
    mu: float
    nu: float
    alpha: float
    Pr: float


def air_properties(T_K: float) -> AirProperties:
    """Propriedades do ar a 1 atm por interpolação linear da tabela acima, em T_K (Kelvin)."""
    T_C = T_K - 273.15
    return AirProperties(
        rho=float(np.interp(T_C, _AIR_T_C, _AIR_RHO)),
        cp=float(np.interp(T_C, _AIR_T_C, _AIR_CP)),
        k=float(np.interp(T_C, _AIR_T_C, _AIR_K)),
        mu=float(np.interp(T_C, _AIR_T_C, _AIR_MU)),
        nu=float(np.interp(T_C, _AIR_T_C, _AIR_NU)),
        alpha=float(np.interp(T_C, _AIR_T_C, _AIR_ALPHA)),
        Pr=float(np.interp(T_C, _AIR_T_C, _AIR_PR)),
    )


# --------------------------------------------------------------------------- #
# Correlações de convecção natural interna (SPEC §4 — Eqs. 5.0-5.4)
# --------------------------------------------------------------------------- #


def rayleigh_number(T_s: float, T_int: float, Lc: float, g: float) -> tuple[float, AirProperties]:
    """Eq. 5.0 — número de Rayleigh e propriedades do ar avaliadas em T_f=(T_s+T_int)/2."""
    T_f = 0.5 * (T_s + T_int)
    beta = 1.0 / T_f  # gás ideal
    air = air_properties(T_f)
    Ra = g * beta * abs(T_s - T_int) * Lc**3 / (air.nu * air.alpha)
    return Ra, air


def h_floor(T1: float, T_int: float, Lc: float, g: float) -> float:
    """Eq. 5.1 — piso S1, placa horizontal com face superior aquecida (T1 >> T_int)."""
    Ra, air = rayleigh_number(T1, T_int, Lc, g)
    Nu = 0.54 * Ra**0.25 if Ra <= 1e7 else 0.15 * Ra ** (1 / 3)
    return Nu * air.k / Lc


def h_ceiling(T2: float, T_int: float, Lc: float, g: float) -> float:
    """Eq. 5.2 — teto S2, ramo condicional ao sinal de (T2-T_int)."""
    Ra, air = rayleigh_number(T2, T_int, Lc, g)
    if T2 < T_int:
        Nu = 0.54 * Ra**0.25 if Ra <= 1e7 else 0.15 * Ra ** (1 / 3)
    else:
        Nu = 0.27 * Ra**0.25
    return Nu * air.k / Lc


def h_vertical_wall(Tk: float, T_int: float, H: float, g: float) -> float:
    """Eq. 5.3 — paredes laterais S3-S6, Churchill-Chu (1975)."""
    Ra, air = rayleigh_number(Tk, T_int, H, g)
    Nu = (
        0.825 + 0.387 * Ra ** (1 / 6) / (1 + (0.492 / air.Pr) ** (9 / 16)) ** (8 / 27)
    ) ** 2
    return Nu * air.k / H


def compute_h_c(T: np.ndarray, T_int: float, p: "ChamberParams") -> np.ndarray:
    """Eq. 5.4 — vetor h_c das 6 superfícies, na ordem S1..S6."""
    h = np.empty(6)
    h[0] = h_floor(T[0], T_int, p.Lc_horizontal, p.g)
    h[1] = h_ceiling(T[1], T_int, p.Lc_horizontal, p.g)
    for k in range(2, 6):
        h[k] = h_vertical_wall(T[k], T_int, p.H, p.g)
    return h


# --------------------------------------------------------------------------- #
# Radiosidade (SPEC §5.1-5.2 — Eqs. 6.1-6.2)
# --------------------------------------------------------------------------- #


def solve_radiosity(T_K: np.ndarray, F: np.ndarray, eps: np.ndarray, sigma: float) -> np.ndarray:
    """Eq. 6.1 — [I-(I-eps)F] J = eps*sigma*T^4, resolvido por solver direto (regra 1, §7.1)."""
    identity = np.eye(6)
    A_matrix = identity - np.diag(1.0 - eps) @ F
    b = eps * sigma * T_K**4
    return np.linalg.solve(A_matrix, b)


def radiative_flux(T_K: np.ndarray, J: np.ndarray, areas: np.ndarray, eps: np.ndarray, sigma: float) -> np.ndarray:
    """Eq. 6.2 — fluxo radiativo líquido por superfície q_i."""
    return eps * areas / (1.0 - eps) * (sigma * T_K**4 - J)


# --------------------------------------------------------------------------- #
# Parâmetros da câmara (SPEC §1)
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class ChamberParams:
    W: float
    L: float
    H: float
    T1: float            # K (fixo, superfície ativa)
    T_inf: float          # K
    U: np.ndarray          # (6,) W/m2.K
    eps: np.ndarray         # (6,)
    sigma: float
    g: float
    areas: np.ndarray       # (6,) m2
    F: np.ndarray           # (6,6) fatores de forma F_i->j
    Lc_horizontal: float    # m (piso/teto, SPEC §4.2)
    mdot_inf: float         # kg/s (infiltração)
    cp_ar: float            # J/kg.K (avaliado em T_inf, Premissa A2)


def default_chamber_params() -> ChamberParams:
    """Constantes da câmara secadora (docs/SPEC.md §1)."""
    W, L, H = 8.0, 4.0, 3.0
    T1 = 350.0 + 273.15
    T_inf = 25.0 + 273.15
    U = np.full(6, 1.0)
    eps = np.full(6, 0.9)
    sigma = 5.670e-8
    g = 9.81
    V_inf_dot = 3.0 / 60.0  # 3 m3/min -> m3/s

    areas = np.array(chamber_surface_areas(W, L, H))
    F = np.array(build_view_factor_matrix(W, L, H))
    Lc_horizontal = (W * L) / (2 * (W + L))

    air_ext = air_properties(T_inf)
    mdot_inf = air_ext.rho * V_inf_dot
    cp_ar = air_ext.cp

    return ChamberParams(
        W=W, L=L, H=H, T1=T1, T_inf=T_inf, U=U, eps=eps, sigma=sigma, g=g,
        areas=areas, F=F, Lc_horizontal=Lc_horizontal, mdot_inf=mdot_inf, cp_ar=cp_ar,
    )


# --------------------------------------------------------------------------- #
# Resíduo do sistema acoplado e solução não linear (SPEC §5.3-5.7, §7)
# --------------------------------------------------------------------------- #


def residual(u: np.ndarray, p: ChamberParams, eval_count: list[int]) -> np.ndarray:
    """R(u) (Eq. 6.7): u=[T2,T3,T4,T5,T6,T_int]. Normalizado por Q_ref (regra 3, §7.1)."""
    eval_count[0] += 1
    u_safe = np.maximum(u, 1.0)  # guarda de domínio físico (regra 2, §7.1)

    T = np.empty(6)
    T[0] = p.T1
    T[1:6] = u_safe[0:5]
    T_int = u_safe[5]

    h_c = compute_h_c(T, T_int, p)
    J = solve_radiosity(T, p.F, p.eps, p.sigma)
    q = radiative_flux(T, J, p.areas, p.eps, p.sigma)

    R = np.empty(6)
    for idx, k in enumerate(range(1, 6)):  # superfícies passivas S2..S6 (Eq. 6.3)
        q_ref = p.U[k] * p.areas[k] * (p.T1 - p.T_inf)
        R[idx] = (
            h_c[k] * p.areas[k] * (T_int - T[k]) - q[k] - p.U[k] * p.areas[k] * (T[k] - p.T_inf)
        ) / q_ref

    q_ref_ar = p.mdot_inf * p.cp_ar * (p.T1 - p.T_inf)  # nó de ar (Eq. 6.4)
    conv_to_air = float(np.sum(h_c * p.areas * (T - T_int)))
    R[5] = (conv_to_air - p.mdot_inf * p.cp_ar * (T_int - p.T_inf)) / q_ref_ar

    logger.info(
        "eval=%-4d ||R||_inf=%.3e  T_int=%6.2f°C  T(2..6)=%s",
        eval_count[0],
        float(np.max(np.abs(R))),
        T_int - 273.15,
        np.round(T[1:6] - 273.15, 2).tolist(),
    )
    return R


def initial_guess(p: ChamberParams) -> np.ndarray:
    """Palpite inicial u0 (ver SPEC_DEVIATION 2 no topo do arquivo).

    Quebra a simetria entre superfícies passivas usando os fatores de forma
    F_(1->i) já disponíveis (espírito da Eq. 8.4), evitando o ponto degenerado
    T_i=T_int descrito em §7.2, sem reproduzir a rede de resistências completa.
    """
    T_p0 = p.T_inf + 0.85 * (p.T1 - p.T_inf)
    F1 = p.F[0, 1:6]
    delta = 0.1
    T_passive0 = p.T_inf + (T_p0 - p.T_inf) * (1.0 + delta * (F1 - F1.mean()))
    return np.concatenate([T_passive0, [T_p0]])


@dataclass(frozen=True)
class ChamberSolution:
    T_C: np.ndarray       # (6,) °C, S1..S6
    T_int_C: float
    T_ar_sai_C: float
    J: np.ndarray          # (6,) W/m2
    h_c: np.ndarray         # (6,) W/m2.K
    q: np.ndarray           # (6,) W
    Q_resist: float          # W
    result: optimize.OptimizeResult


def solve_chamber(p: ChamberParams, method: str = "lm") -> ChamberSolution:
    """Resolve R(u)=0 via scipy.optimize.root (method='lm' — ver SPEC_DEVIATION 1)."""
    u0 = initial_guess(p)
    eval_count = [0]

    logger.info(
        "Iniciando scipy.optimize.root(method=%r)  u0=%s°C",
        method,
        np.round(np.append(u0[:5], u0[5]) - 273.15, 2).tolist(),
    )

    result = optimize.root(residual, u0, args=(p, eval_count), method=method, tol=1e-8)

    logger.info(
        "Solver finalizado: success=%s  nfev=%d  message=%r",
        result.success, result.nfev, result.message,
    )
    if not result.success:
        logger.warning("scipy.optimize.root NÃO convergiu: %s", result.message)

    T = np.empty(6)
    T[0] = p.T1
    T[1:6] = np.maximum(result.x[0:5], 1.0)
    T_int = max(float(result.x[5]), 1.0)

    h_c = compute_h_c(T, T_int, p)
    J = solve_radiosity(T, p.F, p.eps, p.sigma)
    q = radiative_flux(T, J, p.areas, p.eps, p.sigma)
    Q_resist = float(h_c[0] * p.areas[0] * (p.T1 - T_int) + q[0] + p.U[0] * p.areas[0] * (p.T1 - p.T_inf))

    return ChamberSolution(
        T_C=T - 273.15,
        T_int_C=T_int - 273.15,
        T_ar_sai_C=T_int - 273.15,
        J=J,
        h_c=h_c,
        q=q,
        Q_resist=Q_resist,
        result=result,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    params = default_chamber_params()
    solution = solve_chamber(params, method="lm")

    labels = ["S1 (base)", "S2 (topo)", "S3 (x=0)", "S4 (x=W)", "S5 (y=0)", "S6 (y=L)"]

    print("\n=== Balanço de energia em regime permanente -- câmara 8x4x3 m ===\n")
    print(f"{'Superfície':<12}{'T [°C]':>10}{'h_c [W/m².K]':>16}{'J [W/m²]':>14}{'q [W]':>14}")
    for label, T_i, h_i, J_i, q_i in zip(labels, solution.T_C, solution.h_c, solution.J, solution.q):
        print(f"{label:<12}{T_i:>10.2f}{h_i:>16.3f}{J_i:>14.2f}{q_i:>14.2f}")

    print(f"\n{'T_int':<12}{solution.T_int_C:>10.2f} °C")
    print(f"{'T_ar_sai':<12}{solution.T_ar_sai_C:>10.2f} °C")
    print(f"{'Q_resist':<12}{solution.Q_resist:>10.2f} W")

    print("\nValidação (SPEC §10):")
    sum_q = float(solution.q.sum())
    print(f"  soma(q_i)            = {sum_q:.3e} W   [{'OK' if abs(sum_q) < 1e-3 else 'FALHA'}]  (conservação radiativa, item 3)")

    T_inf_C = params.T_inf - 273.15
    envelope_loss = float(np.sum(params.U * params.areas * (solution.T_C - T_inf_C)))
    infiltration_gain = params.mdot_inf * params.cp_ar * (solution.T_int_C - T_inf_C)
    closure_rhs = envelope_loss + infiltration_gain
    closure_err = solution.Q_resist - closure_rhs
    status = "OK" if abs(closure_err) < 1e-2 * abs(solution.Q_resist) else "FALHA"
    print(
        f"  fechamento energético: Q_resist={solution.Q_resist:.2f} W  vs.  "
        f"ΣU_iA_iΔT+ṁcpΔT={closure_rhs:.2f} W  (erro {closure_err:.2e} W)  [{status}]  (item 4)"
    )
    print(f"  convergência scipy.optimize.root: success={solution.result.success}  nfev={solution.result.nfev}")
