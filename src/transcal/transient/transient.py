"""Regime transitório da esfera vitrocerâmica (docs/SPEC.md Seção 6, Eqs. 7.1-7.8).

Importa T_int e as temperaturas das 6 superfícies da câmara em regime permanente
(consolidadas por balance/solver.py) para montar a temperatura de vizinhança
radiativa T_surr (Eq. 7.4) e o coeficiente combinado convecção+radiação
h_total que aquece a esfera ao longo da travessia (0-120 min). Calcula o
número de Biot e resolve a condução transiente 1D em coordenadas esféricas por
série analítica de autovalores (núcleo e superfície), plotando e salvando em
disco o histórico térmico resultante.

SPEC_DEVIATION (1): docs/SPEC.md §2/§8 prevê dois módulos, `transient/biot.py`
e `transient/radial_fdm.py` (este último por diferenças finitas). Por
instrução explícita desta task, ambos foram colapsados neste único arquivo e
o método numérico foi substituído pela solução analítica em série de
autovalores (raízes de zeta*cot(zeta)=1-Bi, Incropera Tabela 5.1 para esfera),
em vez de um esquema FDM radial.

SPEC_DEVIATION (2): h_total depende da própria temperatura de superfície da
esfera T_s,esf(t) (Eqs. 7.2-7.6), enquanto a série analítica em série exige
coeficiente convectivo e temperatura de vizinhança constantes ao longo do
tempo. Trata-se o problema de forma quase-estática: itera-se uma temperatura
de superfície representativa T_s,rep (usada para avaliar Ra_D, h_conv e h_rad)
até autoconsistência com a própria média temporal da curva de superfície
resultante -- poucas iterações bastam (ver `compute_transient_profile`).
Convecção e radiação são combinadas num único h_total e numa temperatura de
acionamento efetiva T_eff = (h_conv*T_int + h_rad*T_surr)/h_total, de modo que
a condição de contorno de 3º tipo padrão (h, T∞) da solução em série se
mantém válida com (h_total, T_eff).

Atenção de notação (SPEC §6.4): o Biot do critério lumped x distribuído
(Eq. 7.8) usa L_c,esf=D/6; o Biot que entra na equação de autovalores da série
(Incropera Tabela 5.1) usa o raio cheio r0=D/2=3*(D/6). Os dois são reportados
separadamente para não confundir as escalas.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend não interativo: salva em disco sem precisar de display

import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

from transcal.balance.solver import (
    ChamberParams,
    ChamberSolution,
    rayleigh_number,
)

logger = logging.getLogger("transcal.transient.transient")

# --------------------------------------------------------------------------- #
# Cinemática e critério de regime convectivo (SPEC §6.1 -- Eq. 7.1)
# --------------------------------------------------------------------------- #


def sphere_velocity(W: float, t_total_s: float) -> float:
    """Velocidade média de travessia da esfera (SPEC §6.1)."""
    return W / t_total_s


def grashof_over_reynolds2(Ts_K: float, T_int_K: float, D: float, V_esf: float, g: float) -> float:
    """Eq. 7.1 -- Gr_D/Re_D²; beta=1/T_f (gás ideal) avaliado na temperatura de filme."""
    T_f = 0.5 * (Ts_K + T_int_K)
    beta = 1.0 / T_f
    return g * beta * abs(Ts_K - T_int_K) * D / V_esf**2


# --------------------------------------------------------------------------- #
# Coeficientes convectivo e radiativo da esfera (SPEC §6.2-6.4 -- Eqs. 7.2-7.6)
# --------------------------------------------------------------------------- #


def nusselt_sphere(Ra_D: float, Pr: float) -> float:
    """Eq. 7.2 -- Churchill (1975), válida para Ra_D<=1e11, Pr>=0.7."""
    if Ra_D > 1e11 or Pr < 0.7:
        logger.warning(
            "Correlação de Churchill (esfera) fora do domínio de validade: Ra_D=%.3e, Pr=%.3f",
            Ra_D, Pr,
        )
    return 2.0 + 0.589 * Ra_D**0.25 / (1.0 + (0.469 / Pr) ** (9 / 16)) ** (4 / 9)


def h_rad_sphere(Ts_K: float, T_surr_K: float, eps_esf: float, sigma: float) -> float:
    """Eq. 7.5 -- coeficiente radiativo linearizado (esfera pintada de preto, eps_esf~=1, Premissa A6)."""
    return eps_esf * sigma * (Ts_K + T_surr_K) * (Ts_K**2 + T_surr_K**2)


def surrounding_radiation_temperature(T_K: np.ndarray, areas: np.ndarray, eps: np.ndarray) -> float:
    """Eq. 7.4 -- temperatura de vizinhança radiativa efetiva (média de 4ª potência, peso eps_k*A_k)."""
    return float((np.sum(eps * areas * T_K**4) / np.sum(eps * areas)) ** 0.25)


def biot_number(h: float, Lc: float, k: float) -> float:
    """Bi = h*Lc/k, genérico (ver nota de notação no topo do arquivo sobre a escala Lc usada)."""
    return h * Lc / k


# --------------------------------------------------------------------------- #
# Série analítica de autovalores -- condução transiente 1D esférica
# --------------------------------------------------------------------------- #


def sphere_eigenvalues(Bi: float, n_terms: int) -> np.ndarray:
    """Raízes de zeta*cot(zeta) = 1-Bi, uma em cada intervalo ((n-1)pi, n*pi),
    n=1..n_terms (Incropera, condução transiente esférica com convecção, Tabela 5.1)."""

    def f(zeta: float) -> float:
        return zeta * np.cos(zeta) / np.sin(zeta) - (1.0 - Bi)

    roots = np.empty(n_terms)
    for n in range(1, n_terms + 1):
        lo = (n - 1) * np.pi + 1e-9
        hi = n * np.pi - 1e-9
        roots[n - 1] = optimize.brentq(f, lo, hi)
    return roots


def series_coefficients(zeta: np.ndarray) -> np.ndarray:
    """C_n = 4(sin(zeta_n)-zeta_n*cos(zeta_n)) / (2*zeta_n - sin(2*zeta_n)) (Incropera, esfera)."""
    return 4.0 * (np.sin(zeta) - zeta * np.cos(zeta)) / (2.0 * zeta - np.sin(2.0 * zeta))


def _effective_n_terms(zeta: np.ndarray, Fo_min: float, tol: float) -> int:
    """Menor N tal que exp(-zeta_N²*Fo_min) < tol -- cauda da série desprezível no menor Fo>0 da malha."""
    tail = np.exp(-(zeta**2) * Fo_min)
    below = np.nonzero(tail < tol)[0]
    return int(below[0] + 1) if below.size else zeta.size


def eigenseries(Bi: float, Fo_min: float, max_terms: int = 400, tol: float = 1e-12) -> tuple[np.ndarray, np.ndarray, int]:
    """Autovalores/coeficientes truncados no menor N suficiente para o Fo>0 mais exigente da malha."""
    zeta_full = sphere_eigenvalues(Bi, max_terms)
    n_terms = _effective_n_terms(zeta_full, Fo_min, tol)
    zeta = zeta_full[:n_terms]
    return zeta, series_coefficients(zeta), n_terms


def evaluate_series(Fo: np.ndarray, zeta: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """theta*(Fo) no núcleo (r*=0) e na superfície (r*=1), vetorizado sobre o vetor Fo."""
    decay = np.exp(-np.outer(Fo, zeta**2))  # (n_pontos, n_termos)
    theta_center = decay @ C
    theta_surface = decay @ (C * np.sin(zeta) / zeta)
    return theta_center, theta_surface


# --------------------------------------------------------------------------- #
# Parâmetros da esfera (SPEC §1, requisitos.md §3) e orquestração
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SphereParams:
    D: float          # m
    k: float           # W/m.°C
    rho: float          # kg/m3
    cp: float            # J/kg.°C
    eps: float            # emissividade (Premissa A6, tinta preta ~= 1.0)
    T0_C: float            # °C, condição inicial (requisitos.md §3)
    W_chamber: float        # m, comprimento percorrido (eixo x da câmara)
    t_total_s: float         # s, tempo total de travessia (120 min)


def default_sphere_params() -> SphereParams:
    """Esfera vitrocerâmica (requisitos.md §3)."""
    return SphereParams(
        D=0.5, k=1.4, rho=2520.0, cp=790.0, eps=1.0,
        T0_C=25.0, W_chamber=8.0, t_total_s=120.0 * 60.0,
    )


@dataclass(frozen=True)
class SphereTransientResult:
    t_min: np.ndarray          # (n,) minutos, 0..120
    T_core_C: np.ndarray         # (n,) °C
    T_surface_C: np.ndarray       # (n,) °C
    Bi_series: float                # h_total*r0/k -- Biot do autovalor (Incropera, r0=D/2)
    Bi_lumped: float                 # h_total*(D/6)/k -- Eq. 7.8, critério lumped x distribuído
    h_conv: float                     # W/m².K
    h_rad: float                       # W/m².K
    h_total: float                      # W/m².K
    Ra_D: float
    T_eff_C: float                       # °C, acionamento combinado conv+rad
    T_surr_C: float                       # °C, Eq. 7.4
    T_int_C: float                         # °C, importado de balance/solver.py
    Gr_over_Re2: float                      # Eq. 7.1
    n_terms: int
    n_iter: int                               # iterações de autoconsistência de T_s,rep


def compute_transient_profile(
    chamber: ChamberParams,
    solution: ChamberSolution,
    sphere: SphereParams,
    g: float = 9.81,
    sigma: float = 5.670e-8,
    n_points: int = 241,
    ts_tol_K: float = 0.05,
    max_iter: int = 25,
) -> SphereTransientResult:
    """Monta T_surr e T_int a partir do balanço de regime permanente já resolvido
    (`solution`, de balance/solver.py) e resolve a condução transiente da esfera.
    """
    T_chamber_K = solution.T_C + 273.15
    T_int_K = solution.T_int_C + 273.15
    T_surr_K = surrounding_radiation_temperature(T_chamber_K, chamber.areas, chamber.eps)
    T0_K = sphere.T0_C + 273.15
    r0 = sphere.D / 2.0
    alpha = sphere.k / (sphere.rho * sphere.cp)

    t_s = np.linspace(0.0, sphere.t_total_s, n_points)
    Fo = alpha * t_s / r0**2
    Fo_min = Fo[1]  # primeiro Fo>0; Fo[0]=0 é tratado como condição inicial exata

    V_esf = sphere_velocity(sphere.W_chamber, sphere.t_total_s)

    T_s_rep_K = 0.5 * (T0_K + T_int_K)
    for n_iter in range(1, max_iter + 1):
        Ra_D, air = rayleigh_number(T_s_rep_K, T_int_K, sphere.D, g)
        h_conv = nusselt_sphere(Ra_D, air.Pr) * air.k / sphere.D
        h_rad = h_rad_sphere(T_s_rep_K, T_surr_K, sphere.eps, sigma)
        h_total = h_conv + h_rad
        T_eff_K = (h_conv * T_int_K + h_rad * T_surr_K) / h_total
        Bi_series = biot_number(h_total, r0, sphere.k)

        zeta, C, n_terms = eigenseries(Bi_series, Fo_min)
        theta_core, theta_surface = evaluate_series(Fo, zeta, C)
        T_core_K = T_eff_K + theta_core * (T0_K - T_eff_K)
        T_surface_K = T_eff_K + theta_surface * (T0_K - T_eff_K)
        T_core_K[0] = T0_K
        T_surface_K[0] = T0_K

        T_s_rep_new_K = float(np.mean(T_surface_K))
        logger.info(
            "autoconsistência it=%-2d T_s,rep=%6.2f°C -> %6.2f°C  Bi_series=%.4f  h_total=%.3f W/m².K",
            n_iter, T_s_rep_K - 273.15, T_s_rep_new_K - 273.15, Bi_series, h_total,
        )
        if abs(T_s_rep_new_K - T_s_rep_K) < ts_tol_K:
            T_s_rep_K = T_s_rep_new_K
            break
        T_s_rep_K = T_s_rep_new_K

    Bi_lumped = biot_number(h_total, sphere.D / 6.0, sphere.k)
    Gr_Re2 = grashof_over_reynolds2(T_s_rep_K, T_int_K, sphere.D, V_esf, g)

    return SphereTransientResult(
        t_min=t_s / 60.0,
        T_core_C=T_core_K - 273.15,
        T_surface_C=T_surface_K - 273.15,
        Bi_series=Bi_series,
        Bi_lumped=Bi_lumped,
        h_conv=h_conv,
        h_rad=h_rad,
        h_total=h_total,
        Ra_D=Ra_D,
        T_eff_C=T_eff_K - 273.15,
        T_surr_C=T_surr_K - 273.15,
        T_int_C=solution.T_int_C,
        Gr_over_Re2=Gr_Re2,
        n_terms=n_terms,
        n_iter=n_iter,
    )


# --------------------------------------------------------------------------- #
# Plot e exportação automática em disco (SPEC §7.4, item 4)
# --------------------------------------------------------------------------- #


def _default_figure_path() -> Path:
    return Path(__file__).resolve().parents[3] / "figures" / "sphere_temperature_history.png"


def plot_sphere_history(result: SphereTransientResult, out_path: Path | None = None) -> Path:
    """Plota T_núcleo(t) e T_superfície(t), 0-120 min, e salva a figura em disco."""
    out_path = out_path or _default_figure_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(result.t_min, result.T_surface_C, color="tab:red", label="Superfície")
    ax.plot(result.t_min, result.T_core_C, color="tab:blue", label="Núcleo")
    ax.set_xlabel("Tempo [min]")
    ax.set_ylabel("Temperatura [°C]")
    ax.set_title("Histórico térmico da esfera vitrocerâmica (0-120 min)")
    ax.set_xlim(0, result.t_min[-1])
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    logger.info("Figura salva em %s", out_path)
    return out_path


if __name__ == "__main__":
    from transcal.balance.solver import default_chamber_params, solve_chamber

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    chamber = default_chamber_params()
    solution = solve_chamber(chamber, method="lm")
    sphere = default_sphere_params()
    result = compute_transient_profile(chamber, solution, sphere)

    print("\n=== Regime transitório da esfera vitrocerâmica (0-120 min) ===\n")
    print(f"T_int (importado de solver.py) = {result.T_int_C:.2f} °C")
    print(f"T_surr (Eq. 7.4)                = {result.T_surr_C:.2f} °C")
    print(f"T_eff (acionamento combinado)   = {result.T_eff_C:.2f} °C")
    print(f"Ra_D                             = {result.Ra_D:.3e}")
    print(f"h_conv                           = {result.h_conv:.3f} W/m².K")
    print(f"h_rad                            = {result.h_rad:.3f} W/m².K")
    print(f"h_total                          = {result.h_total:.3f} W/m².K")
    print(f"Bi_series (r0=D/2, autovalor)    = {result.Bi_series:.4f}")
    print(f"Bi_lumped (L_c=D/6, Eq. 7.8)     = {result.Bi_lumped:.4f}")
    print(f"Termos da série utilizados       = {result.n_terms}")
    print(f"Iterações de autoconsistência    = {result.n_iter}")

    print("\nValidação (SPEC §10 / Premissas §9):")
    modelo = "distribuído (núcleo x superfície)" if result.Bi_lumped >= 0.1 else "capacitância concentrada (lumped)"
    print(f"  critério Bi_lumped>=0.1: {result.Bi_lumped:.4f} -> modelo {modelo}  (item 6, Premissa A8)")
    print(
        f"  Gr_D/Re_D² = {result.Gr_over_Re2:.3e} >> 1: convecção forçada desprezável "
        f"[{'OK' if result.Gr_over_Re2 > 100 else 'FALHA'}]  (Premissa A5)"
    )
    print(f"  domínio de validade de Churchill (Ra_D<=1e11): [{'OK' if result.Ra_D <= 1e11 else 'FALHA'}]")

    fig_path = plot_sphere_history(result)
    print(f"\nFigura salva em: {fig_path}")

    print(f"\n{'t [min]':>10}{'T_núcleo [°C]':>16}{'T_superfície [°C]':>20}")
    for t_chk in (0, 30, 60, 90, 120):
        T_core_chk = float(np.interp(t_chk, result.t_min, result.T_core_C))
        T_surf_chk = float(np.interp(t_chk, result.t_min, result.T_surface_C))
        print(f"{t_chk:>10}{T_core_chk:>16.2f}{T_surf_chk:>20.2f}")
