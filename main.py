"""Orquestrador final do sistema (docs/SPEC.md §7.4, §8; requisitos.md §4).

Executa, em sequência, os três módulos de física implementados:

1. `transcal.radiation.view_factors` — fatores de forma F_i->j da câmara (item 1).
2. `transcal.balance.solver`          — balanço de energia em regime permanente
   da câmara (itens 2, 3, 4, 5, 6).
3. `transcal.transient.transient`     — regime transitório da esfera vitrocerâmica
   (item 7, perfil núcleo/superfície + gráfico).

Compila os resultados dos 7 itens de `requisitos.md §4` em um dicionário
estruturado (`compilar_resultados`), no schema canônico de `docs/SPEC.md §7.4`,
usando `pandas.DataFrame` para os dois resultados naturalmente tabulares: a
matriz F_ij (item 1) e a série temporal núcleo/superfície da esfera (item 7).
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from transcal.radiation.view_factors import (
    SURFACE_LABELS,
    build_view_factor_matrix,
    chamber_surface_areas,
)
from transcal.balance.solver import ChamberSolution, default_chamber_params, solve_chamber
from transcal.transient.transient import (
    SphereTransientResult,
    compute_transient_profile,
    default_sphere_params,
    plot_sphere_history,
)

logger = logging.getLogger("transcal.main")


def compilar_resultados(
    F: list[list[float]],
    solution: ChamberSolution,
    transient: SphereTransientResult,
    fig_path: str,
) -> dict[int, dict[str, Any]]:
    """Monta o dicionário estruturado dos itens 1-7 de `requisitos.md §4`,
    no schema de campos de `docs/SPEC.md §7.4`."""

    df_F = pd.DataFrame(np.round(F, 3), index=list(SURFACE_LABELS), columns=list(SURFACE_LABELS))

    df_sphere_profile = pd.DataFrame(
        {
            "t_min": transient.t_min,
            "T_superficie_C": transient.T_surface_C,
            "T_nucleo_C": transient.T_core_C,
        }
    )

    return {
        1: {
            "campo": "F_ij",
            "valor": df_F,
            "unidade": "adimensional",
            "equacao": "4.A, 4.B",
        },
        2: {
            "campo": "h_c",
            "valor": pd.Series(solution.h_c, index=list(SURFACE_LABELS)),
            "unidade": "W/m².K",
            "equacao": "5.1-5.4",
        },
        3: {
            "campo": "T_surfaces, T_int",
            "valor": {
                "T_surfaces": pd.Series(solution.T_C[1:6], index=list(SURFACE_LABELS[1:6])),
                "T_int": solution.T_int_C,
            },
            "unidade": "°C",
            "equacao": "6.3, 6.4, 6.7",
        },
        4: {
            "campo": "T_ar_sai",
            "valor": solution.T_ar_sai_C,
            "unidade": "°C",
            "equacao": "6.5",
        },
        5: {
            "campo": "J",
            "valor": pd.Series(solution.J, index=list(SURFACE_LABELS)),
            "unidade": "W/m²",
            "equacao": "6.1",
        },
        6: {
            "campo": "Q_resist",
            "valor": solution.Q_resist,
            "unidade": "W",
            "equacao": "6.6",
        },
        7: {
            "campo": "sphere_profile",
            "valor": df_sphere_profile,
            "plot": fig_path,
            "unidade": "°C",
            "equacao": "7.1-7.8",
        },
    }


def run() -> dict[int, dict[str, Any]]:
    """Executa o pipeline completo: geometria -> F_ij -> balanço da câmara -> esfera."""
    areas = chamber_surface_areas(8.0, 4.0, 3.0)
    F = build_view_factor_matrix(8.0, 4.0, 3.0)

    chamber_params = default_chamber_params()
    solution = solve_chamber(chamber_params, method="lm")

    sphere_params = default_sphere_params()
    transient = compute_transient_profile(chamber_params, solution, sphere_params)
    fig_path = plot_sphere_history(transient)

    logger.info("Áreas das 6 superfícies (m²): %s", np.round(areas, 3).tolist())

    return compilar_resultados(F, solution, transient, str(fig_path))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    resultados = run()

    print("\n=== Resultados consolidados (requisitos.md §4, itens 1-7) ===\n")
    for item, dados in resultados.items():
        print(f"--- Item {item}: {dados['campo']} [{dados['unidade']}] (Eq. {dados['equacao']}) ---")
        print(dados["valor"])
        print()
