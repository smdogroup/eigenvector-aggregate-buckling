import argparse

"""
Adjust: N_a, N_b, lam_a, lam_b, and np.max([0, N_a - 1]) for accuracy
"""


def parse_cmd_args():
    problem = {
        "problem": "buckling",  # ["natural_frequency", "buckling"]
        "domain": "building",  # ["square", "beam", "lbracket", "building", "leg", "rhombus"]
        "objf": "frequency",  #  ["frequency", "stress", "volume", "compliance", "displacement"]
        "confs": [
            "volume_ub",
            "compliance",
        ],  # ["volume_ub", "volume_lb", "frequency", "stress", "displacement", "compliance"]
        "nx": 100,  # number of elements along x direction
        "frequency_scale": 3.5,  # scale the frequency objective obj = frequency * scale
        "stress_scale": 1.0,  # scale the stress objective obj = stress * scale
        "compliance_scale": 1e5,  # scale the compliance objective obj = compliance * scale
    }
    
    constraint_bounds = {
        "omega_lb": None,
        "stress_ub": None,
        "compliance_ub": None,
        "vol_frac_ub": None,
        "vol_frac_lb": None,
        "dis_ub": None,
    }

    ks_rho = {
        "ks_rho_buckling": 1000.0,
        "ks_rho_natural_freq": 1000.0,
        "ks_rho_stress": 10.0,
        "ks_rho_freq": 160.0, # from ferrari2021 paper
    }

    softmax = {
        "N_a": 0,  # lower bound of selected indices of eigenvalues
        "N_b": 5,  # upper bound of selected indices of eigenvalues
        "atype": 1,  # 0: 0-b based index, N_a=0, "exp"; 1: a-b based index
        "fun": "tanh",  # ["exp", "sech", "tanh", "erf", "erfc", "sigmoid", "ncdf"]:
    }

    filter = {
        "filter": "spatial",  # ["spatial", "helmholtz"]
        "m0": 0.0,  # magnitude of non-design mass
        "r0": 2.1,  # filter radius = r0 * lx / nx
        "ptype_K": "simp",  # material penalization for stiffness matrix: ["ramp", "simp"]
        "ptype_M": "linear",  # material penalization for mass matrix: ["ramp", "msimp", "linear"]
        "p": 3.0,  # SIMP penalization parameter
        "q": 5.0,  # RAMP penalization parameter
        "rho0_K": 1e-3,  # rho offset to prevent singular K
        "rho0_M": 1e-7,  # rho offset to prevent singular M
        "proj": True,  # projector for filter
        "beta0": 2.0,  # projector parameter at the beginning
    }

    optimizer = {
        "optimizer": "pmma",  # ["pmma", "mma4py", "tr"]
        "movelim": 0.2,  # move limit for design variables, for mma4py only
        "lb": 1e-06,  # lower bound of design variables
        "maxit": 200,  # maximum number of iterations
    }

    check = {
        "kokkos": True,
        "check_gradient": False,
        "check_kokkos": False,
        "note": "",
    }

    list_of_defaults = [
        problem,
        ks_rho,
        softmax,
        check,
        constraint_bounds,
        filter,
        optimizer,
    ]

    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    for default in list_of_defaults:
        for arg, default in default.items():
            if isinstance(default, bool):
                p.add_argument(
                    f"--{arg.replace('_', '-')}",
                    default=default,
                    action="store_true",
                )
            elif default is None:
                p.add_argument(
                    f"--{arg.replace('_', '-')}", default=default, type=float
                )
            elif isinstance(default, list):
                p.add_argument(
                    f"--{arg.replace('_', '-')}",
                    default=default,
                    nargs="*",
                )
            else:
                p.add_argument(
                    f"--{arg.replace('_', '-')}",
                    default=default,
                    type=type(default),
                )

    # OS
    p.add_argument("--prefix", default="output", type=str, help="output folder")

    # Analysis
    p.add_argument(
        "--assume-same-element",
        action="store_true",
        help="assume all elements have identical shape",
    )
    p.add_argument(
        "--m0-block-frac",
        default=0.0,
        type=float,
        help="fraction of the size of non-design mass block with respect to the domain",
    )
    p.add_argument(
        "--stress-relax", default=0.3, type=float, help="stress relaxation factor"
    )
    p.add_argument(
        "--mode",
        default=1,
        type=int,
        help='mode number, only effective when "displacement" is in the confs',
    )

    args = p.parse_args()

    return args
