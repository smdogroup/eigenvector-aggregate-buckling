import os
from shutil import rmtree
from time import perf_counter_ns
from datetime import datetime

import numpy as np


class MyProfiler:
    counter = 0  # a static variable
    timer_is_on = True
    print_to_stdout = False
    buffer = []
    istart = []  # stack of indices of open parantheses
    pairs = {}
    t_min = 1  # unit: ms
    log_name = "profiler.log"
    old_log_removed = False
    saved_times = {}

    @staticmethod
    def timer_set_log_path(log_path):
        MyProfiler.log_name = log_path

    @staticmethod
    def timer_set_threshold(t: float):
        """
        Don't show entries with elapse time smaller than this. Unit: ms
        """
        MyProfiler.t_min = t
        return

    @staticmethod
    def timer_to_stdout():
        """
        print the profiler output to stdout, otherwise save it as a file
        """
        MyProfiler.print_to_stdout = True
        return

    @staticmethod
    def timer_on():
        """
        Call this function before execution to switch on the profiler
        """
        MyProfiler.timer_is_on = True
        return

    @staticmethod
    def timer_off():
        """
        Call this function before execution to switch off the profiler
        """
        MyProfiler.timer_is_on = False
        return

    @staticmethod
    def time_this(func):
        """
        Decorator: time the execution of a function
        """
        tab = "    "
        fun_name = func.__qualname__

        if not MyProfiler.timer_is_on:

            def wrapper(*args, **kwargs):
                ret = func(*args, **kwargs)
                return ret

            return wrapper

        def wrapper(*args, **kwargs):
            info_str = f"{tab*MyProfiler.counter}{fun_name}() called"
            entry = {"msg": f"{info_str:<40s}", "type": "("}
            MyProfiler.buffer.append(entry)

            MyProfiler.counter += 1
            t0 = perf_counter_ns()
            ret = func(*args, **kwargs)
            t1 = perf_counter_ns()
            t_elapse = (t1 - t0) / 1e6  # unit: ms
            MyProfiler.counter -= 1

            info_str = f"{tab*MyProfiler.counter}{fun_name}() return"
            entry = {
                "msg": f"{info_str:<80s} ({t_elapse:.2f} ms)",
                "type": ")",
                "fun_name": fun_name,
                "t": t_elapse,
            }
            MyProfiler.buffer.append(entry)

            # Once the most outer function returns, we filter the buffer such
            # that we only keep entry pairs whose elapse time is above threshold
            if MyProfiler.counter == 0:
                for idx, entry in enumerate(MyProfiler.buffer):
                    if entry["type"] == "(":
                        MyProfiler.istart.append(idx)
                    if entry["type"] == ")":
                        try:
                            start_idx = MyProfiler.istart.pop()
                            if entry["t"] > MyProfiler.t_min:
                                MyProfiler.pairs[start_idx] = idx
                        except IndexError:
                            print("[Warning]Too many return message")

                # Now our stack should be empty, otherwise we have unpaired
                # called/return message
                if MyProfiler.istart:
                    print("[Warning]Too many called message")

                # Now, we only keep the entries for expensive function calls
                idx = list(MyProfiler.pairs.keys()) + list(MyProfiler.pairs.values())
                if idx:
                    idx.sort()
                keep_buffer = [MyProfiler.buffer[i] for i in idx]

                if MyProfiler.print_to_stdout:
                    for entry in keep_buffer:
                        print(entry["msg"])
                else:
                    if (
                        os.path.exists(MyProfiler.log_name)
                        and not MyProfiler.old_log_removed
                    ):
                        os.remove(MyProfiler.log_name)
                        MyProfiler.old_log_removed = True
                    with open(MyProfiler.log_name, "a") as f:
                        for entry in keep_buffer:
                            f.write(entry["msg"] + "\n")

                # Save time information to dictionary
                for entry in keep_buffer:
                    if "t" in entry.keys():
                        _fun_name = entry["fun_name"]
                        _t = entry["t"]
                        if _fun_name in MyProfiler.saved_times.keys():
                            MyProfiler.saved_times[_fun_name].append(_t)
                        else:
                            MyProfiler.saved_times[_fun_name] = [_t]

                # Reset buffer and pairs
                MyProfiler.buffer = []
                MyProfiler.pairs = {}
            return ret

        return wrapper


time_this = MyProfiler.time_this
timer_on = MyProfiler.timer_on
timer_off = MyProfiler.timer_off
timer_to_stdout = MyProfiler.timer_to_stdout
timer_set_threshold = MyProfiler.timer_set_threshold
timer_set_log_path = MyProfiler.timer_set_log_path


def create_folder(args):
    if not os.path.isdir(args.prefix):
        os.mkdir(args.prefix)
    if args.confs == ["stress", "frequency"]:
        args.confs = ["frequency", "stress"]
    if args.confs == ["frequency", "volume_ub"]:
        args.confs = ["volume_ub", "frequency"]
    if args.confs == ["stress", "volume_ub"]:
        args.confs = ["volume_ub", "stress"]
    if args.confs == ["volume_ub", "volume_lb"]:
        args.confs = ["volume_lb", "volume_ub"]
    if args.confs == ["volume_ub", "displacement"]:
        args.confs = ["displacement", "volume_ub"]

    name = f"{args.domain}"
    if not os.path.isdir(os.path.join(args.prefix, name)):
        os.mkdir(os.path.join(args.prefix, name))
    args.prefix = os.path.join(args.prefix, name)

    # make a folder inside each domain folder to store the results of each run
    name2 = f"{args.objf}{args.confs}"
    if not os.path.isdir(os.path.join(args.prefix, name2)):
        os.mkdir(os.path.join(args.prefix, name2))
    args.prefix = os.path.join(args.prefix, name2)

    o = f"{args.optimizer[0]}"
    args.prefix = os.path.join(args.prefix, o)

    n = f"{args.nx}"
    args.prefix = args.prefix + ", n=" + n

    if args.confs != []:
        v = f"{args.vol_frac_ub:.2f}"
        args.prefix = args.prefix + ", v=" + v

    if "displacement" in args.confs:
        if args.mode != 1:
            args.prefix = args.prefix + ", mode=" + str(args.mode)

    if "displacement" in args.confs or "stress" in args.confs:
        N_a = f"{args.N_a}"
        args.prefix = args.prefix + ", Na=" + N_a

        N_b = f"{args.N_b}"
        args.prefix = args.prefix + ", Nb=" + N_b

    r = f"{args.r0}"
    args.prefix = args.prefix + ", r=" + r

    if "compliance-buckling" in args.objf:
        w = f"{args.weight:.2f}"
        args.prefix = args.prefix + ", w=" + w

    if args.m0_block_frac != 0.0:
        m = f"{args.m0_block_frac:.2f}"
        args.prefix = args.prefix + ", m0=" + m

    if "frequency" in args.confs:
        if args.omega_lb != 0.0:
            w = f"{args.omega_lb}"
            args.prefix = args.prefix + ", w=" + w

    if "compliance" in args.confs:
        c = f"{args.compliance_ub_percent:.2f}"
        args.prefix = args.prefix + ", c=" + c

    if "stress" in args.confs:
        s = f"{args.stress_ub}"
        args.prefix = args.prefix + ", s=" + s

    if "displacement" in args.confs:
        d = f"{args.dis_ub:.2f}"
        args.prefix = args.prefix + ", d=" + d

    # if args.proj:
    #     beta0 = f"{args.beta0}"
    #     args.prefix = args.prefix + ", beta0=" + beta0
    #     delta_beta = f"{args.delta_beta}"
    #     args.prefix = args.prefix + ", dbeta=" + delta_beta

    # if args.dis_ub is not None:
    #     d = f"{args.dis_ub:.3f}"
    #     args.prefix = os.path.join(args.prefix, d)

    # if args.stress_ub is not None:
    #     s = f"{args.stress_ub/ 1e12}"
    #     args.prefix = os.path.join(args.prefix, s)

    # # create a folder inside the result folder to store the results of each run
    # if args.confs == ["volume_ub", "stress"]:
    #     args.prefix = os.path.join(
    #         args.prefix,
    #         o + ", n=" + n + ", s=" + s + args.note,
    #     )
    # elif args.confs == ["volume_ub"]:
    #     args.prefix = os.path.join(
    #         args.prefix,
    #         o + ", n=" + n + args.note,
    #     )
    # elif args.confs == ["displacement", "volume_ub"]:
    #     args.prefix = os.path.join(
    #         args.prefix,
    #         o + ", n=" + n + ", d=" + d + args.note,
    #     )
    # elif args.confs == ["volume_ub", "displacement", "stress"]:
    #     args.prefix = os.path.join(
    #         args.prefix,
    #         o + ", n=" + n + ", d=" + d + ", s=" + s + args.note,
    #     )

    # if os.path.isdir(args.prefix):
    #     rmtree(args.prefix)
    if not os.path.isdir(args.prefix):
        os.mkdir(args.prefix)

    if args.note != "":
        args.prefix = os.path.join(args.prefix, args.note)
        args.prefix = args.prefix + ", " + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    else:
        args.prefix = os.path.join(
            args.prefix, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
    os.mkdir(args.prefix)

    return args


def to_vtk(vtk_path, conn, X, nodal_sols={}, cell_sols={}, nodal_vecs={}, cell_vecs={}):
    """
    Generate a vtk given conn, X, and optionally list of nodal solutions

    Args:
        nodal_sols: dictionary of arrays of length nnodes
        cell_sols: dictionary of arrays of length nelems
        nodal_vecs: dictionary of list of components [vx, vy], each has length nnodes
        cell_vecs: dictionary of list of components [vx, vy], each has length nelems
    """
    # vtk requires a 3-dimensional data point
    X = np.append(X, np.zeros((X.shape[0], 1)), axis=1)

    nnodes = X.shape[0]
    nelems = conn.shape[0]

    # Create a empty vtk file and write headers
    with open(vtk_path, "w") as fh:
        fh.write("# vtk DataFile Version 3.0\n")
        fh.write("my example\n")
        fh.write("ASCII\n")
        fh.write("DATASET UNSTRUCTURED_GRID\n")

        # Write nodal points
        fh.write("POINTS {:d} double\n".format(nnodes))
        for x in X:
            row = f"{x}"[1:-1]  # Remove square brackets in the string
            fh.write(f"{row}\n")

        # Write connectivity
        size = 5 * nelems

        fh.write(f"CELLS {nelems} {size}\n")
        for c in conn:
            node_idx = f"{c}"[1:-1]  # remove square bracket [ and ]
            npts = 4
            fh.write(f"{npts} {node_idx}\n")

        # Write cell type
        fh.write(f"CELL_TYPES {nelems}\n")
        for c in conn:
            vtk_type = 9
            fh.write(f"{vtk_type}\n")

        # Write solution
        if nodal_sols or nodal_vecs:
            fh.write(f"POINT_DATA {nnodes}\n")

        if nodal_sols:
            for name, data in nodal_sols.items():
                fh.write(f"SCALARS {name} double 1\n")
                fh.write("LOOKUP_TABLE default\n")
                for val in data:
                    fh.write(f"{val}\n")

        if nodal_vecs:
            for name, data in nodal_vecs.items():
                fh.write(f"VECTORS {name} double\n")
                for val in np.array(data).T:
                    fh.write(f"{val[0]} {val[1]} 0.\n")

        if cell_sols or cell_vecs:
            fh.write(f"CELL_DATA {nelems}\n")

        if cell_sols:
            for name, data in cell_sols.items():
                fh.write(f"SCALARS {name} double 1\n")
                fh.write("LOOKUP_TABLE default\n")
                for val in data:
                    fh.write(f"{val}\n")

        if cell_vecs:
            for name, data in cell_vecs.items():
                fh.write(f"VECTORS {name} double\n")
                for val in np.array(data).T:
                    fh.write(f"{val[0]} {val[1]} 0.\n")

    return
