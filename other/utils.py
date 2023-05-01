from time import perf_counter_ns
import os


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
