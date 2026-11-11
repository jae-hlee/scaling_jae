"""Largest pristine diamond-Si cell a force field's own engine runs on one GB10.

For a cell of 8 i^3 atoms (a = 5.43 A, the cell of the ALIGNN-FF sweeps) the
engine is started as its own process and does one static energy and force
evaluation followed by a few more steps. i grows until a run fails, then the
largest working size is bracketed by bisection. Every size is recorded as soon
as it ends, and the size being attempted is written before it starts, so a job
that is killed loses nothing and a resubmitted job continues.

A size counts as working only if the engine finished AND its static energy per
atom equals that of the 512-atom cell: 32-bit index overflow in an engine can
give wrong numbers instead of an error.

The GB10 has one pool of memory for CPU and GPU and nvidia-smi reports no
memory use on it, so memory is the largest drop of MemAvailable in
/proc/meminfo while the engine runs (everything on the node, so nothing else
should be running there), next to the peak resident memory of the process.

Engines (standard library only):
  gpumd   NEP. --potential nep.txt. Static evaluation, then NVE steps.
  lammps  SNAP and others. --pair_file holds the pair_style and pair_coeff
          lines, --copy the potential files. run 0, then NVE steps.
  python  a script that evaluates one size and prints "RESULT {json}"
          (py_eval.py: Allegro, DP). The steps are repeated evaluations.

    python sweep.py gpumd --exe GPUMD/src/gpumd --potential nep.txt \
        --work work/nep --out results/nep/sweep
    python sweep.py lammps --exe lmp --exe_args "-k on g 1 -sf kk \
        -pk kokkos newton on neigh half" --pair_file pair.in \
        --copy Si.snapcoeff Si.snapparam --work work/snap --out results/snap/sweep
    python sweep.py python --exe python --exe_args "py_eval.py size --kind \
        allegro --model si.nequip.pt2" --work work/allegro --out results/allegro/sweep
"""
import argparse
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import time

A_SI = 5.43
DIAMOND = [(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0),
           (.25, .25, .25), (.25, .75, .75), (.75, .25, .75), (.75, .75, .25)]
GB = 1e9
OOM_WORDS = ("out of memory", "out-of-memory", "bad_alloc", "failed to allocate",
             "failed to reallocate", "memory allocation failed",
             "cudaerrormemoryallocation", "outofmemoryerror")


def mem_available_kb():
    """MemAvailable of the node."""
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return 0


def vm_hwm_kb(pid):
    """Peak resident memory of a process, 0 once it is gone."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def run_watched(cmd, cwd, timeout):
    """Run cmd in cwd with stdout and stderr in engine.out and engine.err.
    Returns exit code (negative: killed by that signal), wall time, the
    largest drop of MemAvailable and the peak resident memory."""
    time.sleep(1.0)
    base = low = mem_available_kb()
    hwm, timed_out = 0, False
    t0 = time.time()
    with open(os.path.join(cwd, "engine.out"), "w") as fo, \
            open(os.path.join(cwd, "engine.err"), "w") as fe:
        p = subprocess.Popen(cmd, cwd=cwd, stdout=fo, stderr=fe)
        while True:
            rc = p.poll()
            low = min(low, mem_available_kb())
            hwm = max(hwm, vm_hwm_kb(p.pid))
            if rc is not None:
                break
            if time.time() - t0 > timeout:
                p.kill()
                p.wait()
                timed_out = True
                break
            time.sleep(0.2)
    return {"rc": p.returncode, "wall_s": time.time() - t0,
            "mem_used_gb": (base - low) * 1024 / GB,
            "mem_available_before_gb": base * 1024 / GB,
            "rss_peak_gb": hwm * 1024 / GB, "timed_out": timed_out}


def tail(path, n=12, width=300):
    """Last lines of a text file, for the record of a failure."""
    try:
        with open(path, errors="replace") as fh:
            return [x.rstrip()[:width] for x in fh.readlines()[-n:]]
    except OSError:
        return []


def read(path):
    """Whole text file, or an empty string."""
    try:
        with open(path, errors="replace") as fh:
            return fh.read()
    except OSError:
        return ""


def failure_status(r, text):
    """Why a run that did not finish stopped."""
    low = text.lower()
    if r["timed_out"]:
        return "timeout"
    if any(w in low for w in OOM_WORDS):
        return "oom"
    if r["rc"] in (-9, 137):
        return "killed"
    if r["rc"] in (-11, 139, -6, 134, -7, 135):
        return "crash"
    return "error"


def fresh_dir(args, i):
    """An empty folder for one size, with the files named by --copy."""
    work = os.path.join(args.work, f"i{i:04d}")
    shutil.rmtree(work, ignore_errors=True)
    os.makedirs(work)
    for f in args.copy or []:
        shutil.copy(f, work)
    return work


def failed(row, r, work, text):
    """Fill a row for a run that did not finish."""
    row["ok"] = False
    row["status"] = failure_status(r, text)
    row["stderr_tail"] = tail(os.path.join(work, "engine.err"))
    row["stdout_tail"] = tail(os.path.join(work, "engine.out"))
    return row


# ------------------------------------------------------------------- GPUMD

def gpumd_header(args):
    """Cutoffs and neighbor-list sizes that gpumd takes from nep.txt."""
    info = {"potential": os.path.abspath(args.potential), "potential_lines": []}
    with open(args.potential) as fh:
        for line in fh:
            t = line.split()
            if not t or not t[0][0].isalpha():
                break
            info["potential_lines"].append(line.strip())
            if t[0] == "cutoff" and len(t) >= 5:
                info.update(cutoff_radial=float(t[1]), cutoff_angular=float(t[2]),
                            mn_radial=int(t[3]), mn_angular=int(t[4]))
    if "mn_angular" in info:
        # gpumd sizes its lists at 1.25 times these, and indexes the angular
        # list with 32-bit integers
        cap = math.ceil(1.25 * info["mn_angular"])
        info["angular_list_int32_limit_atoms"] = (2 ** 31 - 1) // cap
    return info


def gpumd_run(i, args):
    """One size with gpumd: a static evaluation (time step 0), then NVE."""
    work = fresh_dir(args, i)
    shutil.copy(args.potential, os.path.join(work, "nep.txt"))
    with open(os.path.join(work, "model.xyz"), "w") as fh:
        fh.write("8\n")
        fh.write(f'Lattice="{A_SI} 0 0 0 {A_SI} 0 0 0 {A_SI}" '
                 "Properties=species:S:1:pos:R:3\n")
        for f in DIAMOND:
            fh.write("Si " + " ".join(f"{x * A_SI:.6f}" for x in f) + "\n")
    with open(os.path.join(work, "run.in"), "w") as fh:
        fh.write(f"replicate {i} {i} {i}\n"
                 "potential nep.txt\n"
                 "velocity 300\n"
                 "ensemble nve\ntime_step 0\ndump_thermo 1\nrun 1\n"
                 f"ensemble nve\ntime_step 1\ndump_thermo 1\nrun {args.steps}\n")
    r = run_watched([os.path.abspath(args.exe)] + shlex.split(args.exe_args),
                    work, args.timeout)
    n = 8 * i ** 3
    row = {"i": i, "atoms": n, **r}
    out = read(os.path.join(work, "engine.out"))
    err = read(os.path.join(work, "engine.err"))
    thermo = []
    for line in read(os.path.join(work, "thermo.out")).splitlines():
        t = line.split()
        if len(t) >= 3:
            try:
                thermo.append([float(x) for x in t[:3]])
            except ValueError:
                pass
    speeds = [float(x) for x in re.findall(
        r"Speed of this run\s*=\s*([0-9.eE+-]+)", out)]
    if "Finished running GPUMD" not in out or len(thermo) < 1 + args.steps:
        return failed(row, r, work, out + err)
    row["ok"] = True
    row["status"] = "ok"
    row["e_static_per_atom"] = thermo[0][2] / n
    etot = [(k + u) / n for _, k, u in thermo[1:]]
    row["nve_steps"] = len(etot)
    row["nve_etot_range_meV_atom"] = 1e3 * (max(etot) - min(etot))
    row["temperature_end_K"] = thermo[-1][0]
    if speeds and speeds[-1] > 0:
        row["atom_steps_per_s"] = speeds[-1]
        row["s_per_atom_step"] = 1.0 / speeds[-1]
    return row


# ------------------------------------------------------------------ LAMMPS

def lammps_header(args):
    """The pair lines and the run settings."""
    return {"pair_lines": [x.strip() for x in open(args.pair_file)
                           if x.strip() and not x.lstrip().startswith("#")],
            "copied_files": [os.path.abspath(f) for f in args.copy or []],
            "exe_args": args.exe_args, "skin_A": args.skin,
            "timestep_ps": args.timestep}


def lammps_run(i, args):
    """One size with LAMMPS: run 0 (static), then NVE at 300 K."""
    work = fresh_dir(args, i)
    shutil.copy(args.pair_file, os.path.join(work, "pair.in"))
    with open(os.path.join(work, "in.si"), "w") as fh:
        fh.write(f"""units metal
atom_style atomic
boundary p p p
lattice diamond {A_SI}
region box block 0 {i} 0 {i} 0 {i}
create_box 1 box
create_atoms 1 box
mass 1 28.0855
include pair.in
neighbor {args.skin} bin
thermo_style custom step temp pe ke etotal
thermo_modify norm no format float %.12g
thermo 1
run 0
velocity all create 300.0 4928459 loop geom
fix 1 all nve
timestep {args.timestep}
run {args.steps}
""")
    cmd = [os.path.abspath(args.exe)] + shlex.split(args.exe_args) + ["-in", "in.si"]
    r = run_watched(cmd, work, args.timeout)
    n = 8 * i ** 3
    row = {"i": i, "atoms": n, **r}
    out = read(os.path.join(work, "engine.out"))
    err = read(os.path.join(work, "engine.err"))
    blocks, cur = [], None
    for line in out.splitlines():
        t = line.split()
        if t[:2] == ["Step", "Temp"]:
            cur = []
            blocks.append(cur)
        elif line.startswith("Loop time of"):
            cur = None
        elif cur is not None and len(t) == 5:
            try:
                cur.append([float(x) for x in t])
            except ValueError:
                pass
    loops = re.findall(r"Loop time of\s+([0-9.eE+-]+)\s+on\s+\d+\s+procs\s+for\s+"
                       r"(\d+)\s+steps\s+with\s+(\d+)\s+atoms", out)
    if len(loops) < 2 or len(blocks) < 2 or not blocks[0] or \
            len(blocks[1]) < 1 + args.steps:
        return failed(row, r, work, out + err)
    if int(loops[-1][2]) != n:
        row["atoms_reported"] = int(loops[-1][2])
        return failed(row, r, work, out + err)
    row["ok"] = True
    row["status"] = "ok"
    row["e_static_per_atom"] = blocks[0][0][2] / n
    etot = [b[4] / n for b in blocks[1]]
    row["nve_steps"] = len(etot) - 1
    row["nve_etot_range_meV_atom"] = 1e3 * (max(etot) - min(etot))
    row["temperature_end_K"] = blocks[1][-1][1]
    t_loop, steps = float(loops[-1][0]), int(loops[-1][1])
    if t_loop > 0 and steps > 0:
        row["atom_steps_per_s"] = steps * n / t_loop
        row["s_per_atom_step"] = t_loop / (steps * n)
    return row


# ------------------------------------------------------------------ python

def python_header(args):
    """The command line of the evaluation script."""
    return {"exe_args": args.exe_args}


def python_run(i, args):
    """One size through a script that prints a RESULT line (py_eval.py)."""
    work = fresh_dir(args, i)
    cmd = [os.path.abspath(args.exe)] + shlex.split(args.exe_args) + \
        ["--i", str(i), "--steps", str(args.steps)]
    r = run_watched(cmd, work, args.timeout)
    n = 8 * i ** 3
    row = {"i": i, "atoms": n, **r}
    out = read(os.path.join(work, "engine.out"))
    err = read(os.path.join(work, "engine.err"))
    res = None
    for line in out.splitlines():
        if line.startswith("RESULT "):
            try:
                res = json.loads(line[7:])
            except ValueError:
                res = None
    if r["rc"] != 0 or not res or res.get("atoms") != n \
            or "e_static_per_atom" not in res:
        return failed(row, r, work, out + err)
    row.update(res)
    row["ok"] = True
    row["status"] = "ok"
    return row


ENGINES = {"gpumd": (gpumd_run, gpumd_header),
           "lammps": (lammps_run, lammps_header),
           "python": (python_run, python_header)}


# ------------------------------------------------------------------- sweep

def signature(args):
    """Fingerprint of what is swept: the engine, its arguments and binary, and
    the content of every file the arguments name. An earlier result file is
    continued only if its fingerprint is the same, because sizes measured with
    another model have other energies and another memory use."""
    h = hashlib.md5()
    st = os.stat(args.exe)
    for x in (args.engine, args.exe_args, args.steps, args.skin, args.timestep,
              st.st_size, st.st_mtime_ns):
        h.update(f"{x}|".encode())
    files = [f for f in (args.potential, args.pair_file) if f] + list(args.copy)
    files += [t for t in shlex.split(args.exe_args) if os.path.isfile(t)]
    for f in files:
        h.update(os.path.basename(f).encode())
        with open(f, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                h.update(block)
    return h.hexdigest()


def show(row):
    """One line per size."""
    s = (f"i {row['i']:4d}  atoms {row['atoms']:>13,}  {row['status']:<12} "
         f"mem {row['mem_used_gb']:6.1f} GB  rss {row['rss_peak_gb']:6.1f} GB  "
         f"wall {row['wall_s']:7.1f} s")
    if row.get("e_static_per_atom") is not None:
        s += f"  E {row['e_static_per_atom']:.6f} eV/atom"
    if row.get("atom_steps_per_s"):
        s += f"  {row['atom_steps_per_s']:.3e} atom-steps/s"
    print(s, flush=True)


def main():
    """CLI entry."""
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("engine", choices=sorted(ENGINES))
    p.add_argument("--exe", required=True)
    p.add_argument("--exe_args", default="", help="arguments of the engine, one string")
    p.add_argument("--potential", help="gpumd: nep.txt")
    p.add_argument("--pair_file", help="lammps: file with pair_style and pair_coeff")
    p.add_argument("--copy", nargs="*", default=[], help="files copied next to the input")
    p.add_argument("--skin", type=float, default=2.0, help="lammps: neighbor skin, A")
    p.add_argument("--timestep", type=float, default=0.001, help="lammps: ps")
    p.add_argument("--work", required=True, help="scratch folder, one subfolder per size")
    p.add_argument("--out", required=True, help="result path without .json")
    p.add_argument("--steps", type=int, default=10, help="steps after the static run")
    p.add_argument("--smoke", type=int, default=4, help="i of the first, small test")
    p.add_argument("--start", type=int, default=20)
    p.add_argument("--growth", type=float, default=2.0, help="factor in atoms per step up")
    p.add_argument("--tol", type=float, default=0.02, help="stop bisecting at this fraction of i")
    p.add_argument("--max_i", type=int, default=600)
    p.add_argument("--e_tol", type=float, default=1e-4,
                   help="eV/atom allowed between a size and the smallest cell")
    p.add_argument("--timeout", type=float, default=3600.0, help="seconds per size")
    p.add_argument("--fresh", action="store_true", help="ignore an earlier result file")
    p.add_argument("--note", default="", help="free text stored in the result")
    args = p.parse_args()
    if args.engine == "gpumd" and not args.potential:
        p.error("gpumd needs --potential")
    if args.engine == "lammps" and not args.pair_file:
        p.error("lammps needs --pair_file")
    args.copy = [os.path.abspath(f) for f in args.copy]
    for name in ("potential", "pair_file"):
        if getattr(args, name):
            setattr(args, name, os.path.abspath(getattr(args, name)))

    run_size, header = ENGINES[args.engine]
    out = args.out + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)
    os.makedirs(args.work, exist_ok=True)
    sig = signature(args)
    state = {"rows": [], "attempting": None, "result": None, "signature": sig}
    old = None
    if os.path.exists(out) and not args.fresh:
        old = json.load(open(out))
        if old.get("signature") != sig:
            print(f"{out} is from another model, engine build or settings: "
                  "starting again", flush=True)
            os.replace(out, out[:-5] + f".old_{int(time.time())}.json")
            old = None
    if old is not None:
        # errors and timeouts may have been transient, so they are run again
        state["rows"] = [r for r in old.get("rows", [])
                         if r.get("status") not in ("error", "timeout")]
        # a size during which the job itself ended (time limit, or the node
        # ran out of memory) is tried once more before it counts as failed
        state["died"] = old.get("died", {})
        if old.get("attempting") is not None:
            i = old["attempting"]
            state["died"][str(i)] = state["died"].get(str(i), 0) + 1
            if state["died"][str(i)] >= 2:
                state["rows"].append({
                    "i": i, "atoms": 8 * i ** 3, "ok": False, "status": "job_died",
                    "rc": None, "wall_s": 0.0, "mem_used_gb": 0.0,
                    "rss_peak_gb": 0.0, "timed_out": False})
            else:
                print(f"the job ended during size i {i}: trying it once more",
                      flush=True)
        print(f"continuing from {out}: {len(state['rows'])} sizes kept", flush=True)
    state.update(engine=args.engine, exe=os.path.abspath(args.exe),
                 engine_info=header(args), a_si=A_SI,
                 steps=args.steps, e_tol=args.e_tol, note=args.note,
                 host=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID"))
    with open("/proc/meminfo") as fh:
        state["mem_total_gb"] = int(fh.readline().split()[1]) * 1024 / GB
    print("engine:", json.dumps(state["engine_info"]), flush=True)

    def save():
        tmp = out + ".tmp"
        with open(tmp, "w") as fh:
            json.dump(state, fh, indent=1)
        os.replace(tmp, out)

    done = {r["i"]: r for r in state["rows"]}

    def get(i):
        if i in done:
            show(done[i])
            return done[i]
        state["attempting"] = i
        save()
        row = run_size(i, args)
        ref = state.get("e_ref_per_atom")
        # written so that a NaN energy also fails
        if row["ok"] and ref is not None and \
                not abs(row["e_static_per_atom"] - ref) <= args.e_tol:
            row["ok"] = False
            row["status"] = "wrong_energy"
        state["rows"].append(row)
        done[i] = row
        state["attempting"] = None
        save()
        show(row)
        return row

    first = get(args.smoke)
    if not first["ok"]:
        print("the small test failed, nothing else was run:")
        for k in ("stdout_tail", "stderr_tail"):
            print(k)
            for line in first.get(k, []):
                print("   ", line)
        state["result"] = {"failed_at_smoke_test": first["status"]}
        save()
        return 1
    state["e_ref_per_atom"] = first["e_static_per_atom"]

    lo, hi, i = args.smoke, None, max(args.start, args.smoke + 1)
    while hi is None:
        if get(i)["ok"]:
            lo = i
            nxt = max(i + 1, round(i * args.growth ** (1.0 / 3.0)))
            if nxt > args.max_i:
                break
            i = nxt
        else:
            hi = i
    while hi is not None and hi - lo > max(1, int(args.tol * lo)):
        mid = (lo + hi) // 2
        if get(mid)["ok"]:
            lo = mid
        else:
            hi = mid

    best = done[lo]
    state["result"] = {
        "largest_ok_i": lo, "largest_ok_atoms": best["atoms"],
        "largest_ok_mem_used_gb": best["mem_used_gb"],
        "largest_ok_kb_per_atom": 1e6 * best["mem_used_gb"] / best["atoms"],
        "first_failed_i": hi,
        "first_failed_atoms": None if hi is None else 8 * hi ** 3,
        "first_failed_status": None if hi is None else done[hi]["status"],
        "reached_max_i_without_failure": hi is None}
    # gpumd reports a list too long for 32-bit sizes as "out of memory" too
    limit = state["engine_info"].get("angular_list_int32_limit_atoms")
    if limit is not None and hi is not None:
        state["result"]["int32_limit_atoms"] = limit
        state["result"]["first_failed_is_above_int32_limit"] = 8 * hi ** 3 > limit
    save()
    print("\nresult:", json.dumps(state["result"]))
    if hi is not None:
        f = done[hi]
        print(f"first failing size: i {hi}, {f['atoms']:,} atoms, {f['status']}")
        for k in ("stderr_tail", "stdout_tail"):
            for line in f.get(k, [])[-6:]:
                print("   ", line)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
