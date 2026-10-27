# protein_compare.py -- Compare Protein GP-fitting variants on a real env, end to end.
#
# For each variant and each of K sweep seeds, runs a full sweep (paired: seed s uses the same
# Protein seed and training seed for every variant). Then, at common compute budgets (fractions
# of the smallest total sweep compute), takes each sweep's Pareto set of runs completed within
# the budget and re-trains those configs with fresh seeds to estimate their true score and cost.
# Reports frontier regret from the re-evaluated scores (and from observed scores) against the
# pooled front of every evaluated config, with paired differences against the first variant.
#
#   regret = mean over a log cost grid of (pooled_front(c) - sweep_front(c)) / pooled_best
#
# where front(c) is the best (mean re-evaluated) score of a Pareto config costing <= c.
# Lower is better; 0 means the sweep's front matches the best front any sweep found.
# Also reports hypervolume (exact area under the front over log cost, absolute rather than
# relative to the pool), the mean of each metric over 20 budgets (AUC), and paired win counts.
#
# Usage (repo root, or inside the puffertank image with the repo at /puffertank/pufferlib):
#   python3 tests/protein_compare.py ENV [--sweeps 3] [--runs N] [--cap SECONDS]
#       [--eval-seeds 3] [--eval-max 12] [--budgets 0.25,0.5,1] [--out DIR] [--no-build]
#       [--variants 'adam:gp_fisher_steps=0,gp_noise_floor=0 fcl:gp_fisher_steps=3,...']
#       [--extra='--sweep.train.ent_coef.max=0.5 ...']
#   --runs / --cap override the ini's sweep.max_runs / max_suggestion_cost (breakout.ini's 3 s cap
#   is below a default run, so pass --cap 30 there)
# Resumable: finished sweeps (DONE marker) and evaluations (eval.jsonl) are skipped on rerun.
# Parallel: --build-only, then --task I --no-build per sweep (tests/protein_compare_array.sh),
# then a plain --no-build run does the evals and summary.
import argparse, glob, json, math, os, shutil, subprocess, sys, time

FISHER = 'gp_fisher_steps=3,gp_fisher_hutch=1,gp_noise_floor=0.01,gp_lin_sigma=1'
DEFAULT_VARIANTS = (f'adam:gp_fisher_steps=0,gp_noise_floor=0 fcl:{FISHER} '
    f'fcl3:{FISHER},gp_lin_median=0.3')
SKIP_SECTIONS = ('base', 'metrics')


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


def read_ini(path):
    secs, sec = {}, None
    for line in open(path):
        line = line.strip()
        if line.startswith('[') and line.endswith(']'):
            sec = line[1:-1]
            secs[sec] = {}
        elif sec is not None and ' = ' in line and not line.startswith('#'):
            k, v = line.split(' = ', 1)
            secs[sec][k] = v
    return secs


def last(v):
    return float(v.split(',')[-1])


def run_result(path, metric):
    m = read_ini(path).get('metrics', {})
    if f'env/{metric}' not in m or 'uptime' not in m:
        return None
    return last(m[f'env/{metric}']), last(m['uptime'])


def config_args(secs):
    args = []
    for sec, kv in secs.items():
        if sec in SKIP_SECTIONS or sec.startswith('sweep'):
            continue
        args += [f'--{sec}.{k}={v}' for k, v in kv.items()]
    return args


def pareto(runs):
    # runs: (score, cost, key); keep runs no other run beats on both
    out, best = [], -math.inf
    for s, c, k in sorted(runs, key=lambda r: (r[1], -r[0])):
        if s > best:
            out.append((s, c, k))
            best = s
    return out


def front(points, c):
    v = [s for s, cc in points if cc <= c]
    return max(v) if v else 0.0


def regret(points, ref, grid, ref_best):
    return sum(front(ref, c) - front(points, c) for c in grid) / len(grid) / ref_best


def hypervolume(points, lo, hi, best):
    # Exact area under the front over log cost in [lo, hi] (score floor 0), normalized by
    # best * log(hi / lo). Absolute, so unlike regret it doesn't shift when variants are added.
    area, cur, prev, end = 0.0, 0.0, math.log(lo), math.log(hi)
    for s, c in sorted((p for p in points if p[1] > 0), key=lambda p: p[1]):
        x = min(max(math.log(c), prev), end)
        area += cur * (x - prev)
        prev, cur = x, max(cur, s)
    area += cur * (end - prev)
    return area / (best * (end - math.log(lo)))


def mean_se(xs):
    if not xs:
        return math.nan, math.nan
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, math.nan
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1) / len(xs))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('env')
    p.add_argument('--sweeps', type=int, default=3)
    p.add_argument('--runs', type=int, default=None, help='sweep.max_runs (default: from the ini)')
    p.add_argument('--cap', type=float, default=None, help='sweep.max_suggestion_cost (s)')
    p.add_argument('--eval-seeds', type=int, default=3)
    p.add_argument('--eval-max', type=int, default=12, help='max Pareto configs per sweep')
    p.add_argument('--budgets', default='0.25,0.5,1')
    p.add_argument('--variants', default=DEFAULT_VARIANTS)
    p.add_argument('--extra', default='', help='extra args for every sweep')
    p.add_argument('--out', default=None)
    p.add_argument('--no-build', action='store_true')
    p.add_argument('--build-only', action='store_true')
    p.add_argument('--task', type=int, default=None,
        help='run only sweep TASK (seed TASK // variants, variant TASK % variants) and exit')
    p.add_argument('--num-tasks', action='store_true', help='print sweeps x variants and exit')
    a = p.parse_args()
    os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    out = os.path.abspath(a.out or f'protein_compare/{a.env}')
    os.makedirs(out, exist_ok=True)
    binary = f'./puffer_compare_{a.env}'
    variants = []
    for spec in a.variants.split():
        name, kv = spec.split(':', 1)
        variants.append((name, [f'--sweep.{x}' for x in kv.split(',') if x]))
    if a.num_tasks:
        print(a.sweeps * len(variants))
        return

    if not a.no_build:
        cu = 'gpu_env = 1' in open(f'config/{a.env}.ini').read()
        log(f'building {binary}')
        subprocess.run(['bash', 'build.sh', a.env, binary[2:]] + (['--cu'] if cu else []),
            check=True)
    if a.build_only:
        return
    overrides = [f'--sweep.max_suggestion_cost={a.cap}'] if a.cap else []
    overrides += [f'--sweep.max_runs={a.runs}'] if a.runs else []

    # Sweeps
    for i in range(a.sweeps * len(variants)):
        s, (name, vargs) = i // len(variants), variants[i % len(variants)]
        d = f'{out}/sweeps/{name}_s{s}'
        if (a.task is not None and a.task != i) or os.path.exists(f'{d}/DONE'):
            continue
        shutil.rmtree(d, ignore_errors=True)
        os.makedirs(d)
        cmd = [binary, 'sweep', f'--sweep.seed={73 + s}', f'--base.seed={73 + s}',
            f'--base.log_dir={d}/logs',
            f'--base.checkpoint_dir={d}/ckpt'] + overrides + vargs + a.extra.split()
        log(f'sweep {name} seed {s}: {" ".join(cmd)}')
        t0 = time.time()
        with open(f'{d}/sweep.log', 'w') as f:
            subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
        shutil.rmtree(f'{d}/ckpt', ignore_errors=True)
        open(f'{d}/DONE', 'w').write(f'{time.time() - t0:.0f}\n')
        log(f'sweep {name} seed {s} done in {time.time() - t0:.0f} s')
    if a.task is not None:
        return

    # Completed runs per sweep, in completion order
    sweeps = {}
    for s in range(a.sweeps):
        for name, _ in variants:
            runs = []
            for path in glob.glob(f'{out}/sweeps/{name}_s{s}/logs/*/sweep_*.ini'):
                secs = read_ini(path)
                r = run_result(path, secs['sweep']['metric'].strip())
                if r is not None:  # completion time: run-id start (ms) + cost; mtimes don't survive copies
                    start = int(os.path.basename(path).split('_')[1]) / 1000
                    runs.append((start + r[1], r[0], r[1], path))
            sweeps[(name, s)] = [(sc, c, path) for _, sc, c, path in sorted(runs)]
    total = {k: sum(c for _, c, _ in v) for k, v in sweeps.items()}
    budgets = [float(x) * min(total.values()) for x in a.budgets.split(',')]
    curve = [(i + 1) / 20 * min(total.values()) for i in range(20)]  # AUC budgets

    def within(runs, b):
        acc, kept = 0.0, []
        for r in runs:
            acc += r[1]
            if acc > b:
                break
            kept.append(r)
        return kept

    # Pareto configs to re-evaluate: union over report and AUC budgets, thinned evenly by cost to eval_max
    todo = {}
    for k, runs in sweeps.items():
        keys = []
        for b in budgets + curve:
            for r in pareto(within(runs, b)):
                if r[2] not in keys:
                    keys.append(r[2])
        if len(keys) > a.eval_max:
            keys = sorted(keys, key=lambda p: next(c for _, c, q in runs if q == p))
            keys = [keys[round(i * (len(keys) - 1) / (a.eval_max - 1))]
                for i in range(a.eval_max)]
        todo[k] = keys

    evals = {}
    ev_path = f'{out}/eval.jsonl'
    if os.path.exists(ev_path):
        for line in open(ev_path):
            e = json.loads(line)
            evals.setdefault(e['config'], {})[e['seed']] = (e['score'], e['cost'])
    n_eval = sum(len(v) for v in todo.values()) * a.eval_seeds
    # Paired sweeps share their random phase, so evals are cached per config, not per run
    cfg = {}
    for keys in todo.values():
        for path in keys:
            cfg[path] = ' '.join(config_args(read_ini(path)))
    done = 0
    for k, keys in todo.items():
        for path in keys:
            secs = read_ini(path)
            metric = secs['sweep']['metric'].strip()
            for e in range(a.eval_seeds):
                done += 1
                seed = 1000 + e
                if seed in evals.get(cfg[path], {}):
                    continue
                d = f'{out}/eval/{len(evals)}_{seed}'
                shutil.rmtree(d, ignore_errors=True)
                cmd = [binary, 'train', f'--base.seed={seed}', f'--base.log_dir={d}/logs',
                    f'--base.checkpoint_dir={d}/ckpt'] + config_args(secs)
                subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                res = [run_result(q, metric) for q in glob.glob(f'{d}/logs/*/*.ini')]
                res = [r for r in res if r is not None]
                score, cost = res[0] if res else (0.0, math.nan)
                evals.setdefault(cfg[path], {})[seed] = (score, cost)
                with open(ev_path, 'a') as f:
                    f.write(json.dumps({'config': cfg[path], 'seed': seed, 'score': score,
                        'cost': cost}) + '\n')
                shutil.rmtree(d, ignore_errors=True)
                log(f'eval {done}/{n_eval}: {os.path.basename(path)} seed {seed} -> '
                    f'{score:.4g} in {cost:.1f} s')

    # True (re-evaluated) points: mean score, mean cost (falls back to the sweep's cost)
    def true_point(path, obs_cost):
        v = [x for x in evals.get(cfg[path], {}).values() if not math.isnan(x[1])]
        if not v:
            return None
        sc = sum(x[0] for x in v) / len(v)
        c = sum(x[1] for x in v) / len(v)
        return sc, c if c > 0 else obs_cost

    obs_pts, true_pts = {}, {}
    for k, runs in sweeps.items():
        for b in budgets + curve:
            kept = within(runs, b)
            obs_pts[(k, b)] = [(s, c) for s, c, _ in kept]
            tp = [true_point(q, c) for s, c, q in pareto(kept) if q in todo[k]]
            true_pts[(k, b)] = [t for t in tp if t is not None]

    def refs(pts):
        allp = [x for v in pts.values() for x in v]
        costs = [c for _, c in allp if c > 0]
        lo, hi = min(costs), max(costs)
        grid = [math.exp(math.log(lo) + i * (math.log(hi) - math.log(lo)) / 31) for i in range(32)]
        return allp, grid, max(s for s, _ in allp), lo, hi

    lines = [f'env {a.env}: {a.sweeps} paired sweeps x {len(variants)} variants, {a.runs or "ini"} runs, cap {a.cap or "ini"}, '
        f'budgets {", ".join(f"{b:.0f}" for b in budgets)} s of sweep compute, '
        f'{a.eval_seeds} eval seeds per Pareto config',
        f'cells: mean+-se over sweeps (d: paired diff vs {variants[0][0]}, w: paired wins vs '
        f'{variants[0][0]}, ties 0.5); AUC: mean over 20 budgets up to {budgets[-1]:.0f} s']
    cols = [(f'@{b:.0f}s', [b]) for b in budgets] + [('AUC', curve)]
    metrics = (('regret (lower is better)', -1, lambda p, R: regret(p, R[0], R[1], R[2])),
        ('hypervolume (higher is better)', 1, lambda p, R: hypervolume(p, R[3], R[4], R[2])))
    for mname, sign, fn in metrics:
        for label, pts in (('true (re-evaluated)', true_pts), ('observed', obs_pts)):
            R = refs(pts)
            lines.append(f'\n{mname}, {label} scores')
            lines.append(f'{"variant":12s}' + ''.join(f'  {c:>38s}' for c, _ in cols))
            for name, _ in variants:
                row = f'{name:12s}'
                for _, bs in cols:
                    r, r0 = ([sum(fn(pts[((v, s), b)], R) for b in bs) / len(bs)
                        for s in range(a.sweeps)] for v in (name, variants[0][0]))
                    m, se = mean_se(r)
                    cell = f'{m:+.3f}+-{se:.3f}'
                    if name != variants[0][0]:
                        dm, dse = mean_se([x - y for x, y in zip(r, r0)])
                        w = sum(1.0 if (x - y) * sign > 0 else 0.5 if x == y else 0.0
                            for x, y in zip(r, r0))
                        cell += f' (d {dm:+.3f}+-{dse:.3f}, w {w:g}/{a.sweeps})'
                    row += f'  {cell:>38s}'
                lines.append(row)
    lines.append(f'\n{"sweep":14s}{"runs":>6s}{"compute":>10s}{"best obs":>10s}')
    for (name, s), runs in sweeps.items():
        lines.append(f'{f"{name}_s{s}":14s}{len(runs):6d}{total[(name, s)]:10.0f}'
            f'{max(sc for sc, _, _ in runs):10.4g}')
    text = '\n'.join(lines)
    print(text)
    open(f'{out}/summary.txt', 'w').write(text + '\n')


if __name__ == '__main__':
    main()
