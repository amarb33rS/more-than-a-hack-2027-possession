"""Bounded next experiments, gated by completed baselines and paired results."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
from .common import write_json
from .download import sha256
from .queue_status import collect_status, read_json
from .run_lock import run_lock, AlreadyRunningError
from .development_run import validation_summary, analyze


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text())


def completed(path):
    value, _ = read_json(path)
    return isinstance(value, dict) and value.get('status') == 'completed'


def gate(baseline, candidate, baseline_transfer, candidate_transfer, delta=.02, clean_limit=.02):
    def clean(summary):
        rows = [v['spell_f1'] for k, v in summary['matches'].items() if k.endswith('/clean')]
        return sum(rows)/len(rows)
    def transfer(report):
        rows = [v['models']['neural']['decoder']['3']['spell_f1_mean']
                for k, v in report['matches'].items() if k.startswith('target_ball_gaps/')]
        return sum(rows)/len(rows)
    score_gain = candidate['score']-baseline['score']
    transfer_gain = transfer(candidate_transfer)-transfer(baseline_transfer)
    clean_gain = clean(candidate)-clean(baseline)
    promising = (score_gain+1e-12 >= delta or transfer_gain+1e-12 >= delta) and clean_gain >= -clean_limit-1e-12
    return dict(promising=promising, validation_gain=score_gain, transfer_gain=transfer_gain,
                clean_gain=clean_gain, min_gain=delta, max_clean_regression=clean_limit,
                interpretation='Operational budget gate using seed 42, not statistical significance or final model selection')


def command(root, stage, args, log):
    write_json(root/'status.json', dict(state='running', stage=stage, updated_at_utc=now()))
    log.parent.mkdir(parents=True, exist_ok=True)
    print('STAGE', stage, *map(str, args), flush=True)
    with open(log, 'a', encoding='utf-8') as stream:
        stream.write('\nCOMMAND '+json.dumps(list(map(str, args)))+'\n'); stream.flush()
        child = subprocess.Popen([sys.executable, '-u', '-m', *map(str, args)],
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in child.stdout:
            stream.write(line); stream.flush()
            print(line, end='', flush=True)
        code = child.wait()
    if code:
        raise RuntimeError(f'{stage} failed with exit {code}; inspect {log}')


def benchmark(root, source, out=None, tree=None):
    dest = out or source/'transfer-benchmark-seed-42.json'
    existing, _ = read_json(dest)
    if isinstance(existing, dict) and existing.get('status') == 'completed':
        return
    args = ['modeling.transfer_benchmark', '--root', source, '--seed', 42]
    if out:
        args += ['--out', out]
    if tree:
        args += ['--tree-model', tree]
    command(root, 'transfer_'+source.name, args, dest.with_suffix('.log'))


def main(a):
    root, source = a.out, Path('model-runs/development-mixed-v1')
    with run_lock(root):
        plan = dict(source=str(source), pilot_seed=42, confirmation_seeds=[43, 44],
            architecture='set-tcn-motion-v3', sampling='mixed', max_epochs=100,
            check_every=10, patience=3, gate_min_gain=.02, gate_max_clean_regression=.02,
            tree_experiment='50/50 standard-vs-target-gap noisy rows at matched total row budget',
            scope='Development-only experiments; no holdout evaluation, target scoring, submission or video training')
        if (root/'plan.json').exists() and read(root/'plan.json') != plan:
            raise ValueError('Next-experiment plan differs; use a fresh output directory')
        write_json(root/'plan.json', plan)
        if completed(root/'complete.json'):
            print('Next-experiment suite already complete', flush=True)
            return
        # Follow durable artifacts and verified processes, not one particular PID.
        while True:
            status = collect_status(Path('.'), include_improvements=False)
            if status['complete']:
                break
            write_json(root/'status.json', dict(state='waiting', stage='baseline_and_mixed_queue',
                updated_at_utc=now(), stages=status['stages']))
            live = any(s['process']['state'] in ('alive', 'alive_unverified', 'unverified') for s in status['stages'])
            if not live and any(s['state'] in ('failed', 'interrupted_or_not_running', 'inconsistent') for s in status['stages']):
                raise RuntimeError('Upstream queue stopped or inconsistent; inspect queue_status and upstream logs, then resume this command')
            print('Waiting for baseline/mixed suite', flush=True)
            time.sleep(30)
        snapshot = root/'source-snapshot'
        if not (root/'environment.json').exists():
            snapshot.mkdir(exist_ok=True)
            files = list(Path('modeling').glob('*.py'))+[Path('modeling/splits.json')]
            for path in files:
                shutil.copy2(path, snapshot/path.name)
            write_json(root/'environment.json', dict(python=platform.python_version(),
                captured_at_utc=now(), sources={p.name: sha256(p) for p in files}))
        # Run the cheap tree adaptation before spending another neural-run budget.
        tree_folder = root/'target_gap_tree'
        split = read('modeling/splits.json')
        if not (tree_folder/'report.json').exists():
            command(root, 'target_gap_tree', ['modeling.train_transfer_tree', '--train', *split['train'],
                '--validation', *split['validation'], '--out', tree_folder], tree_folder/'run.log')
        benchmark(root, source, out=tree_folder/'transfer-benchmark.json', tree=tree_folder/'tree.pkl')
        if not (tree_folder/'error-analysis.json').exists():
            analyze(tree_folder, split['validation'])
        write_json(root/'tree-result.json', validation_summary(tree_folder, split['validation']))
        pilot = root/'motion-pilot'
        command(root, 'motion_pilot', ['modeling.development_run', '--out', pilot,
            '--sampling', 'mixed', '--architecture', 'set-tcn-motion-v3', '--seeds', 42,
            '--tree-from', source/'tree'], pilot/'run.log')
        benchmark(root, pilot)
        base = read(source/'summary.json')['neural']['42']
        candidate = read(pilot/'summary.json')['neural']['42']
        decision = gate(base, candidate, read(source/'transfer-benchmark-seed-42.json'),
                        read(pilot/'transfer-benchmark-seed-42.json'))
        write_json(root/'motion-gate.json', decision)
        if decision['promising']:
            confirmation = root/'motion-confirmation'
            command(root, 'motion_confirmation', ['modeling.development_run', '--out', confirmation,
                '--sampling', 'mixed', '--architecture', 'set-tcn-motion-v3', '--seeds', 43, 44,
                '--tree-from', source/'tree'], confirmation/'run.log')
            if not completed(confirmation/'reliability/complete.json'):
                command(root, 'motion_confirmation_reliability', ['modeling.reliability', '--root', confirmation], confirmation/'reliability.log')
        if not completed(pilot/'reliability/complete.json'):
            command(root, 'motion_pilot_reliability', ['modeling.reliability', '--root', pilot], pilot/'reliability.log')
        write_json(root/'summary.json', dict(motion_gate=decision,
            baseline=read(source/'summary.json'), motion_pilot=read(pilot/'summary.json'),
            motion_confirmation=read(root/'motion-confirmation/summary.json') if decision['promising'] else None,
            adapted_tree=validation_summary(tree_folder, split['validation']),
            interpretation='Paired development comparisons; freeze a candidate before any final holdout evaluation'))
        write_json(root/'complete.json', dict(status='completed', motion_confirmation_run=decision['promising']))
        write_json(root/'status.json', dict(state='completed', updated_at_utc=now()))
        print('Bounded improvement suite complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=Path('model-runs/improvements-v1'))
    args = parser.parse_args()
    try:
        main(args)
    except Exception as error:
        if not isinstance(error, AlreadyRunningError):
            write_json(args.out/'status.json', dict(state='failed', updated_at_utc=now(), error=str(error)))
        raise
