"""Read-only experiment queue status; no torch loads or training side effects.

Usage: python -m modeling.queue_status [--json] [--stale-minutes 30]
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import time


def read_json(path):
    """A writer may be replacing this file; incomplete JSON is not a fatal error."""
    try:
        data = json.loads(path.read_text(encoding='utf-8-sig'))
        return data, None
    except FileNotFoundError:
        return None, 'missing'
    except (OSError, ValueError, UnicodeError) as error:
        return None, f'unreadable_or_partial: {type(error).__name__}'


def _nonempty(path):
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _tail(path, count=32768):
    try:
        with path.open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-count))
            return stream.read().decode('utf-8', errors='replace')
    except OSError:
        return ''


def _age(path, now):
    try:
        return max(0, now-path.stat().st_mtime)
    except OSError:
        return None


def _pid(path):
    try:
        value = int(path.read_text().strip())
        return value if value > 0 else None
    except (OSError, ValueError):
        return None


def query_processes(pids):
    """Query only recorded PIDs, keeping command lines internal to identity checks."""
    pids = sorted(set(p for p in pids if p is not None))
    if not pids:
        return {}, None
    if os.name != 'nt':
        return {}, 'PID identity verification requires Windows; supply process_info when testing'
    expression = ' OR '.join(f'ProcessId = {int(pid)}' for pid in pids)
    command = (
        f'Get-CimInstance Win32_Process -Filter "{expression}" | '
        'ForEach-Object { [pscustomobject]@{pid=$_.ProcessId; command=$_.CommandLine; '
        "created=$_.CreationDate.ToUniversalTime().ToString('o')} } | ConvertTo-Json -Compress"
    )
    try:
        result = subprocess.run(['powershell.exe', '-NoProfile', '-Command', command],
                                capture_output=True, text=True, timeout=15)
        if result.returncode:
            return {}, 'Windows process query failed; liveness is unverified'
        rows = json.loads(result.stdout) if result.stdout.strip() else []
        rows = [rows] if isinstance(rows, dict) else rows
        info = {}
        for row in rows:
            created = row.get('created')
            info[int(row['pid'])] = dict(command=row.get('command'),
                created=datetime.fromisoformat(created.replace('Z', '+00:00')).timestamp() if created else None)
        return info, None
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return {}, 'Windows process query unavailable; liveness is unverified'


def _process(pid_path, expected, process_info, query_error):
    pid = _pid(pid_path)
    result = dict(pid=pid, state='not_recorded', identity_verified=False)
    if pid is None:
        return result
    if query_error:
        return {**result, 'state': 'unverified'}
    info = process_info.get(pid)
    if info is None:
        return {**result, 'state': 'exited'}
    command = info.get('command')
    if not command:
        return {**result, 'state': 'alive_unverified'}
    if expected.lower() not in command.lower():
        return {**result, 'state': 'identity_mismatch'}
    # A process created after its PID record cannot be the recorded job, even if
    # an unrelated reused PID happens to have a similar executable/command.
    created = info.get('created')
    try:
        if created is not None and created > pid_path.stat().st_mtime+5:
            return {**result, 'state': 'identity_mismatch'}
    except OSError:
        return {**result, 'state': 'unverified'}
    return {**result, 'state': 'alive', 'identity_verified': True}


def _fatal_log(path):
    # Warnings alone do not mean a run failed. Do not echo a full traceback.
    text = _tail(path)
    patterns = [r'Traceback \(most recent call last\)',
                r'(?m)^(?:\w+\.)?(?:RuntimeError|AssertionError|FileNotFoundError|ValueError|CalledProcessError):',
                r'(?m)^\s*(?:FullyQualifiedErrorId|\+.*throw)\s*[: ]']
    return any(re.search(pattern, text) for pattern in patterns)


def _log_epochs(path):
    current, epochs = None, {}
    text = _tail(path, count=262144)
    for line in text.splitlines():
        if line.startswith(('RUN modeling.train_neural ', 'RUN modeling.train_control_neural ')):
            match = re.search(r'neural-seed-(\d+)', line)
            current = match.group(1) if match else None
        elif current:
            match = re.match(r'Epoch (\d+):', line)
            if match:
                epochs[current] = int(match.group(1))
    return epochs


def _seeds(root, plan, issues):
    expected = plan.get('seeds', [42, 43, 44]) if isinstance(plan, dict) else [42, 43, 44]
    max_epochs = plan.get('max_epochs') if isinstance(plan, dict) else None
    patience = plan.get('patience') if isinstance(plan, dict) else None
    logged = _log_epochs(root/'run.log')
    rows = []
    for seed in expected:
        folder = root/f'neural-seed-{seed}'
        data, error = read_json(folder/'selection.json')
        row = dict(seed=seed, state='pending', current_epoch=logged.get(str(seed)),
                   last_validation_epoch=None, best_epoch=None, best_score=None, stop_reason=None)
        if error == 'missing':
            if row['current_epoch'] is not None:
                row['state'] = 'training_before_first_validation'
        elif error or not isinstance(data, dict):
            row['state'] = 'selection_unreadable'
            issues.append(f'{root.name}/seed-{seed}: selection JSON is incomplete or unreadable')
        else:
            checks = data.get('checks', [])
            last = checks[-1].get('epoch') if checks and isinstance(checks[-1], dict) else None
            row.update(last_validation_epoch=last, best_epoch=data.get('best_epoch'),
                       best_score=data.get('best_score'), stale_checks=data.get('stale'))
            row['current_epoch'] = max([v for v in (row['current_epoch'], last) if isinstance(v, int)], default=None)
            if data.get('complete') is True:
                missing = [name for name in ('last.pt', 'report.json') if not _nonempty(folder/'best'/name)]
                if missing or not checks:
                    row['state'] = 'inconsistent'
                    issues.append(f'{root.name}/seed-{seed}: marked complete but selected artifacts/checks are missing')
                else:
                    row['state'] = 'completed'
                    if max_epochs is not None and last is not None and last >= max_epochs:
                        row['stop_reason'] = 'epoch_ceiling'
                    elif patience is not None and data.get('stale', 0) >= patience:
                        row['stop_reason'] = 'early_stopping'
                    else:
                        row['stop_reason'] = 'completion_reason_unverified'
            else:
                row['state'] = 'in_progress'
        rows.append(row)
    return rows


def _stage(name, marker_path, pid_path, expected_command, log, errors, dependencies,
           process_info, query_error, now, stale_seconds, artifacts=(), transfer=False):
    marker, marker_error = read_json(marker_path)
    complete = isinstance(marker, dict) and marker.get('status') == 'completed'
    process = _process(pid_path, expected_command, process_info, query_error)
    log_age = _age(log, now) if _nonempty(log) else None
    result = dict(name=name, state='pending', process=process,
                  last_log_age_seconds=round(log_age, 1) if log_age is not None else None,
                  completion_marker=str(marker_path), notes=[])
    if marker_error not in (None, 'missing'):
        result['notes'].append('Completion JSON is partial or unreadable; completion is unconfirmed')
    elif marker_error is None and not complete and not transfer:
        result['notes'].append('Completion marker has an unexpected status')
    if complete:
        missing = [str(p) for p in artifacts if not _nonempty(p)]
        if missing or not all(dependencies):
            result['state'] = 'inconsistent'
            result['notes'].append('Completion marker conflicts with missing outputs or unfinished dependencies')
        else:
            result['state'] = 'completed'
        return result
    if not all(dependencies):
        if process['state'] == 'alive':
            result['state'] = 'queued'
        elif process['state'] in ('unverified', 'alive_unverified'):
            result['state'] = 'queued_unverified'
        elif process['state'] == 'not_recorded':
            result['state'] = 'pending'
        else:
            result['state'] = 'failed' if _fatal_log(errors) else 'watcher_not_running'
            result['notes'].append('Dependency unfinished and recorded watcher is no longer verified alive')
        return result
    if process['state'] == 'alive':
        result['state'] = 'running'
        if log_age is not None and log_age > stale_seconds:
            result['state'] = 'running_stale'
            result['notes'].append('Process is alive but the stage log has not advanced within the stale threshold')
        if _fatal_log(errors):
            result['notes'].append('Fatal-looking text exists in stderr; process is still alive, inspect log before declaring failure')
    elif process['state'] in ('unverified', 'alive_unverified'):
        result['state'] = 'liveness_unverified'
    elif _fatal_log(errors):
        result['state'] = 'failed'
        result['notes'].append('Fatal-looking stderr and no verified live runner')
    elif process['state'] == 'identity_mismatch':
        result['state'] = 'runner_identity_mismatch'
        result['notes'].append('Recorded PID belongs to a different or newer process')
    elif process['state'] == 'exited' or _nonempty(log):
        result['state'] = 'interrupted_or_not_running'
    return result


def collect_status(workspace=Path('.'), *, now=None, stale_seconds=1800, process_info=None, include_improvements=True):
    workspace = Path(workspace)
    now = time.time() if now is None else now
    base, mixed = (workspace/'model-runs'/name for name in ('development-v2', 'development-mixed-v1'))
    issues = []
    roots = {}
    for root in (base, mixed):
        plan, error = read_json(root/'plan.json')
        if error not in (None, 'missing'):
            issues.append(f'{root.name}: plan is incomplete or unreadable')
        roots[root.name] = dict(seeds=_seeds(root, plan, issues))
    pid_files = [base/'runner.pid', base/'reliability-runner.pid', mixed/'runner.pid', mixed/'transfer-runner.pid']
    improvements = workspace/'model-runs/improvements-v1'
    show_improvements = include_improvements and (improvements/'plan.json').exists()
    robust = workspace/'model-runs/robust-velocity-v1'
    show_robust = show_improvements and ((robust/'runner.pid').exists() or (robust/'status.json').exists())
    robust_motion = workspace/'model-runs/motion-robust-v4'
    show_robust_motion = show_robust and ((robust_motion/'runner.pid').exists() or (robust_motion/'status.json').exists())
    ensemble = workspace/'model-runs/motion-ensemble-v1'
    show_ensemble = include_improvements and ((ensemble/'manifest.json').exists() or (ensemble/'status.json').exists())
    decoder = workspace/'model-runs/decoder-lowcost-v1'
    show_decoder = include_improvements and ((decoder/'plan.json').exists() or (decoder/'status.json').exists())
    confirmation = workspace/'model-runs/motion-robust-confirmation-v1'
    show_confirmation = include_improvements and ((confirmation/'driver-plan.json').exists() or (confirmation/'status.json').exists())
    v4_ensemble = workspace/'model-runs/motion-v4-ensemble-v1'
    show_v4_ensemble = include_improvements and ((v4_ensemble/'protocol.json').exists() or (v4_ensemble/'status.json').exists())
    video_data = workspace/'model-runs/video-ball-pilot-v1/data'
    video_model = workspace/'model-runs/video-ball-pilot-v1/model'
    show_video = include_improvements and ((video_data/'status.json').exists() or (video_model/'status.json').exists())
    control = workspace/'model-runs/control-duration-v1'
    show_control = include_improvements and ((control/'driver-plan.json').exists() or (control/'status.json').exists())
    if show_improvements:
        pid_files.append(improvements/'runner.pid')
    if show_robust:
        pid_files.append(robust/'runner.pid')
    if show_robust_motion:
        pid_files.append(robust_motion/'runner.pid')
        motion_plan, _ = read_json(robust_motion/'plan.json')
        if not isinstance(motion_plan, dict):motion_plan=dict(seeds=[42],max_epochs=100,patience=3)
        roots[robust_motion.name]=dict(seeds=_seeds(robust_motion,motion_plan,issues))
    if show_ensemble:
        pid_files.append(ensemble/'runner.pid')
    if show_decoder:
        pid_files.append(decoder/'runner.pid')
    if show_confirmation:
        pid_files.append(confirmation/'runner.pid')
        confirmation_plan, _ = read_json(confirmation/'plan.json')
        if not isinstance(confirmation_plan, dict):
            confirmation_plan = dict(seeds=[43, 44], max_epochs=100, patience=3)
        roots[confirmation.name] = dict(seeds=_seeds(confirmation, confirmation_plan, issues))
    if show_v4_ensemble:
        pid_files.append(v4_ensemble/'runner.pid')
    if show_video:
        pid_files.extend([video_data/'runner.pid', video_model/'runner.pid'])
    if show_control:
        pid_files.append(control/'runner.pid')
        control_plan, _ = read_json(control/'plan.json')
        if not isinstance(control_plan, dict):
            control_plan = dict(seeds=[42], max_epochs=100, patience=3)
        roots[control.name] = dict(seeds=_seeds(control, control_plan, issues))
    query_error = None
    if process_info is None:
        process_info, query_error = query_processes([_pid(p) for p in pid_files])
    if query_error:
        issues.append(query_error)
    stages = []
    def stage(root, name, marker, pid, command, log, errors, deps, artifacts=(), transfer=False):
        value = _stage(name, root/marker, root/pid, command, root/log, root/errors, deps,
                       process_info, query_error, now, stale_seconds, artifacts, transfer)
        stages.append(value)
        return value['state'] == 'completed'
    base_seeds = all(v['state'] == 'completed' for v in roots[base.name]['seeds'])
    base_training = stage(base, 'baseline_training_and_decoder_analysis', 'complete.json', 'runner.pid',
        'modeling.development_run', 'run.log', 'errors.log', [True],
        [base/'tree'/'tree.pkl', base/'tree'/'report.json'] +
        [base/f"neural-seed-{v['seed']}"/'best'/'error-analysis.json' for v in roots[base.name]['seeds']])
    if base_training and not base_seeds:
        stages[-1]['state'] = 'inconsistent'; base_training = False
        stages[-1]['notes'].append('Completion marker conflicts with unfinished neural seeds')
    base_reliability = stage(base, 'baseline_reliability', 'reliability/complete.json', 'reliability-runner.pid',
        'run-reliability-after-training.ps1', 'reliability.log', 'reliability-errors.log', [base_training],
        [base/'reliability'/'fusion-and-calibration.json', base/'reliability'/'stress.json'])
    mixed_seeds = all(v['state'] == 'completed' for v in roots[mixed.name]['seeds'])
    mixed_training = stage(mixed, 'mixed_training_and_decoder_analysis', 'complete.json', 'runner.pid',
        'run-mixed-after-baseline.ps1', 'run.log', 'errors.log', [base_reliability],
        [mixed/'tree'/'tree.pkl', mixed/'tree'/'report.json']+
        [mixed/f"neural-seed-{v['seed']}"/'best'/'error-analysis.json' for v in roots[mixed.name]['seeds']])
    if mixed_training and not mixed_seeds:
        stages[-1]['state'] = 'inconsistent'; mixed_training = False
        stages[-1]['notes'].append('Completion marker conflicts with unfinished neural seeds')
    mixed_reliability = stage(mixed, 'mixed_reliability', 'reliability/complete.json', 'runner.pid',
        'run-mixed-after-baseline.ps1', 'run.log', 'errors.log', [mixed_training],
        [mixed/'reliability'/'fusion-and-calibration.json', mixed/'reliability'/'stress.json'])
    transfer = stage(mixed, 'mixed_transfer_benchmark', 'transfer-benchmark-seed-42.json', 'transfer-runner.pid',
        'run-transfer-after-mixed.ps1', 'transfer.log', 'transfer-errors.log', [mixed_reliability], transfer=True)
    robust_motion_complete = False
    if show_improvements:
        improvements_complete = stage(improvements, 'adaptive_improvements', 'complete.json', 'runner.pid',
              'modeling.improve_run', 'run.log', 'errors.log', [transfer],
              [improvements/'summary.json'])
        runtime, _ = read_json(improvements/'status.json')
        if isinstance(runtime, dict):
            stages[-1]['notes'].append('Managed stage: '+str(runtime.get('stage', runtime.get('state', 'unknown'))))
        if show_robust:
            robust_complete = stage(robust, 'robust_derivative_tree', 'complete.json', 'runner.pid',
                  'modeling.robust_tree_run', 'run.log', 'errors.log', [improvements_complete],
                  [robust/'summary.json', robust/'paired-standard-noise.json', robust/'paired-transfer.json'])
            if show_robust_motion:
                robust_motion_complete = stage(robust_motion,'robust_motion_pilot','driver-complete.json','runner.pid',
                      'modeling.robust_motion_run','run.log','errors.log',[robust_complete],
                      [robust_motion/'robust-motion-decision.json',robust_motion/'summary.json'])
    ensemble_complete = False
    if show_ensemble:
        ensemble_complete = stage(ensemble, 'three_seed_motion_ensemble', 'complete.json', 'runner.pid',
              'modeling.ensemble_run', 'run.log', 'errors.log', [robust_motion_complete],
              [ensemble/'manifest.json', ensemble/'report.json', ensemble/'summary.json',
               ensemble/'target-observation-masks.json'])
        runtime, _ = read_json(ensemble/'status.json')
        if isinstance(runtime, dict):
            stages[-1]['notes'].append('Managed stage: '+str(runtime.get('stage', runtime.get('state', 'unknown'))))
    decoder_complete = False
    if show_decoder:
        decoder_complete = stage(decoder, 'low_cost_decoder_ablation', 'complete.json', 'runner.pid',
              'modeling.decoder_ablation', 'run.log', 'errors.log', [ensemble_complete],
              [decoder/'plan.json', decoder/'report.json', decoder/'summary.json'])
        runtime, _ = read_json(decoder/'status.json')
        if isinstance(runtime, dict):
            stages[-1]['notes'].append('Managed stage: '+str(runtime.get('stage', runtime.get('state', 'unknown'))))
    confirmation_complete = False
    if show_confirmation:
        confirmation_complete = stage(confirmation, 'robust_motion_confirmation', 'driver-complete.json', 'runner.pid',
              'modeling.confirm_motion_run', 'run.log', 'errors.log',
              [robust_motion_complete, ensemble_complete, decoder_complete],
              [confirmation/'driver-plan.json', confirmation/'summary.json',
               confirmation/'reliability/complete.json', confirmation/'transfer-benchmark-seed-43.json',
               confirmation/'transfer-benchmark-seed-44.json'])
        runtime, _ = read_json(confirmation/'status.json')
        if isinstance(runtime, dict):
            stages[-1]['notes'].append('Managed stage: '+str(runtime.get('stage', runtime.get('state', 'unknown'))))
    v4_ensemble_complete = False
    if show_v4_ensemble:
        v4_ensemble_complete = stage(v4_ensemble, 'robust_motion_three_seed_ensemble', 'complete.json', 'runner.pid',
              'modeling.ensemble_v4_run', 'run.log', 'errors.log', [confirmation_complete, ensemble_complete],
              [v4_ensemble/'manifest.json', v4_ensemble/'report.json', v4_ensemble/'summary.json',
               v4_ensemble/'generated-cache-manifest.json'])
        runtime, _ = read_json(v4_ensemble/'status.json')
        if isinstance(runtime, dict):
            stages[-1]['notes'].append('Managed stage: '+str(runtime.get('stage', runtime.get('state', 'unknown'))))
    if show_video:
        video_prepared = stage(video_data, 'video_ball_data_preparation', 'complete.json', 'runner.pid',
              'modeling.video_ball_data', 'run.log', 'errors.log', [True],
              [video_data/'manifest.json', video_data/'train.npz', video_data/'references.json'])
        stage(video_model, 'weak_video_ball_localization', 'complete.json', 'runner.pid',
              'modeling.train_video_ball', 'run.log', 'errors.log', [video_prepared],
              [video_model/'protocol.json', video_model/'last.pt', video_model/'summary.json'])
        runtime, _ = read_json(video_model/'status.json')
        if isinstance(runtime, dict):
            stages[-1]['notes'].append('Managed stage: '+str(runtime.get('stage', runtime.get('state', 'unknown'))))
    if show_control:
        stage(control, 'control_duration_loss_pilot', 'driver-complete.json', 'runner.pid',
              'modeling.control_duration_run', 'run.log', 'errors.log', [v4_ensemble_complete],
              [control/'driver-plan.json', control/'summary.json', control/'control-loss-decision.json',
               control/'typed-transition-comparison.json', control/'budget-comparison.json'])
        runtime, _ = read_json(control/'status.json')
        if isinstance(runtime, dict):
            stages[-1]['notes'].append('Managed stage: '+str(runtime.get('stage', runtime.get('state', 'unknown'))))
    return dict(checked_at_utc=datetime.fromtimestamp(now, timezone.utc).isoformat(),
                complete=all(v['state'] == 'completed' for v in stages), stages=stages, experiments=roots, issues=issues)


def human_report(report):
    lines = []
    for stage in report['stages']:
        age = stage['last_log_age_seconds']
        suffix = f"; log {int(age)}s ago" if age is not None else ''
        lines.append(f"{stage['name']}: {stage['state']}{suffix}")
        lines.extend(f"  {note}" for note in stage['notes'])
    for name, root in report['experiments'].items():
        lines.append(name+':')
        for seed in root['seeds']:
            detail = f"epoch {seed['current_epoch'] or '-'}, checked {seed['last_validation_epoch'] or '-'}, best {seed['best_epoch'] or '-'}"
            if seed['best_score'] is not None:
                detail += f" ({seed['best_score']:.4f})"
            if seed['stop_reason']:
                detail += ', '+seed['stop_reason']
            lines.append(f"  seed {seed['seed']}: {seed['state']}; {detail}")
    lines.extend('Attention: '+note for note in report['issues'])
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workspace', type=Path, default=Path('.'))
    parser.add_argument('--stale-minutes', type=float, default=30)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()
    if args.stale_minutes <= 0:
        parser.error('--stale-minutes must be positive')
    report = collect_status(args.workspace, stale_seconds=args.stale_minutes*60)
    print(json.dumps(report, indent=2) if args.json else human_report(report))


if __name__ == '__main__':
    main()
