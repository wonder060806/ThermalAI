#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "Usage: $0 {render|start|status|attach|stop|run} {small|physics|3d} [project_root] [python]" >&2
  exit 2
}

operation="${1:-}"
stage="${2:-}"
project_root="${3:-${HOME}/aicooling/ThermalAI-full/experiments/report_revision}"
python_command="${4:-python3}"

case "$operation" in
  render|start|status|attach|stop|run) ;;
  *) usage ;;
esac

case "$stage" in
  small)
    matrix_relative="configs/small_sample_workstation.json"
    artifacts_relative="workstation_artifacts"
    ;;
  physics)
    matrix_relative="configs/physics_workstation.json"
    artifacts_relative="workstation_artifacts_physics"
    ;;
  3d)
    matrix_relative="configs/realistic_3d_workstation.json"
    artifacts_relative="workstation_artifacts_3d"
    ;;
  *) usage ;;
esac

project_root="$(realpath -m "$project_root")"
runner="$project_root/workstation_runner.py"
matrix="$project_root/$matrix_relative"
artifacts="$project_root/$artifacts_relative"
results="$project_root/results"
log="$results/ssh_${stage}.log"
status_file="$results/ssh_${stage}_status.json"
session_name="thermalai-$stage"
script_path="$(realpath -m "$0")"

resolve_python() {
  if [[ "$python_command" == */* ]]; then
    [[ -x "$python_command" ]] || { echo "Python executable does not exist: $python_command" >&2; exit 1; }
    local python_dir python_name
    python_dir="$(cd "$(dirname -- "$python_command")" && pwd -P)"
    python_name="$(basename -- "$python_command")"
    printf '%s/%s\n' "$python_dir" "$python_name"
  else
    command -v "$python_command" || { echo "Python executable does not exist: $python_command" >&2; exit 1; }
  fi
}

assert_training_inputs() {
  [[ -f "$runner" ]] || { echo "runner does not exist: $runner" >&2; exit 1; }
  [[ -f "$matrix" ]] || { echo "matrix does not exist: $matrix" >&2; exit 1; }
}

render_json() {
  local python_exec="$1"
  local runner_command
  runner_command="$python_exec '$runner' --matrix '$matrix' --gpus 0,1,2,3 --artifacts '$artifacts'"
  "$python_exec" -c 'import json,sys; print(json.dumps({"session_name":sys.argv[1],"stage":sys.argv[2],"project_root":sys.argv[3],"python":sys.argv[4],"matrix":sys.argv[5],"artifacts":sys.argv[6],"log":sys.argv[7],"status_file":sys.argv[8],"runner_command":sys.argv[9],"disconnect_safe":True}, ensure_ascii=False))' \
    "$session_name" "$stage" "$project_root" "$python_exec" "$matrix" "$artifacts" "$log" "$status_file" "$runner_command"
}

if [[ "$operation" == "render" || "$operation" == "start" || "$operation" == "run" ]]; then
  assert_training_inputs
  python_exec="$(resolve_python)"
fi

case "$operation" in
  render)
    render_json "$python_exec"
    ;;
  start)
    command -v tmux >/dev/null || { echo "tmux is not installed" >&2; exit 1; }
    for other_stage in small physics 3d; do
      other_session="thermalai-$other_stage"
      if tmux has-session -t "$other_session" 2>/dev/null; then
        echo "ThermalAI session already running: $other_session" >&2
        exit 1
      fi
    done
    mkdir -p "$results" "$artifacts"
    printf -v launch_command '%q ' "$script_path" run "$stage" "$project_root" "$python_exec"
    tmux new-session -d -s "$session_name" "${launch_command% }"
    render_json "$python_exec"
    ;;
  status)
    if command -v tmux >/dev/null && tmux has-session -t "$session_name" 2>/dev/null; then
      printf '{"session_name":"%s","stage":"%s","state":"running","log":"%s"}\n' "$session_name" "$stage" "$log"
    elif [[ -f "$status_file" ]]; then
      cat "$status_file"
    else
      printf '{"session_name":"%s","stage":"%s","state":"not_started","log":"%s"}\n' "$session_name" "$stage" "$log"
    fi
    ;;
  attach)
    command -v tmux >/dev/null || { echo "tmux is not installed" >&2; exit 1; }
    tmux has-session -t "$session_name" 2>/dev/null || { echo "session is not running: $session_name" >&2; exit 1; }
    exec tmux attach-session -t "$session_name"
    ;;
  stop)
    command -v tmux >/dev/null || { echo "tmux is not installed" >&2; exit 1; }
    tmux has-session -t "$session_name" 2>/dev/null || { echo "session is not running: $session_name" >&2; exit 1; }
    tmux kill-session -t "$session_name"
    printf '{"session_name":"%s","stage":"%s","state":"stopped","artifacts_preserved":true}\n' "$session_name" "$stage"
    ;;
  run)
    mkdir -p "$results" "$artifacts"
    started_at="$(date --iso-8601=seconds)"
    printf '{"session_name":"%s","stage":"%s","state":"running","started_at":"%s"}\n' "$session_name" "$stage" "$started_at" > "$status_file"
    finish_status() {
      exit_code=$?
      finished_at="$(date --iso-8601=seconds)"
      printf '{"session_name":"%s","stage":"%s","state":"finished","exit_code":%d,"started_at":"%s","finished_at":"%s","log":"%s"}\n' \
        "$session_name" "$stage" "$exit_code" "$started_at" "$finished_at" "$log" > "$status_file"
    }
    trap finish_status EXIT
    echo "[$(date --iso-8601=seconds)] starting $session_name" | tee -a "$log"
    "$python_exec" "$runner" --matrix "$matrix" --gpus 0,1,2,3 --artifacts "$artifacts" 2>&1 | tee -a "$log"
    ;;
esac
