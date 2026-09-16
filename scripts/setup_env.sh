#!/usr/bin/env bash
# camera_lidar_calibration GUI 환경을 한 번에 구성한다.
#
#   ./scripts/setup_env.sh
#
# 하는 일: 아키텍처에 맞는 파이썬 확인 (x86_64 -> 3.10, Jetson Orin/aarch64 ->
# 3.8, JetPack ROS Noetic 기본값에 맞춘 의도적인 선택) -> Qt(xcb) 시스템
# 라이브러리 설치 -> .venv 생성 -> requirements.txt 설치. 실행 후에는:
#
#   source .venv/bin/activate
#   python gui/app.py
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
venv_dir="${project_dir}/.venv"
arch="$(uname -m)"

if [[ "${arch}" == "aarch64" ]]; then
    default_python="python3.8"
else
    default_python="python3.10"
fi
python_bin="${PYTHON_BIN:-${default_python}}"

if ! command -v "${python_bin}" >/dev/null 2>&1; then
    echo "ERROR: ${python_bin}을 찾을 수 없습니다." >&2
    echo "  이 아키텍처(${arch})의 기본 파이썬은 ${default_python}이다 (gui/README.md 참고)." >&2
    if [[ "${arch}" == "x86_64" ]]; then
        echo "  Ubuntu에서 설치:" >&2
        echo "    sudo add-apt-repository ppa:deadsnakes/ppa" >&2
        echo "    sudo apt update && sudo apt install -y python3.10 python3.10-venv" >&2
    elif [[ "${arch}" == "aarch64" ]]; then
        echo "  Jetson Orin(JetPack)에는 보통 ROS Noetic과 함께 python3.8이 이미 있다." >&2
        echo "  없다면: sudo apt install -y python3.8 python3.8-venv" >&2
    else
        echo "  PYTHON_BIN=<경로>로 원하는 파이썬을 지정해 재실행하라." >&2
    fi
    exit 1
fi

echo "[1/3] 시스템 패키지 설치 (Qt6 xcb 플러그인, venv 모듈)"
sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    "${python_bin}-venv" \
    libxcb-cursor0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 \
    libxcb-randr0 libxcb-render-util0 libxcb-shape0 libxcb-xinerama0 \
    libegl1 libgl1 libxkbcommon0 libxkbcommon-x11-0

echo "[2/3] 가상환경 생성: ${venv_dir}"
"${python_bin}" -m venv "${venv_dir}"
"${venv_dir}/bin/python" -m pip install --upgrade pip

echo "[3/3] 의존성 설치"
# open3d(60MB대)처럼 큰 wheel이 있어 느리거나 불안정한 네트워크에서 기본
# 타임아웃(pip 기본 15초)에 걸리기 쉽다. 타임아웃을 늘리고 재시도한다.
"${venv_dir}/bin/python" -m pip install --default-timeout=180 --retries 5 \
    -r "${project_dir}/requirements.txt"

echo
echo "설치 완료. 실행:"
echo "  source ${venv_dir}/bin/activate"
echo "  python gui/app.py"
