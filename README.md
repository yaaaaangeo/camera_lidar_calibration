# Camera-LiDAR Calibration GUI

타겟(체스보드+ArUco 마커+원형 구멍 4개) 기반 카메라-LiDAR extrinsic 캘리브레이션을
GUI 마법사로 진행하는 도구. 알고리즘은 [FAST-Calib](https://github.com/xuankuzcr/FAST-Calib)을
그대로 쓰고, ROS를 띄우고 `roslaunch`로 한 번에 계산하던 원본 사용 방식을 단계별로
확인하며 진행할 수 있는 데스크톱 앱으로 바꾼 것이다. 
원본 알고리즘/원본 사용법 ([README.fastcalib.md](README.fastcalib.md), [workflow.md](workflow.md))

## 1. 요구 사항

- **Python**: x86_64 개발 머신은 **3.10**, Jetson Orin(aarch64) 배포 대상은
  **3.8** — Orin은 JetPack의 기본 ROS Noetic이 3.8을 쓰기 때문에 의도적으로
  맞춘 것이지 미지원이 아니다. 두 조합 모두 `requirements.txt` 하나로 설치된다
- OS: Linux
- **ROS 불필요.** `rosbags`가 순수 파이썬으로 `.bag`/ROS2 bag 폴더를 직접 읽으므로
  `source /opt/ros/...`도 `roscore`도 필요 없다
- 시스템 라이브러리: Qt6 xcb 플러그인(`libxcb-cursor0` 등) — 아래 설치 스크립트가 처리

## 2. 설치 (한 번에)

```bash
cd camera_lidar_calibration
./scripts/setup_env.sh
```

아키텍처에 맞는 파이썬(x86_64는 3.10, aarch64/Jetson Orin은 3.8) 확인 → Qt
시스템 라이브러리 설치(`apt`, sudo 필요) → `.venv` 생성 → `requirements.txt`
설치까지 한 번에 끝낸다. 해당 파이썬이 없으면 설치 방법을 안내하고 종료한다.

**수동으로 하려면:**

```bash
python3.10 -m venv .venv   # Jetson Orin(aarch64)는 python3.8 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

sudo apt install -y libxcb-cursor0   # Qt 6.5+ xcb 플러그인
```

`requirements.txt`가 설치하는 것:

| 패키지 | 용도 |
|---|---|
| `PySide6` | 데스크톱 GUI (Qt6) |
| `pyqtgraph` | 3D 포인트클라우드 뷰, undistort 미리보기 플롯 |
| `rosbags` | ROS 설치 없이 `.bag`/ROS2 bag 읽기 |
| `opencv-python` | ArUco 검출, `solvePnP` 기반 보드 pose 추정 |
| `open3d` | RANSAC 평면 피팅 등 포인트클라우드 유틸 |
| `numpy`, `scipy` | 배열 연산, 피팅 |
| `PyYAML` | 카메라/타겟 프리셋과 프로젝트 파일(`.calib.yaml`) 읽기·쓰기 |

**네트워크가 느리거나 자꾸 끊길 때** — `open3d`는 58MB대라 느린 회선에서는
pip가 다운로드 도중 끊기면 처음부터 다시 받는다. `wget -c`(이어받기)로 미리
받아둔 뒤 로컬 파일로 설치하면 끊겨도 받은 부분부터 이어진다:

```bash
# PyPI에서 정확한 wheel 주소 확인 (버전/아키텍처/파이썬 버전에 맞게)
python3 -c "
import json, urllib.request
data = json.load(urllib.request.urlopen('https://pypi.org/pypi/open3d/0.18.0/json'))
for f in data['urls']:
    if 'aarch64' in f['url'] and 'cp38' in f['url']:
        print(f['url'])
"

wget -c "<위에서 나온 URL>" -O /tmp/open3d.whl   # 끊기면 같은 명령 재실행 -> 이어받기
pip install /tmp/open3d.whl
pip install --default-timeout=180 --retries 5 -r requirements.txt   # 나머지
```

## 3. 실행

```bash
source .venv/bin/activate
python gui/app.py                          # 새 프로젝트
python gui/app.py 내프로젝트.calib.yaml    # 이어서 작업
```

**모든 명령은 이 폴더(`camera_lidar_calibration/`)에서 실행한다.**

## 4. 이 도구가 하는 일

- FAST-Calib의 계산 로직(`src/`, `include/`)은 그대로 두고, 데이터 준비부터
  캘리브레이션까지 진행 과정을 **7단계 마법사**로 만들었다. 각 단계는 다음
  단계로 넘어가기 전에 결과를 눈으로 확인할 수 있게 되어 있다
- bag을 **자르지 않는다.** scene은 `(bag 파일, 타임스탬프)`로만 기록되고,
  실제 디코딩은 그때그때 이루어진다
- 카메라 intrinsics·타겟 치수를 코드가 아니라 `gui/config/*.yaml` 프리셋으로
  관리해서, 새 카메라/보드가 늘어나도 파일만 고치면 된다
- LiDAR 원(구멍) 검출은 원본 방식(기계식 링 점프, 솔리드 이웃 각도) 외에
  이 저장소에서 추가한 방식(평면 격자, 링 점프 쌍 묶기, 실험용)까지
  **5가지**를 지원하고 같은 scene으로 바로 비교할 수 있다
- 여러 scene을 합쳐 하나의 extrinsic을 풀고, **scene 하나를 뺐을 때 답이
  얼마나 움직이는지**(영향도)를 보여줘서 조용히 틀린 scene을 걸러낼 수 있게 한다
- 환경/데이터를 점검하는 진단 스크립트 모음(`gui/check_*.py`)이 딸려 있다

## 5. 폴더 구조

```
camera_lidar_calibration/
├── gui/                    # 파이썬 GUI 툴 — 이 문서가 다루는 것
│   ├── app.py              # 진입점
│   ├── ui/                 # PySide6 위젯: 메인 윈도우(main_window.py) + 단계별 페이지(steps/)
│   ├── core/               # 알고리즘 — bag 읽기, 카메라/LiDAR 검출, 정합(solve), 검증
│   │                       #   (ROS·Qt에 의존하지 않아 headless 테스트/스크립트에서도 재사용)
│   ├── config/             # 카메라 intrinsics·타겟 치수 프리셋 (yaml, 코드 아님)
│   ├── check_*.py          # 진단/회귀 스크립트 (8번 참고)
│   └── README.md           # 단계별 조작법·판단 기준 상세 문서
├── src/, include/          # 원본 FAST-Calib 알고리즘 (C++, 참조용 — 건드리지 않음)
├── launch/, CMakeLists.txt,
│   package.xml             # 원본 ROS1 노드 빌드/실행 설정
├── config/qr_params.yaml   # 원본 ROS 노드용 파라미터 (GUI는 쓰지 않음)
├── calib_data/, calib_result/   # 원본 방식 사용 시의 입출력 예시
├── scripts/
│   ├── setup_env.sh        # 환경 일괄 설정 (2번)
│   └── distance_filter_tool.py  # 거리 필터 파라미터를 빠르게 잡아보는 보조 스크립트
├── requirements.txt        # gui/ 파이썬 의존성
├── README.md               # 이 문서
├── README.fastcalib.md     # 원본 FAST-Calib README (설치·roslaunch 사용법)
└── workflow.md             # 원본 알고리즘(원 중심 추출) 워크플로우 설명
```

## 6. GUI 7단계

| 단계 | 내용 | 상태 |
|---|---|---|
| 1. 데이터 | bag 추가, LiDAR/카메라 토픽 지정 | 동작 |
| 2. 카메라 | intrinsics 프리셋 선택/입력, undistort로 검증 | 동작 |
| 3. 타겟 | 보드 치수 입력, 실척 도면으로 확인 | 동작 |
| 4. Scene 캡처 | 타임라인에서 시점 기록(자르지 않음), 자동 추천 | 동작 |
| 5. 거리 필터 | 3D 박스로 보드만 남기고 원 검출(5가지 방식 비교) | 동작 |
| 6. 캘리브레이션 | scene들을 합쳐 extrinsic 계산, scene별 영향도 확인 | 동작 |
| 7. 검증 | extrinsic으로 포인트클라우드를 이미지에 투영해 확인 | 미구현 |

각 단계의 세부 조작법, 경고 값이 뜨는 이유, 알려진 함정은
**[gui/README.md](gui/README.md)**에 정리되어 있다.

## 7. 진단 도구

| 스크립트 | 용도 |
|---|---|
| `gui/check_gl.py` | OpenGL/포인트클라우드 렌더링이 되는지 확인 |
| `gui/check_bag.py <bag...>` | bag의 토픽/타입/개수/주파수 확인 |
| `gui/check_frame.py <bag> --lidar <토픽> --camera <토픽>` | 특정 시점 한 프레임을 디코딩해 점 개수·검출 여부 확인 |
| `gui/check_synth.py [--detail\|--sweep ...]` | 합성 데이터 96조건으로 검출 방식 회귀 테스트 |
| `gui/check_all.py cases.yaml` | 여러 bag을 일괄로 돌려 검출 결과 비교 (회귀 하네스) |

명령 예시는 [gui/README.md](gui/README.md#진단-도구)에 있다.

## 8. 알려진 제약

- **7단계(투영 검증)는 아직 구현되지 않았다.**
- `gui/core/detect_auto.py`(스윕 수 자동 선택)는 미연결 상태다 — 연결 조건은
  파일 상단 주석에 근거(측정치)와 함께 남아 있다
- **두 조합을 함께 지원한다:** x86_64+python3.10(개발)과 Jetson Orin
  aarch64+python3.8(JetPack ROS Noetic 기본값에 맞춘 것). `requirements.txt`는
  `python_version` 마커로 버전을 나눠 고정한다 — python3.8 쪽은 PyPI가 그 이후
  버전부터 3.8 지원을 끊어서 마지막으로 지원하는 버전에 묶여 있다:
  - `open3d`: aarch64 wheel이 0.18.0까지만 있어 범위(`>=0.18,<0.20`)로 열어둠
  - `rosbags`: 0.10.0부터 `Requires-Python >=3.10` → 3.8은 `0.9.23` 고정
  - `PySide6`: 6.9.3의 aarch64 wheel은 `manylinux_2_39`(glibc 2.39)+cp39
    이상 전용이라 Ubuntu 20.04(glibc 2.31)+python3.8에서 설치 자체가 안 됨 →
    `manylinux_2_31`+cp38로 받을 수 있는 마지막 버전인 `6.6.3.1` 고정
  - `pyqtgraph`: 0.14.0부터 `Requires-Python >=3.10` → 3.8은 `0.13.3` 고정
- `scripts/setup_env.sh`는 아키텍처로 필요한 파이썬을 판단하지만, 해당 파이썬
  자체가 없으면(예: 이 시스템에 3.8/3.10이 아예 없는 경우) 설치까지 자동화하지
  않고 안내만 출력한다
- 원본 FAST-Calib(ROS1 C++, `src/`)은 그대로 유지되며 GUI와 무관하게
  `roslaunch`로 독립 실행할 수 있다 — [README.fastcalib.md](README.fastcalib.md) 참고

## 라이선스

원본 FAST-Calib를 따라 [GPLv2](LICENSE).
