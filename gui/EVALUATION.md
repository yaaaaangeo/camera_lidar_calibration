# 7단계 "정량 평가" 가이드

7단계(검증)에는 탭이 두 개 있다: **검증**(오버레이 보면서 눈으로 확인 — flip, 캡처, extrinsic 내보내기)과
**정량 평가**(숫자로 확인 — 이 문서가 다루는 것). 정량 평가 탭 안에 기능이 6개나 있어서
헷갈리기 쉬우니, 각각 뭐고 왜 있고 어떤 순서로 쓰는지 정리한다.

## 절대 원칙: 아무것도 자동으로 안 바꾼다

이 문서에 나오는 어떤 버튼도 **현재 extrinsic(`T_cam_lidar`)을 고치지 않는다.**
전부 "지금 T가 맞는지/더 나은 게 있는지"를 계산해서 보여주기만 한다.
"이 값이 더 나아 보이니 적용" 버튼은 **의도적으로 없다** — 더 나아 보이는 후보를 찾으면
Candidate A/B Test Package Export로 내보내서 실제 차량/파이프라인에서 검증한 뒤,
사람이 판단해서 6단계를 다시 돌리거나 손으로 extrinsic 파일을 바꾸는 것까지가 이 도구의 역할이다.

## 큰 그림 — 기능 6개가 어떻게 이어지는가

```
① Multi-frame Consistency  ──┐  T가 여러 프레임에서 얼마나 안정적인지(탐색 아님)
                              │
② Perturbation Sensitivity ──┤  "6개 축 중 어디를 건드리면 더 좋아지나?" 넓게 훑기 (Quick/Full)
   (Discovery)                │      ↓ 후보 하나 찍기 (axis, delta)
③ Fine Scan  ─────────────────┤  그 축 하나만 촘촘하게 다시 훑기 (선택)
   (Discovery)                │      ↓ candidate 확정
④ Independent Hold-out ───────┤  Discovery에 안 쓴 별도 데이터로 그 candidate가
   Validation                 │  진짜 baseline보다 나은지 독립 검증
                              │      ↓
⑤ Diagnostic Evidence ────────┤  ①~④ 결과를 모아 "무슨 축 문제로 보이는지" 한번에 요약
                              │      ↓
⑥ Candidate A/B Test ─────────┘  baseline·candidate·검증 결과를 파일 패키지로 내보내기
   Package Export                (여기까지 — 적용은 사람이 한다)
```

②③이 **Discovery**(후보를 찾는 데이터), ④가 **Hold-out**(그 후보를 검증하는, Discovery와
분리된 데이터)이다. 이 구분이 이 탭 전체의 핵심 규칙이다 — 아래 "Discovery vs Hold-out"에서 자세히.

---

## ① Multi-frame Consistency (다중 프레임 일관성)

**질문**: "지금 T가 여러 시점에서 봐도 일관되게 맞나, 아니면 특정 프레임에서만 우연히 맞아 보이나?"

캡처한 여러 프레임에서 Edge Alignment 오차를 각각 재고 분포를 본다. 후보를 찾지도, 축을
건드리지도 않는다 — 순수하게 "지금 T"만 여러 프레임에 대해 재평가한다. 프레임 간 편차가
크면(특정 구간에서만 유독 오차가 크면) 캘리브레이션 자체보다 그 구간의 데이터/타이밍 쪽
문제일 수 있다는 신호다.

## ② Perturbation Sensitivity — Quick / Full (Discovery)

**질문**: "6개 축(roll/pitch/yaw/tx/ty/tz) 중 지금 T 근처를 조금 건드리면 오차가 줄어드는
축이 있나?"

각 축을 **하나씩만**, baseline은 그대로 둔 채 작은 delta로 흔들어보고 Edge Alignment 오차가
어떻게 움직이는지 보는 **Independent Extrinsic Parameter Sensitivity** 방법이다.
"6DoF를 동시에 최적화해서 새 T를 찾는" 것이 **아니다** — 그랬다면 이름부터 그렇게 지었을
것이다. 한 축씩만 흔드는 건 "이 파라미터 하나가 오차에 얼마나 민감한가"를 보려는 것이고,
동시에 여러 축을 흔기면 그 질문에 답할 수 없다.

- **Quick**: 축마다 delta 3개(회전 ±0.2°, 이동 ±10mm)만 — 첫 감을 빨리 잡을 때
- **Full**: 축마다 delta 더 많이(회전 9개: ±0.05~±0.5°, 이동 7개: ±5~±20mm) — 훑는 범위가 넓음

결과표에서 "Frame 단위 median 기준으로 최저점"이 baseline보다 뚜렷이 낮은 축이 있으면
그게 Fine Scan으로 넘어갈 후보다. 이 단계에서 쓴 프레임/시각은 내부적으로 기록되어
④ Hold-out 단계의 오염 방지에 쓰인다.

## ③ Fine Scan (Discovery)

②에서 유망해 보인 축 **하나만** 골라, 범위/간격을 직접 지정해 더 촘촘하게 다시 훑는다.
②와 완전히 같은 스코어링 코드를 쓴다 — 그냥 delta 목록이 사용자 지정이라는 점만 다르다.
필수 단계는 아니다: ②의 Quick/Full 결과 중 하나를 candidate로 바로 써도 된다.

## ④ Independent Hold-out Validation

**질문**: "②/③에서 찾은 candidate가, 그걸 찾는 데 안 쓴 다른 데이터에서도 진짜 baseline보다
나은가, 아니면 그 데이터에만 맞춰진 우연이었나?"

이게 이 탭에서 제일 복잡한 기능이라 따로 설명한다.

### Candidate 잠그기

축 콤보(②③과 공유), candidate 소스(Perturbation Grid 또는 Fine Scan), 그 delta를 고르고
**"Hold-out 평가 실행"**을 누르는 순간 baseline T와 candidate(axis, delta)가 그 자리에서
**고정(lock)**된다. 이후 콤보를 바꿔도 이미 실행 중인/끝난 결과에는 영향이 없다.

### 독립 데이터 소스

- **Separate Bag**: Discovery에 안 쓴 완전히 다른 bag
- **Current Bag Time Range**: 같은 bag이지만 Discovery가 안 건드린 시간 구간

같은 bag/시간대를 그대로 쓰면 "정답을 알고 시험 보는" 것과 같아서 의미가 없다. 그래서
**Discovery Exclusion Margin**(기본 `2.0초`)만큼 Discovery가 쓴 시각 근처는 Hold-out에서
자동으로 제외된다 — 정확히 같은 timestamp만 막는 게 아니라, 움직이는 차량에서는 수십ms
차이 나는 인접 프레임도 사실상 같은 장면이라 그만큼 여유를 둔 것이다. Hold-out bag이
Discovery bag과 내용까지 같아 보이면(우연히 복사된 파일 등) 경고가 뜬다.

### 결과 읽는 법

Baseline과 candidate를 **완전히 같은 hold-out 프레임 집합**에 대해 채점해서 나란히 비교한다.

| 항목 | 의미 |
|---|---|
| Median / P95 ΔFrame-P95 | candidate − baseline (음수 = candidate가 더 나음) |
| Improved / Worsened / Unchanged | 프레임 단위로 어느 쪽이 몇 개 이겼나 |
| 수평/수직/깊이 비교 | 화면 좌/중/우, 상/중/하, 0-10/10-20/20-30/30-50/50m+ 구간별 |

**95% CI (Non-overlapping Block Bootstrap)** — 개별 프레임을 독립 표본처럼 재표본하면
(연속 주행 프레임은 서로 상관돼 있어서) 불확실성을 과소평가한다. 그래서 시간축을
`Bootstrap Block Duration`(기본 `2.0초`) 폭의 겹치지 않는 블록으로 나누고, **블록 단위로만**
재표본한다. 블록이 5개 미만이면 CI 자체를 계산하지 않고 이유를 표시한다. 대부분 블록이
프레임 1개짜리면("sparse sampling") CI가 있어도 신뢰도가 낮다는 경고가 같이 뜬다.

Discovery Exclusion Margin과 Bootstrap Block Duration은 **서로 독립된 설정**이다 — 하나를
바꿔도 다른 하나는 안 바뀐다. 둘 다 "검증된 절대 기준"이 아니라 조정 가능한 기본값이다.

### Validation Status — Independent vs Exploratory

같은 candidate로 같은 hold-out을 "우연히 여러 번" 눌러본 것과, hold-out 결과를 보고
Discovery를 다시 돌려 candidate를 은근슬쩍 바꾼 것은 완전히 다르다 — 후자는 hold-out
데이터로 몰래 튜닝한 것과 같다. 그래서 결과마다 정확한 상태를 표시한다:

| 상태 | 뜻 |
|---|---|
| **Independent** | 이 candidate, 이 데이터 조합을 처음 봄 |
| **Independent — repeated evaluation** | 똑같은 candidate로 똑같은 데이터를 다시 돌림 (문제 없음) |
| **Independent — new unseen slice** | 같은 bag이지만 완전히 새로운(안 겹치는) 구간 |
| **Exploratory — candidate changed after this hold-out was viewed** | 같은 hold-out 데이터를 본 뒤, Discovery를 다시 안 돌리고 candidate만 바꿈 |
| **Exploratory — discovery was updated after this hold-out was viewed** | 이 hold-out을 본 뒤 Perturbation Grid/Fine Scan을 다시 성공적으로 돌림 (같은 delta로 다시 나와도 Exploratory) |

Exploratory여도 실행/내보내기가 막히지는 않는다 — 다만 "이건 더 이상 손대지 않은 candidate에
대한 독립 검증이 아니다"라는 걸 숨기지 않고 그대로 보여줄 뿐이다. 이 판정은 **프로젝트
파일에 영구 저장**되어 앱을 껐다 켜도, 심지어 candidate가 우연히 똑같은 값으로 다시
나와도 정확히 판정된다.

---

## ⑤ Diagnostic Evidence (진단 증거)

**질문**: "①~④ 결과를 다 모아 보면, 문제가 있어 보이는 축이 있는가? 있다면 어느 축일
가능성이 높아 보이는가?"

**"진단 요약 생성"**을 누르면 그 시점까지 계산된 ①~④ 결과(및 6단계 Leave-One-Out)를
모아 카테고리별(Rotation/Translation sensitivity, Spatial asymmetry, Depth dependence,
Temporal pairing, Multi-frame stability, Calibration-scene stability, Hold-out validation)로
증거를 정리하고, "Evidence strength"(Strong/Moderate/Weak/Mixed/Not observed)로 표시한다.

**이건 0~100점 정답 채점기가 아니다.** 여기서 계산하는 threshold(0.05px, 70% 등)는
"이 프로젝트에서 쓸만한 기본값"일 뿐 센서로 검증된 절대 기준이 아니다. 새로운 평가를
실행하지 않고 이미 계산된 결과만 다시 읽어서 정리하므로, 이 버튼을 누르기 전에
①~④ 중 보고 싶은 것들을 먼저 돌려둬야 한다.

---

## ⑥ Candidate A/B Test Package Export

**질문**: "baseline과 candidate를 실제 차량/projection/E2E 파이프라인에서 A/B로 비교하려면
뭘 넘겨줘야 하나?"

④ Hold-out 결과가 있어야 활성화된다. 누르면 `calib_result/candidate_ab_test_YYYYMMDD_HHMMSS/`
아래 파일 7개가 생긴다:

| 파일 | 내용 |
|---|---|
| `baseline_extrinsic.yaml` / `candidate_extrinsic.yaml` | 그대로 쓸 수 있는 T (기존 extrinsic 포맷 그대로) |
| `validation_summary.json` | 위 모든 결과의 기계 판독용 원본 |
| `per_frame_results.csv` | hold-out 프레임별 baseline/candidate 비교 |
| `spatial_depth_results.csv` | 구간별(수평/수직/깊이) 비교 |
| `validation_report.md` | 사람이 바로 읽는 요약 (맨 위에 Validation Status 굵게 표시) |
| `manifest.json` | 파일별 SHA-256 (재현성 확인용, 보안 서명 아님) |

**내보내기 직전에 두 가지를 다시 확인한다**: 지금 화면의 baseline이 hold-out이 잠갔던
baseline과 fingerprint까지 정확히 같은지, candidate를 baseline+axis+delta로 다시 계산했을 때
잠겼던 candidate fingerprint와 같은지. 하나라도 다르면 (flip 했거나, 다른 extrinsic을
불러왔거나, 6단계를 다시 돌렸거나) **내보내기 자체를 막고** 이유를 보여준다 — 옛날/불일치
결과가 파일로 나가는 것을 막기 위해서다.

`candidate_extrinsic.yaml`에는 "이건 Independent Extrinsic Parameter Sensitivity 결과이고
전체 SE(3) 최적화가 아니며, baseline을 자동으로 덮어쓰지 말 것"이라는 문구가 항상 들어간다.
이 도구가 baseline calibration YAML을 직접 덮어쓰는 기능은 없다.

---

## 핵심 개념 정리

### Discovery vs Hold-out

| | Discovery (②③) | Hold-out (④) |
|---|---|---|
| 역할 | candidate를 **찾는다** | 그 candidate가 **진짜인지 확인한다** |
| 데이터 | Perturbation/Fine Scan이 훑은 프레임 | Discovery가 안 쓴 별도 bag/시간대 |
| candidate를 여기서 바꿀 수 있나 | 예 (탐색이 목적) | **아니오** — 이미 잠긴 candidate만 채점 |

Hold-out 데이터로 candidate를 찾거나 조정하면 이 구분 자체가 무너진다 — 그래서 Hold-out
쪽 코드는 애초에 "탐색"이라는 개념 자체가 없다(grid도, delta 범위도 없이 딱 두 점만 채점).

### 좌표계 — 로보틱스 관례와 다르다

Yaw/Pitch/Roll은 흔한 "roll=X, pitch=Y, yaw=Z" 로보틱스 관례가 **아니라 카메라 자세** 기준이다
(카메라의 Y축이 아래를 향하므로 "좌우로 팬"하는 회전은 Z가 아니라 Y축이다):

| 이름 | 축 | 뜻 |
|---|---|---|
| Yaw / Pan | Camera **Y** (아래) | 좌우로 팬 |
| Pitch / Tilt | Camera **X** (오른쪽) | 위아래로 틸트 |
| Roll | Camera **Z** (앞) | 렌즈 축 중심 회전 |
| Tx / Ty / Tz | Camera X/Y/Z | 오른쪽 / 아래 / 앞 |

`T_cam_lidar`는 **LiDAR → Camera** 방향(`p_camera = R @ p_lidar + t`)이고, 카메라 프레임은
OpenCV 표준(X 오른쪽, Y 아래, Z 앞)이다.

### discovery_revision (지속 카운터)

Perturbation Grid나 Fine Scan이 **성공적으로** 결과를 낼 때마다 1씩 오르는, 프로젝트 파일에
영구 저장되는 카운터다. 앱을 재시작해도 리셋되지 않는다 — Hold-out을 본 뒤 재시작하고
Discovery를 다시 돌려도 "그 사이 Discovery가 움직였다"는 걸 놓치지 않기 위해서다
(실패/취소된 실행, Hold-out 실행 자체는 카운트하지 않는다).

---

## 추천 순서

```
1. (선택) Multi-frame Consistency로 지금 T가 안정적인지 감 잡기
2. Perturbation Sensitivity(Quick 먼저, 필요하면 Full)로 유망한 축 찾기
3. (선택) 그 축을 Fine Scan으로 좁혀서 delta 확정
4. Hold-out 소스를 Discovery와 겹치지 않게 고르고 candidate 잠근 뒤 "Hold-out 평가 실행"
5. 결과가 Independent이고 baseline보다 나아 보이면 Diagnostic Evidence로 다른 증거들과 같이 확인
6. 실제로 적용해볼 가치가 있다고 판단되면 Candidate A/B Test Package Export로 내보내서
   차량/파이프라인에서 실측 → 사람이 최종 판단
```

## 이 탭이 절대 하지 않는 것

- extrinsic을 새로 풀거나(6단계 재계산) 자동으로 고치지 않는다
- candidate를 "적용"하는 버튼이 없다
- baseline calibration YAML을 덮어쓰지 않는다
- 진단(⑤)이 0~100점 정답을 매기지 않는다 — 증거의 강도만 보여준다
- Hold-out(④)은 candidate를 탐색하지 않는다 — 이미 정해진 candidate 하나만 채점한다
