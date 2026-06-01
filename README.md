# G1Nav — 언어 조건부 Unitree G1 하체 내비게이션 (MuJoCo / MJX)

자유형 영문 지시문(예: *"go to the red cube"*)과 온보드 **RGB-D + 고유감각 센서**만으로,
Unitree G1 휴머노이드가 MuJoCo 물리 시뮬레이션 안에서 **하체 관절 목표를 직접 출력**하며
대상 객체까지 걸어간다. 테스트 시 키네마틱 모션 없음 — 오직 물리로만 움직인다.

> Ludo Robotics 과제 전형 · Isaac GR00T **N1.6** 파라미터만 재사용, 그 외 사전학습 체크포인트 없음.

---

## 🎬 결과 영상 (지시문 → 인지 → 보행 → 도달)

각 영상은 **전체 파이프라인을 끝까지 실행**한 것이다: GR00T N1.6 Eagle 인지가 지시문의 대상을
zero-shot으로 특정 → Navigator → 학습된 PPO 보행 정책 → MuJoCo 물리. **왼쪽 = 3인칭, 오른쪽 =
에고(로봇 시점)**. 시작 2초 정지 → 보행 → 도달 후 2초 정지. 3개 씬 모두 **직립 유지·도달 성공**.

| 지시문 | 결과 | 영상 |
|---|:---:|---|
| **"go to the orange cylinder"** | ✅ 도달 (0.39 m) | [▶ seed0](https://github.com/physical-ai-vla/g1_target_walking/raw/main/videos/seed0_go-to-the-orange-cylinder.mp4) |
| **"go to the red cube"** | ✅ 도달 (0.41 m) | [▶ seed1](https://github.com/physical-ai-vla/g1_target_walking/raw/main/videos/seed1_go-to-the-red-cube.mp4) |
| **"go to the purple cylinder"** | ✅ 도달 (0.54 m) | [▶ seed2](https://github.com/physical-ai-vla/g1_target_walking/raw/main/videos/seed2_go-to-the-purple-cylinder.mp4) |

<video src="https://github.com/physical-ai-vla/g1_target_walking/raw/main/videos/seed0_go-to-the-orange-cylinder.mp4" controls muted width="100%"></video>

> 영상이 위에서 바로 재생되지 않으면 표의 ▶ 링크를 클릭하세요. 각 영상은 파일명에 지시문이
> 담겨 있고, 매칭되는 `*.cognition.json`(N1.6 grounding 결과) + `*.ego.png`(시작 프레임)와 함께
> `videos/`에 들어 있다 ([videos/VIDEOS.md](videos/VIDEOS.md)).

---

## 💡 우리는 왜 이렇게 만들었나 (설계 철학)

과제는 "작은 VLA를 학습하라"고 한다. 우리는 그것을 **단일 픽셀→관절 네트워크를 파인튜닝하는
대신, 동결된 VLM 인지 단계(GR00T N1.6 Eagle, zero-shot) + 별도로 학습한 RL 보행 체크포인트의
결합**으로 구현했다. 다섯 가지 이유:

**1. 뇌과학(신경과학) 패턴을 따랐다.**
생물의 운동 제어는 계층적이다 — 느리고 숙고적인 **피질(cortex)**(지각→식별→위치특정)이 빠르고
반응적인 보행용 **척수 중추패턴생성기(CPG)** 위에 얹혀 있다. 우리는 이 분리를 그대로 모방했다:
동결 파운데이션 모델 = 인지 피질, 학습된 반응형 정책 = 척추. 둘을 하나의 블랙박스로 합치지 않는다.
**이 분해 자체가 설계의 핵심이다.**

**2. VLM은 이미 성능이 뛰어나며, 잘못 파인튜닝하면 그 성능이 망가진다.**
GR00T N1.6의 비전-언어 백본은 대상 객체를 이미 zero-shot으로 인식한다. 2~3B 파라미터 모델을
작고 좁은 자체 데이터셋에 파인튜닝하면 그 광범위한 능력을 개선하기보다 **저하(catastrophic
forgetting / 과적합)** 시킬 위험이 훨씬 크다. 동결해 쓰면 일반 추론 능력을 온전히 보존한다.

**3. 현재 VLA 트렌드는 액션 모듈의 최소화다.**
최신 VLA 연구는 무겁고 별도로 학습된 액션 헤드에서 멀어지고 있다. 예컨대 **VLA-0** 은 강력한
VLM이 *전용으로 학습된 액션 모듈 없이도* 본질적으로 행동을 산출할 수 있음을 보였다 — 액션
인터페이스를 최대한 얇게 만들고 VLM의 추론이 그 부담을 떠맡는다. 우리도 같은 철학이다: 우리의
"액션 모듈"은 얇은 Navigator + 컴팩트한 RL 보행 컨트롤러이지, 수집 시연 위에 학습된 거대한
정책 헤드가 아니다.

**4. 뇌과학식 분해를 따르니, grounding을 위한 학습된 액션 모듈이 아예 필요 없다.**
인지(무엇/어디)는 동결 VLM + 고전 CV로 해결되고, 진짜로 학습이 필요한 것은 *물리적 균형/이동*
뿐이다. 따라서 언어+픽셀을 관절로 사상하는 end-to-end 액션 헤드가 필요 없다.

**5. VLA 등장 이후 병목은 아키텍처/학습이 아니라 데이터 수집의 영역으로 바뀌었다 — 우리 방법은
바로 그것을 최소화한다.**
현대 VLA에서 어렵고 시간이 많이 드는 부분은 더 이상 네트워크나 학습 레시피가 아니라 **크고 잘
정제된 시연 데이터셋의 수집**이다. 우리 접근은 그것을 우회한다: 인지는 **데이터가 0**(zero-shot),
이동은 수집 궤적이 아니라 **시뮬레이션 보상**으로 학습된다. 이 분야가 지금 가장 많이 소모하는
자원(데이터)이 여기서는 거의 0으로 줄어든다.

> **한 줄 요약:** *상위 grounding은 데이터가 필요 없고(프롬프팅으로 충분), 하위 이동만 RL이
> 필요하며 그조차 한 번 학습해 모든 지시문에 분할상환된다.*

---

## 🧠 시스템 구조

```
  고정된 영문 지시문 ─┐
                     ▼
  ego RGB-D ───► Eagle 피질 (GR00T N1.6, zero-shot)──► 대상 정체 (무엇)
                 + 색상/형상 CV + ego-depth         ──► 대상 (방위각·거리, 어디)
                             │                          └─► 월드 좌표 (1회 고정; 정적 씬)
                             ▼
                       Navigator  ── world_to_ego ─► 속도 명령 (vx, vy, ωz)
                             │
                             ▼  (매 20 ms)
        obs(103) ─► PPO 보행 정책 ─► action(29) ─► motor = default_pose + a·scale
                             │                              │
                       고유감각 센서 ◄──────── MuJoCo 물리 (mj_step ×10)
```

| 계층 | 무엇을 | 어떻게 | 학습? |
|---|---|---|:---:|
| **인지** (무엇/어디) | 지시 대상 식별·위치특정 | GR00T N1.6 **Eagle** 비전 인코더(지각) + 색상/형상 CV + ego-depth | ❌ zero-shot |
| **제어** (어떻게 이동) | 속도 명령대로 보행 | **PPO** 보행 정책, MJX `G1JoystickFlatTerrain`, **1회 학습** | ✅ RL, 분할상환 |
| **Navigator** | 대상 월드좌표 → `(vx,vy,ωz)` | `world_to_ego` → `goal_to_command`, 매 스텝 | — |

모델 출력은 문자 그대로 **29차원 하체 PD 관절 목표 벡터**이며, 속도나 상위 핸들이 아니다.

---

## 📊 결과 (검증됨)

- 보행 정책: **154.8M 스텝**까지 학습 → `checkpoint/walk_t4_latest.pkl`
- 전진 속도 **0.55 m/s** (명령 0.6 m/s의 ~92%), 몸통 높이 **0.74 m**, raw-env 평가에서 **안 넘어짐**
- 클로즈드루프 **3/3 도달**: 위 영상 표 참조 (직립·물리만)
- 인지 정확도: 3개 씬 모두 지시문 대상을 정확히 grounding (Eagle 임베딩 286 patch × 2048-d)

---

## 🚀 설치 & 실행 (2 커맨드)

```bash
git clone git@github.com:physical-ai-vla/g1_target_walking.git
cd g1_target_walking
bash install.sh              # GPU 자동감지 → 전체 스택, 없으면 대시보드 전용
source .venv/bin/activate && bash run_dashboard.sh
```

- **http://localhost:8502 — 인터랙티브** (직접 실행): 씬 선택 → 객체 확인 → 지시문 입력 → 실행 →
  결과 영상. GPU 있으면 로컬, 없으면 원격 GPU 파드에서 실행 (사이드바에 모드 표시).
- **http://localhost:8501 — 결과 재생**: 데모 영상·학습 곡선·N1.6 인지 (오프라인, GPU 불필요).

CLI 원샷:
```bash
python code/student/g1nav_run.py --seed 0 --instruction "go to the orange cylinder"
```

자세한 설치(원격 파드 모드 포함)·재현 방법: **[INSTALL.md](INSTALL.md)**.

---

## 📁 저장소 구조

```
code/
  teacher/   train_walk.py · walk_eval.py · rollout_walk.py      (RL 학습 + 정직한 평가)
  student/   cortex_eagle.py (N1.6 인지) · color_detect.py · navigate.py · action_module.py
             app_interactive.py (직접 실행 UI) · app_results.py (결과 대시보드) · g1nav_run.py
  envs/      scene_gen.py · instructions.py                      (시드 기반 씬/지시문 생성)
  utils/     setup_ada.sh · pod_bootstrap.sh · smoke_test.py
checkpoint/  walk_t4_latest.pkl  (+ 스윕용 walk_t4_step*.pkl)
dataset/     NOTE.md — 설계상 수집 데이터셋 없음 (zero-shot 인지 + 온라인 RL); 평가 씬만 재생성 가능
videos/      3편 (ego ∥ 3인칭, 파일명에 지시문) + cognition.json + ego.png + VIDEOS.md
report.pdf / report_ko.pdf   전체 보고서 (영문/한글)
```

## ✅ 과제 제약 준수

- **MuJoCo only** (MJX 학습, MuJoCo-C 일부 렌더). Isaac/Gazebo/Bullet 없음.
- **사전학습:** GR00T **N1.6 Eagle** 비전 인코더만 재사용; PPO 정책은 scratch부터 학습.
- **키네마틱 모션 없음:** 데모는 제어되는 바로 그 물리에서 렌더된다.
- **출력 = 하체 관절 목표 벡터** (속도/상위 핸들 아님).

전체 접근·아키텍처·엔지니어링 로그(버그와 수정 포함): **[report_ko.pdf](report_ko.pdf)** (한글) ·
**[report.pdf](report.pdf)** (영문).
