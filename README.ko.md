# ZEST · HuMBLE · MuJoCo

[English](README.md) | **한국어**

`legged-gym-mujoco`를 기반으로 구성한 **Unitree G1(29 DoF)**용 비공식 구현입니다. MuJoCo 물리는 CPU에서, 학습은 **MPS·CUDA·CPU**에서 실행합니다.

| 논문 | 핵심 아이디어 | 상태 |
| --- | --- | --- |
| [ZEST](https://arxiv.org/abs/2602.00401) | 적응형 샘플링과 보조 힘으로 기준 모션 추종을 학습합니다. | 차이점을 문서화한 구현 완료; 수렴 미검증. |
| [HuMBLE](https://arxiv.org/abs/2610.10489) | 추종 교사를 속도 명령 정책으로 증류한 뒤 멀티태스크 RL로 개선합니다. | 평지 학습 단계 구현 완료; 수렴 미검증. |

두 구현 모두 **[HOLD](https://github.com/rai-opensource/hold) G1 locomotion**을 기본 데이터로 사용합니다. HOLD는 RAI Institute와 Boston Dynamics가 공개하고 HuMBLE에 사용한 데이터셋입니다. ZEST에도 사용하는 것은 이 프로젝트의 선택입니다.

## 설치와 데이터 준비

Python 3.10–3.13을 사용하며 Viser는 3.11+가 필요합니다. Linux/NVIDIA에서는 CUDA용 [PyTorch](https://pytorch.org/get-started/locally/)를 먼저 설치하세요.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# HOLD의 S0_normal_walk_1에서 처음 30초를 준비합니다.
zest-data --seconds 30 --height-offset 0.005
```

명시적인 5 mm 높이 보정은 로컬 G1 모델에서 이 보행 모션의 바닥 관통을 줄이며 메타데이터에 기록됩니다. 단일 클립은 시작 예제입니다. 데이터 확장은 [데이터 준비](docs/data.md)를 참고하세요. 기존 변환 파일은 덮어쓰지 않습니다.

## 학습

ZEST 추종 학습 또는 HuMBLE 파이프라인을 선택합니다.

```bash
# ZEST
zest-train --motions data/hold/*.npz --device mps --iterations 10000

# HuMBLE: 교사 → 증류 → critic 적응 → 멀티태스크 PPO
humble-train --stage teacher --motions data/hold/*.npz --config configs/humble-flat.json --device mps --iterations 10000
humble-evaluate --checkpoint TEACHER.pt --mode reference
humble-train --stage distill --init TEACHER.pt --device mps --iterations 20000
humble-train --stage warmup --init PRIOR.pt --device mps --iterations 5000
humble-train --stage finetune --init WARMUP.pt --device mps --iterations 20000
```

`TEACHER.pt`, `PRIOR.pt`, `WARMUP.pt`를 `logs/humble/` 아래 이전 단계의 `latest.pt` 경로로 바꾸세요. 증류 전에 보조 힘 없는 교사의 추종 성능을 확인합니다. 반복 횟수는 예시이며 수렴을 보장하지 않습니다. NVIDIA에서는 `--device cuda`, 병렬도 조절은 `--num-envs`와 `--num-threads`를 사용합니다. 단계별 설명과 8개 클립 준비 예제은 [HuMBLE 가이드](docs/humble.md)를 참고하세요.

## 시각화·평가·재개

`PATH`를 해당 학습기의 체크포인트 경로(`logs/g1/` 또는 `logs/humble/`)로 바꾸세요.

```bash
python -m pip install -e '.[viewer]'
zest-view --checkpoint PATH  # http://127.0.0.1:8080; 두 학습기 모두 지원

# ZEST
zest-evaluate --checkpoint PATH
zest-train --resume PATH --device mps --iterations 10000

# HuMBLE 학생 정책(교사는 --mode reference)
humble-evaluate --checkpoint PATH --mode commands
humble-train --resume PATH --device mps --iterations 10000
```

평가는 보조 힘을 사용하지 않습니다. 기준 모션은 `zest-view --motion data/hold/S0_normal_walk_1.npz`로 확인합니다. HuMBLE은 Ctrl+C로 중단하면 저장하며, 재개 시 반복 횟수는 추가 학습량입니다.

**CPU·MPS에서 짧은 실행을 검증했으며 CUDA는 미검증입니다.** 논문 수준 성능·험지·실제 로봇 배포는 검증 범위에 포함되지 않습니다.

[데이터](docs/data.md) · [구현](docs/implementation.md) · [Viser](docs/viewer.md) · [HuMBLE 검토](docs/humble-review.md) · [검증](docs/validation.md) · [출처](docs/credits.md)
