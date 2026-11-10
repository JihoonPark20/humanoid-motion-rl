# ZEST · MuJoCo

[English](README.md) | **한국어**

`legged-gym-mujoco`를 기반으로 구성한 **Unitree G1(29 DoF)**용 [ZEST](https://arxiv.org/abs/2602.00401) 비공식 구현입니다. MuJoCo 물리는 CPU에서, PPO 학습은 **MPS·CUDA·CPU**에서 실행합니다.

| 논문 | 핵심 아이디어 | 워크스페이스 상태 |
| --- | --- | --- |
| [ZEST](https://arxiv.org/abs/2602.00401) | 적응형 샘플링과 보조 힘으로 기준 모션 추종을 학습합니다. | 차이점을 문서화한 구현 완료; 수렴 미검증. |
| [HuMBLE](https://arxiv.org/abs/2610.10489) | 모션 추종 교사를 속도 명령 정책으로 증류한 뒤 멀티태스크 RL로 개선합니다. | 평지 학습 단계 구현 완료; 수렴 미검증. |

HuMBLE은 [HOLD](https://github.com/rai-opensource/hold)의 G1 locomotion 데이터를 지원합니다. 아래 빠른 시작은 ZEST용 LAFAN1을 사용합니다. 데이터 준비·네 단계 학습·재생·학습 재개는 [HuMBLE 가이드](docs/humble.md)를 참고하세요.

## 빠른 시작

Python 3.10–3.13을 사용합니다. Linux/NVIDIA에서는 CUDA용 [PyTorch](https://pytorch.org/get-started/locally/)를 먼저 설치하세요.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e .

# Unitree LAFAN1 G1 기준 모션 중 10초를 다운로드·변환합니다.
zest-data --seconds 10
zest-train --motions data/lafan1/walk1_subject1.npz --device mps
```

NVIDIA에서는 `--device cuda`, 자동 선택은 `--device auto`를 사용합니다. 기본값은 환경 64개, CPU 스레드 10개, PPO 1,000회입니다. `--num-envs`, `--num-threads`, `--iterations`로 변경할 수 있습니다.

## 재생과 평가

`PATH`를 `logs/g1/` 아래의 체크포인트 경로로 바꾸세요.

```bash
# 브라우저 뷰어(Python 3.11+): http://127.0.0.1:8080
python -m pip install -e '.[viewer]'
zest-view --checkpoint PATH
zest-evaluate --checkpoint PATH
zest-train --resume PATH --iterations 1000
```

평가는 모션 처음부터 시작하며 보조 힘을 사용하지 않습니다. 결과 JSON·CSV는 체크포인트 옆에 저장됩니다. 노이즈·외란·물리 변동을 포함하려면 `--randomized`, 별도 평가 모션은 `--motions`를 사용하세요. 기준 모션은 `zest-view --motion FILE`, 화면 없는 재생은 `zest-play --checkpoint PATH --headless`를 사용합니다.

## 구현 범위

ZEST 추종과 HuMBLE 교사 학습·증류·critic 적응·멀티태스크 PPO, 재생·평가를 포함합니다. **CPU·MPS에서 짧은 실행을 검증했으며 CUDA는 미검증입니다.** 짧은 검증 실행은 수렴이나 논문 수준 성능을 의미하지 않습니다. 험지·실제 로봇 배포·사람 모션 리타게팅은 이번 범위에서 제외합니다.

[데이터](docs/data.md) · [구현](docs/implementation.md) · [Viser](docs/viewer.md) · [HuMBLE 검토](docs/humble-review.md) · [검증](docs/validation.md) · [출처](docs/credits.md)
