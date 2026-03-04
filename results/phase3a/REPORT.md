# Phase 3-A: L_mutual 구현 결과 보고

## 수행 일시
2026-03-04

## 수행 작업 요약
Phase 2-C에서 L_sem 독립 학습이 검증되었다 (mIoU=0.810, gradient 격리 확인). 이는 f_i가 2D GT로부터 의미론적 분류를 학습할 수 있고, 기하학적 파라미터에 간섭하지 않음을 의미한다.

Phase 3-A는 이 전제 위에서 L_mutual을 구현한다. L_mutual은 f_i(의미론)와 R_i(기하학)를 양방향으로 연결하는 손실 함수로, 본 연구의 핵심 기여이다. "벽으로 분류된 프리미티브의 법선을 수평으로 회전시키고, 법선이 수평인 프리미티브를 벽으로 분류한다"는 양방향 상호보강을 수학적으로 구현한다.

Phase 3-A의 검증 대상은 "코드가 수식대로 올바르게 구현되었는가"이다. 학습 효과 검증은 Phase 3-B(ablation)에서 수행한다.

## 정량 지표 (구현 Phase — gradient check/smoke test 결과)

### Gradient check: 양방향 gradient 존재
| Mode | ∂L/∂f_i | ∂L/∂R_i |
|------|---------|---------|
| full | 0.2091 | 0.4620 |
| sem2geo | 0.0000 | 0.4620 |
| geo2sem | 0.2091 | 0.0000 |

(PlanarSplat_Network 실제 모델, N=50 primitives)

### Gradient isolation: 무관한 파라미터에 gradient 없음
| 파라미터 | ∂L_mutual/∂ | 예상 | 비고 |
|---------|-------------|------|------|
| rot_q_normal_wxy | 0.4620 | non-zero | 법선 방향 결정 quaternion |
| rot_q_xyAxis | 0.0 | zero | 면내 회전, 법선 무관 |
| center | 0.0 | zero | 위치, L_mutual 수식에 없음 |

### Smoke test: optimizer.step()으로 L_mutual 감소
| iter | L_mutual |
|------|----------|
| 0 | 0.167294 |
| 4 | 0.161615 |
| 9 | 0.154841 |

10 iter 동안 단조 감소 (0.167→0.155, -7.5%), NaN 없음.

## 정성적 결과
해당 없음 (구현 Phase). 학습 결과는 Phase 3-B에서 생성.

## Go/No-Go 판단
- [x] Go / [ ] Retry / [ ] Switch

### #1. Gradient check — 양방향 gradient 존재 확인

**검증 이유:** L_mutual의 핵심은 f_i와 R_i가 서로를 개선하는 양방향 상호보강이다. gradient가 한쪽에만 흐르면 단방향 전달에 불과하므로, 양쪽 모두 non-zero여야 한다.

**결과: Go** — full mode에서 ∂L/∂f_i=0.209, ∂L/∂R_i=0.462 (양방향 non-zero).

### #2. mutual_mode 분리 — ablation용 detach 동작 확인

**검증 이유:** Phase 3-B ablation (e)(f)에서 "양방향이 각 단방향보다 우수한가"를 검증하려면, detach로 한쪽 gradient만 차단하는 기능이 정확해야 한다. sem2geo에서 f_i gradient가 0이 아니면 (e) 실험 결과를 신뢰할 수 없다.

**결과: Go** — sem2geo: f_i=0/R_i=0.462, geo2sem: f_i=0.209/R_i=0. 정확히 한쪽만 차단됨.

### #3. Smoke test — 실제 loss 감소 확인

**검증 이유:** Gradient check(#1)는 gradient 존재만 확인한다. gradient가 있어도 부호가 반대이거나 수치적으로 불안정하면 optimizer.step() 시 loss가 증가하거나 NaN이 발생한다. Smoke test는 forward→backward→step 사이클에서 L_mutual이 실제로 감소하는지 확인한다.

**결과: Go** — 0.167→0.155 단조 감소, NaN 없음.

### #4. 기존 기능 보존 — L_mutual 코드가 기존 학습에 간섭하지 않는지 확인

**검증 이유:** trainer.py에 L_mutual 코드를 추가했으므로, lambda_mutual=0일 때 기존 동작이 변경되지 않아야 한다. L_mutual 코드 경로 전체가 `if self.lambda_mutual > 0:` 조건문 안에 있으므로, lambda_mutual=0이면 해당 코드에 도달하지 않는다.

**결과: Go** — lambda_mutual=0으로 10 iter 실행: L_mutual=0.000000 (전 iter), 크래시 없음.

### 근거
Phase 2-C에서 gradient 격리가 확인되었고(L_sem→f_i only), Phase 3-A에서 L_mutual이 f_i와 R_i 양방향에 gradient를 보냄이 검증되었다. 따라서 Phase 3-B에서 L_mutual 활성화 시 기하학적 변화(Normal cos)를 ∂L_mutual/∂R_i에, 의미론적 변화(mIoU)를 ∂L_mutual/∂f_i에 귀속시킬 수 있다. 이는 ablation (c) vs (d) 비교의 전제조건이다.

## 추가 검증 사항

### Normal consistency
`get_plane_normals_differentiable()` (신규) vs `get_plane_geometry()` (기존) 법선 최대 차이: 0.0.

### Warmup schedule 수식 산술
경계값 계산이 의도대로인지 확인하는 단위 테스트. 실제 trainer 통합 동작은 Phase 3-B (d) 학습 시 TensorBoard `mutual/lambda_effective` 곡선으로 확인.

| iter | progress | lambda_m |
|------|----------|----------|
| 0 | 0% | 0.0 |
| 1650 | 33% | 0.0 |
| 2500 | 50% | 0.025 |
| 3350 | 67% | 0.050 |
| 5000 | 100% | 0.050 |

## 생성/수정 파일 목록

### 수정
| 파일 | 유형 | 핵심 변경 |
|------|------|----------|
| `planarsplat/utils/loss_util.py` | 수정 | `mutual_loss()` 추가 — L_mutual 3항 구조 (p_wall·L_vert + p_roof·L_slope + p_ground·L_horiz). mode별 detach 제어로 ablation (e)(f) 지원 |
| `planarsplat/net/net_planarSplatting.py` | 수정 | `get_plane_normals_differentiable()` — quat_to_rot()의 in-place ops를 우회하여 autograd 안전한 법선 계산 경로 제공 |
| `planarsplat/run/trainer.py` | 수정 | L_mutual 통합: config 읽기, warmup curriculum, gradient check at start, TensorBoard logging (loss/mutual, gradient norms) |
| `planarsplat/confs/base_conf_planarSplatCuda.conf` | 수정 | L_mutual 기본값 추가 (lambda_mutual=0 — 기존 동작 보존) |
| `docs/EXPERIMENT_PLAN.md` | 수정 | Go/No-Go 기준 명확화, 전 Phase 프롬프트에 REPORT 템플릿 참조 추가 |

### 추가
| 파일 | 유형 | 핵심 변경 |
|------|------|----------|
| `scripts/gradient_check_phase3a.py` | 신규 | Go/No-Go #1, #2 자동화 (14개 테스트) |
| `scripts/smoke_test_phase3a.py` | 신규 | Go/No-Go #3, #4 자동화 |
| `utils_demo/ablation_{a..g}_*.conf` | 신규 | Phase 3-B용 7개 ablation config |
| `results/phase3a/gradient_check.json` | 신규 | #1, #2 결과 수치 |
| `results/phase3a/smoke_test.json` | 신규 | #3, #4 결과 수치 |

## 이슈 및 해결

1. **quat_to_rot() in-place ops와 autograd**: `R[:, 0, 0] = expr` 패턴이 autograd에서 안전한지 불확실 → `get_plane_normals_differentiable()` 신규 구현으로 우회. rotation matrix 3번째 열만 직접 계산 (in-place op 없음). `get_plane_geometry()` 결과와 완전 일치 확인 (diff=0)
2. **rot_q_xyAxis gradient ≈ 0**: 정상 동작. 면내 회전(xy축 회전)은 법선 방향을 변경하지 않으므로, L_mutual에서 gradient가 없는 것이 정확. rot_q_normal_wxy에만 gradient 집중 (0.462)

## 다음 Phase
Phase 3-B: 7개 ablation config로 실제 학습 실행. 핵심 비교는 (c) Independent vs (d) Joint — L_mutual 유무만 다르므로 양방향 상호보강 효과를 직접 증명. 방향성 비교 (d) vs (e) vs (f)로 시너지 검증.

Phase 3-A가 제공하는 전제조건: L_mutual의 gradient가 f_i와 R_i 양방향으로 흐르고, mode별 detach가 정확하므로, Phase 3-B에서 지표 변화를 L_mutual에 귀속시킬 수 있다.
