# Phase 2-B: Semantic Head 구현 결과 보고

## 수행 일시
2026-02-19

## 수행 작업 요약
PlanarSplatting의 각 평면 프리미티브에 4차원 의미론적 특징 벡터 `f_i ∈ R^4`를 추가하고, `L_sem`(CrossEntropyLoss)과 `L_geo`(Normal Consistency Loss)를 구현했다. CUDA rasterizer의 alpha-blend를 통해 f_i가 픽셀 수준으로 렌더링되며, seg_map GT와 비교하여 학습된다.

### 설계 결정
- **Semantic 렌더링**: Option A (raw f_i → rasterizer alpha-blend → softmax → CrossEntropyLoss)
- **NUM_CHANNELS**: 3 → 4 (CUDA rasterizer compile-time 상수)
- **f_i 초기화**: zeros (uniform logits, 4-class equal prior)
- **L_sem**: `ignore_index=0` (background 픽셀은 loss에서 제외)
- **L_geo**: rendered normal vs depth-derived normal (finite-diff, depth discontinuity 제외)
- **Gradient 격리**: L_sem → f_i only (geometry params에 gradient 전달 안함). L_geo → geometry only.

## Gradient Check 결과

### Test 1: Simulated alpha-blend (순수 PyTorch)
| 지표 | 값 |
|------|-----|
| L_sem | 1.3863 (= ln(4), uniform logits 기대값) |
| ∂L_sem/∂f_i | 0.5000 (non-zero ✓) |
| ∂L_sem/∂R_i | 0.0000 (zero ✓) |

### Test 2: Full CUDA Rasterizer (학습된 체크포인트, 2123 planes)
| 지표 | 값 |
|------|-----|
| L_sem | 1.3852 |
| ∂L_sem/∂f_i (abs sum) | 1.3920 (non-zero ✓) |
| Non-zero gradient primitives | 109/2123 (visible + non-bg pixels 대상) |
| ∂L_sem/∂rot_normal_wxy | 0.0 ✓ |
| ∂L_sem/∂rot_xyAxis_w | 0.0 ✓ |
| ∂L_sem/∂rot_xyAxis_z | 0.0 ✓ |
| ∂L_sem/∂center | 0.0 ✓ |
| ∂L_sem/∂radii_xy_p | 0.0 ✓ |
| ∂L_sem/∂radii_xy_n | 0.0 ✓ |

### Test 3: L_geo gradient direction
| 지표 | 값 |
|------|-----|
| L_geo | 1.9580 |
| ∂L_geo/∂f_i | 0.0 ✓ (f_i에 영향 없음) |
| ∂L_geo/∂rot_normal_wxy | 0.4014 (non-zero ✓) |
| ∂L_geo/∂center | 4.5483 (non-zero ✓) |

### 결론
- **L_sem은 f_i만 학습**: 모든 기하학적 파라미터(R_i, center, radii)에 gradient가 0
- **L_geo는 기하학만 학습**: f_i에 gradient 0, geometry params에 non-zero
- Phase 3-A의 L_mutual이 양방향 gradient를 열기 전까지 두 경로가 완전 분리됨

## L_geo (Normal Consistency Loss) 구현 확인
- Rendered depth → 3D unprojection → finite-diff cross product → normal 유도
- Rendered normal과 depth-derived normal 간 `1 - cos(θ)` loss
- Depth discontinuity 제외: `|depth_grad| < depth × 0.05` threshold
- 경계 1픽셀 제외, depth > 0.01 요구
- 학습된 모델에서 L_geo = 1.958 (rendered normal이 depth-consistent하지 않은 영역 존재)

## Density Control 동작 확인

### Prune
- 50 planes → threshold=100 prune → 0 planes
- `_plane_semantic_features` shape: `(50, 4)` → `(0, 4)` ✓
- `check_model()` 통과 ✓

### Split (split_all)
- 50 planes → split_all → 200 planes (4x: x-split + y-split + xy-split)
- `_plane_semantic_features` shape: `(50, 4)` → `(200, 4)` ✓
- 부모 f_i가 자식에게 복사됨 (split 후 동일 특징 유지)
- `check_model()` 통과 ✓

## 기존 기능 보존 확인

### enable_semantic=False
- `_plane_semantic_features.requires_grad = False` ✓
- 기존 학습 코드 경로 변경 없음 (L_sem, L_geo는 lambda=0이면 계산 스킵)
- forward에서 `colors_precomp = torch.rand(N, 4)` (기존과 동일 패턴, 채널 수만 3→4)

### Backward Compatibility
- Phase 0/1 체크포인트(semantic features 없음) 로드 시 zeros 자동 주입 ✓
- `trainer_util.py resume_model()` 및 `evaluate.py load_model_and_data()` 모두 처리

## 수정/추가 파일 목록

### CUDA (rasterizer 수정)
| 파일 | 변경 내용 |
|------|----------|
| `submodules/diff-rect-rasterization/cuda_rasterizer/config.h` | `NUM_CHANNELS` 3→4 |
| `submodules/diff-rect-rasterization/cuda_rasterizer/backward.cu` | `dL_dcolors` atomicAdd 추가 (color gradient 누락 수정) |

### 핵심 코드
| 파일 | 변경 내용 |
|------|----------|
| `planarsplat/net/net_planarSplatting.py` | f_i parameter, bg 4ch, enable_semantic 분기, forward colors_precomp |
| `planarsplat/run/net_wrapper.py` | optimizer에 f_i 추가, split/prune/densify에 f_i 동기 처리 |
| `planarsplat/run/trainer.py` | L_sem/L_geo 계산, TensorBoard semantic 로깅, semantic 이미지 시각화 |
| `planarsplat/data_loader/scene_dataset_demo.py` | seg_maps 로딩 (cv2.IMREAD_GRAYSCALE, INTER_NEAREST resize) |
| `planarsplat/utils/loss_util.py` | `semantic_loss()`, `normal_consistency_loss()` 추가 |

### Config
| 파일 | 변경 내용 |
|------|----------|
| `planarsplat/confs/base_conf_planarSplatCuda.conf` | enable_semantic, semantic_num_classes, lr_semantic, lambda_sem, lambda_geo |
| `utils_demo/demo.conf` | lambda_sem=0.1, lambda_geo=0.1 |

### 유틸리티/스크립트
| 파일 | 변경 내용 |
|------|----------|
| `planarsplat/utils/trainer_util.py` | resume_model()에 semantic features 호환성 처리 |
| `scripts/visualize_primitives.py` | `--color_by class` 옵션 (roof=red, wall=blue, ground=gray) |
| `scripts/evaluate.py` | `--metrics semantic_miou` 옵션 (per-class IoU + mIoU) |
| `scripts/gradient_check_phase2b.py` | Phase 2-B gradient check 스크립트 (6개 자동 테스트) |

## 이슈 및 해결

### 이슈 1: CUDA backward pass에 color gradient 누락
- **증상**: `∂L_sem/∂f_i = 0` (rasterizer 통과 시). Simulated alpha-blend에서는 정상.
- **원인**: `diff-rect-rasterization/cuda_rasterizer/backward.cu`의 `renderCUDA` kernel에서 `dchannel_dcolor = alpha * T`를 계산하지만, `dL_dcolors` buffer에 `atomicAdd`를 하지 않음. PlanarSplatting 원저자가 색상을 학습하지 않으므로(random color) gradient 구현을 생략한 것.
- **해결**: `backward.cu` line 189 이후에 color gradient accumulation 추가.
- **의도적 설계**: Color → alpha gradient path는 추가하지 않음 → L_sem이 geometry에 영향을 주지 않도록 격리. Phase 3-A의 L_mutual은 rasterizer를 거치지 않으므로(per-primitive 직접 계산) 이 격리가 올바름.

### 이슈 2: nn.Parameter requires_grad 설정
- **증상**: `enable_semantic=False`인데 `_plane_semantic_features.requires_grad=True`
- **원인**: `nn.Parameter(tensor.requires_grad_(False))` — tensor의 requires_grad 설정이 nn.Parameter 생성자에 의해 True로 덮어씌워짐
- **해결**: `nn.Parameter(tensor, requires_grad=False)` 로 생성자 인자 사용

### 이슈 3: Freshly initialized model에서 rasterizer test 불가
- **증상**: sphere 초기화 모델에서 visible pixels = 0 (카메라 frustum 밖)
- **해결**: gradient check를 2단계로 분리. (1) Simulated alpha-blend로 gradient 방향 검증, (2) 학습된 체크포인트로 full pipeline 검증.

## Go/No-Go 판정
모든 구현 검증 항목 통과:
- [x] ∂L_sem/∂f_i ≠ 0 (simulated + rasterizer)
- [x] ∂L_sem/∂R_i = 0 (모든 geometry params)
- [x] ∂L_geo/∂geometry ≠ 0, ∂L_geo/∂f_i = 0
- [x] Density control (split/prune) 동작 정상
- [x] Backward compatibility
- [x] enable_semantic=False 기존 동작 보존

**→ Phase 2-C (L_sem 독립 학습) 진행 가능**
