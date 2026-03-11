# Phase 3-B'-Step1: L_photo 구현 및 baseline 확립

## 수행 일시
2026-03-11

## 수행 작업 요약

Phase 3-B에서 L_photo 미포함 상태의 ablation이 완료되었으나, 지도교수 피드백에 따라 L_photo를 기본 설정에 포함하기로 결정하였다. 본 Step에서는 PlanarSplatting에 L_photo를 추가 구현하고, L_photo 포함 baseline (c')을 확립한다. 이 (c') checkpoint가 Phase 4 (CityGML 변환)와 Phase 3-B'-Step2 (L_mutual 재실험)의 기반이 된다.

## 구현 사항

### 렌더링 방안: NUM_CHANNELS 7ch 단일 forward
- `config.h`: NUM_CHANNELS 4→7 (3 RGB + 4 semantic)
- `colors_precomp = cat([rgb(N,3), semantic(N,4)], dim=-1)` → 단일 rasterizer call
- 출력 `rendered_all(7,H,W)` → `[:3]`=RGB, `[3:]`=semantic
- allmap(기하) 기존과 동일

### 추가된 파라미터
- `_plane_colors_rgb` (N,3) nn.Parameter, lr=0.005, sigmoid activation → [0,1]
- 초기화: gray(0.5), learnable when `enable_photo=True`
- density control: split/prune에서 semantic features와 동일 패턴으로 동기

### L_photo 구현
- `photo_loss()` in loss_util.py: `(1-0.2)*L1 + 0.2*(1-SSIM)`
- SSIM: window_size=11, Gaussian kernel, channel-wise convolution
- Config: `enable_photo`, `lambda_photo`, `lr_color`

## 정량 지표

| 지표 | (c) Independent | (c') Independent+Photo | Δ | GT 기준 |
|------|----------------|----------------------|---|---------|
| Depth MAE | 0.0274 ± 0.016 | **0.0260 ± 0.012** | **-0.0014** | MVS |
| Normal cos | 0.7816 ± 0.032 | 0.7817 ± 0.033 | +0.0001 | MVS |
| mIoU | 0.7859 ± 0.092 | 0.7833 ± 0.096 | -0.0026 | seg_v10 |
| PSNR | N/A | **15.00 ± 1.45 dB** | — | GT RGB |
| Planes | 2605 | 2653 | +48 | — |

### 해석
- **Depth MAE 5.1% 개선**: L_photo가 RGB 텍스처 매칭을 통해 프리미티브 위치를 추가 보정.
- **Normal cos 동등**: 법선은 L_normal이 주로 결정하며, L_photo의 간접 효과는 미미.
- **mIoU 미미한 하락 (-0.003)**: 통계적으로 유의하지 않은 수준.
- **PSNR 15 dB**: 학습 가능 color 도입 첫 결과. 프리미티브 수(~2700)가 적어 세밀한 텍스처 표현에 한계.

## 정성적 결과

9개 뷰(View 0, 12, 25, 37, 50, 62, 75, 87, 99)에 대해 Depth, Normal, RGB(GT|Rendered), Semantic(GT|Predicted) 이미지를 `images/`에 저장. 3D PLY는 normal/class/rgb 3종.

### RGB 렌더링 (L_photo 효과의 핵심 산출물)

- **View 00** (`rgb_view00.png`): 교차로 상공 oblique view. Rendered RGB가 건물 외벽의 회색 톤, 도로 표면의 색상을 대략적으로 재현. 다만 프리미티브 경계가 뚜렷한 사각형 패치로 보이며, ~2700개 프리미티브로는 세밀한 텍스처(횡단보도, 간판) 표현 불가. 검은 영역(프리미티브 미커버 부분) 존재.
- **View 04** (`rgb_view04.png`): near-nadir view, 다수 건물 지붕. 지붕 색상(파란색 옥상)이 GT와 일치. 도로의 붉은 포장도 대략 재현. 건물 간 틈새에서 검은 빈 영역이 관찰됨 — 프리미티브가 모든 표면을 커버하지 못하는 한계.
- **View 07** (`rgb_view07.png`): 고층 건물(UBI) 측면 oblique. 건물 전면의 대략적 색조는 일치하나, 창문/간판 등 세부 텍스처는 균일 색상으로 평균화됨. 지면의 붉은 도로 표시가 대형 프리미티브에 반영되어 있음.

### Depth 렌더링

- **View 04** (`depth_view04.png`): GT(MVS) 대비 전반적 깊이 분포 일치. 건물 높이 차이, 도로-건물 경계의 깊이 전환이 정확. GT에 비해 렌더링 결과가 더 매끄럽고 노이즈가 적음 — 평면 프리미티브의 구조적 장점 (자동 denoising 효과).

### Normal 렌더링

- **View 04** (`normal_view04.png`): 지붕 면의 수평 법선(연두색)과 벽면의 수직 법선(갈색/적색) 구분이 명확. GT 대비 프리미티브 법선이 더 균일하고 면 단위로 일관됨 — 평면 가정에 의한 법선 정규화 효과.

### Semantic 렌더링

- **View 04** (`semantic_view04.png`): GT에서 검은 영역(BG, 51%)이 넓으나, 예측에서는 전체 영역을 분류. Roof(빨강)이 건물 상단에, Wall(파랑)이 측면에, Ground(회색)가 도로에 정확히 배치. GT가 없는 ambiguous 영역에서도 기하학적으로 합리적인 분류가 관찰됨 — 3D 프리미티브의 multi-view 일관성 효과.
- **View 07** (`semantic_view07.png`): 고층 건물 측면이 Wall(파랑)로 정확히 분류. 지붕 상단이 Roof(빨강). GT에서 누락된 건물 측면 영역도 예측에서 Wall로 채워짐.

### 3D PLY 시각화

- `cprime_normal.ply` — 법선 색상. 벽면(수직)과 지붕(수평) 법선 방향의 구분 명확.
- `cprime_class.ply` — 클래스 색상. Roof(빨강)/Wall(파랑)/Ground(회색) 공간 분포 확인. 건물 상단에 Roof, 측면에 Wall, 지면에 Ground가 일관되게 배치.
- `cprime_rgb.ply` — 학습된 RGB 색상. 건물별 색조 차이, 지붕/도로 색상 구분 확인. 프리미티브 단위의 균일 색상이므로 세밀한 텍스처는 없으나 건물 식별에는 충분.

## Go/No-Go
- [x] Go
- 근거: L_photo 구현 성공. (c') baseline 확립. Depth MAE 5.1% 개선, Normal cos/mIoU 유지. 이 (c') checkpoint가 Phase 4(CityGML 변환)와 Phase 3-B'-Step2(L_mutual 재실험)의 전제조건을 제공한다.

## 생성/수정 파일 목록

| 파일 | 유형 | 핵심 변경 |
|------|------|----------|
| `config.h` | 수정 | NUM_CHANNELS 4→7 |
| `net_planarSplatting.py` | 수정 | `_plane_colors_rgb`, bg 7ch, forward 7ch concat |
| `net_wrapper.py` | 수정 | optimizer/split/prune colors_rgb 동기 |
| `loss_util.py` | 수정 | `photo_loss()`, `_ssim()` 추가 |
| `trainer.py` | 수정 | L_photo 통합, 7ch split, TB logging |
| `evaluate.py` | 수정 | 7ch 대응 (semantic=last 4ch, RGB=first 3ch) |
| `utils_demo/phase3b_prime_c.conf` | 신규 | (c') config |

## 이슈 및 해결

1. **CUDA GPU 미인식**: CUDA rasterizer rebuild 후 컨테이너에서 GPU가 인식되지 않음. `docker compose restart planarsplat`으로 해결.
2. **evaluate.py 7ch 미대응**: NUM_CHANNELS 변경 후 semantic argmax가 7ch 전체에서 수행되어 mIoU=0. `rendered_rgb[C-4:]`로 마지막 4ch만 사용하도록 수정.
3. **render_views.py 7ch 미대응**: 동일 이슈. semantic/RGB 채널 분리 및 rendered RGB 비교 기능 추가.

## 산출물
- `results/phase3b_prime/eval_cprime.json` — 정량 평가 결과
- `results/phase3b_prime/cprime_normal.ply` — 법선 색상 3D PLY (2653 primitives)
- `results/phase3b_prime/cprime_class.ply` — 클래스 색상 3D PLY (roof/wall/ground)
- `results/phase3b_prime/cprime_rgb.ply` — 학습된 RGB 색상 3D PLY
- `results/phase3b_prime/images/` — 9개 뷰 렌더링 (depth/normal/rgb/semantic)

## 다음 Phase
- **Phase 4 프로토타입**: (c') checkpoint로 CityGML 변환
- **Phase 3-B'-Step2**: L_photo 포함 상태에서 L_mutual 재실험
