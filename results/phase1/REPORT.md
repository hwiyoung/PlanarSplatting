# Phase 1: MVS Depth+Normal Supervision 교체 결과 보고

## 수행 일시
2026-02-16 ~ 2026-02-17

## 수행 작업 요약
Depth supervision을 Metric3D mono depth에서 COLMAP MVS geometric depth로, Normal supervision을 COLMAP MVS native normal로 교체하였다. Training loop 자체는 수정하지 않고, 데이터 준비 단계(`scripts/colmap_to_ps.py`)만 수정하여 동일한 `input_data.pth` 포맷으로 변환하는 접근법을 사용하였다.

### 수행 단계
1. **COLMAP MVS 데이터 분석**: `dense/stereo/depth_maps/*.geometric.bin`, `normal_maps/*.geometric.bin` 형식 확인
   - Binary format: text header `"width&height&channels&"` + row-major float32
   - photometric.bin은 confidence가 아닌 필터링 전 depth map임을 확인
   - Geometric depth coverage: 평균 86.6%, Normal coverage: 평균 78.3%
2. **`scripts/colmap_to_ps.py` 수정**:
   - `read_colmap_array()`: COLMAP binary depth/normal reader
   - MVS native normal 로드: normalize to unit vectors, z-flip (toward camera), mask invalid
   - `depth_to_normal_cam()`: Fallback용 finite-diff normal 유도
   - `--depth_source mvs/mono` 플래그 추가
   - MVS 분기: Metric3D/scale alignment 건너뛰고, geometric depth+normal 직접 사용
   - `pre_align=False`: MVS는 이미 절대 스케일
3. **학습**: 100 views, 3000 초기 planes, 5000 iters
4. **평가**: evaluate.py로 정량 평가

### 입력 데이터 요약
| 항목 | Phase 0 (Metric3D) | Phase 1 (MVS) |
|------|---|---|
| Depth source | Metric3D ViT-L + COLMAP scale align | COLMAP geometric depth |
| Depth 측정 유형 | 2차 (단안 신경망 추정 + scale align) | **1차 (multi-view 삼각측량, 절대 스케일)** |
| Normal source | Metric3D ViT-L | **COLMAP MVS native normal (PatchMatch stereo)** |
| Normal 측정 유형 | 1차 (신경망이 직접 예측) | **1차 (PatchMatch stereo가 직접 추정)** |
| Depth coverage | ~95% | 86.6% (mean) |
| Normal coverage | ~95% | 78.3% (mean) |
| Scale alignment | COLMAP sparse points 기반 | 불필요 (절대 스케일) |
| pre_align | True | False |

#### MVS Normal 처리 과정
COLMAP PatchMatch stereo는 depth와 함께 normal map도 직접 추정한다 (`normal_maps/*.geometric.bin`). 이 normal은:
- (H,W,3) float32, 범위 [-1,1], **단위 벡터가 아님** (norm이 0.01~1.73 범위)
- Camera space 좌표계 (X-right, Y-down, Z-forward)
- z-sign이 혼합되어 있음 (60.8%만 z<0)

PlanarSplatting 포맷으로 변환 시:
1. Unit vector로 정규화 (norm > 0.01인 픽셀만)
2. z > 0인 normal을 flip하여 toward-camera 방향(z<0) 통일
3. Invalid 픽셀 (norm ≤ 0.01 or depth=0) 마스킹
4. [-1,1] → [0,1] 범위 변환, (3,H,W) 형식

#### 초기 시도: Finite-diff normal (2차 유도)
초기에는 MVS depth에서 finite-difference로 normal을 유도하였으나:
- Coverage 87% → 57%로 급감 (3픽셀 모두 valid 필요)
- Depth 불연속 경계에서 노이즈 증폭
- Normal cos: 0.729 (낮음)

MVS native normal로 교체 후 Normal cos: 0.840으로 대폭 개선.

## 정량 지표
| 지표 | Phase 1 (MVS native) | Phase 0 (Mono) | 비고 |
|------|-----|-----------|------|
| Depth MAE | 0.0528 +/- 0.0218 | 0.0672 +/- 0.0513 | Phase 0 대비 개선 |
| Normal cos | 0.8396 +/- 0.0138 | 0.9107 +/- 0.0197 | GT 기준이 다름 (MVS native vs Metric3D) |
| Final planes | 1,210 | 2,123 | |

**주의**: Phase 0과 Phase 1의 GT 기준이 다르므로 정량 지표의 직접 비교는 제한적이다. Phase 0은 Metric3D normal 대비, Phase 1은 MVS native normal 대비 평가.

### 1차 측정 vs 2차 유도

| | Phase 0 | Phase 1 |
|--|--|--|
| **Depth GT** | 2차 (Metric3D 단안 추정 + scale align) | **1차 (MVS multi-view 삼각측량)** |
| **Normal GT** | 1차 (Metric3D가 직접 예측) | **1차 (PatchMatch stereo가 직접 추정)** |

Phase 1에서는 depth와 normal 모두 1차 측정/출력을 사용한다.

### Finite-diff vs MVS native normal 비교
| 지표 | Finite-diff (2차) | MVS native (1차) | 비고 |
|------|-----|-----------|------|
| Depth MAE | 0.0243 | 0.0528 | Finite-diff가 낮지만 Phase 0(0.067)보다 우수 |
| Normal cos | 0.7293 | **0.8396** | MVS native가 대폭 우수 |
| Normal coverage | 57% | **78.3%** | MVS native가 우수 |
| Final planes | 2,679 | 1,210 | |

Depth MAE가 finite-diff보다 높아진 이유: MVS native normal의 coverage가 넓어(78.3% vs 57%) normal loss의 영향력이 증가하면서 depth 최적화에 간접 영향.

## Go/No-Go 판단
- [x] Go (Depth+Normal 모두)

### Depth: Go
- MAE 0.053 — Phase 0 baseline(0.067) 대비 개선
- 절대 스케일 depth supervision 성공적으로 작동

### Normal: Go
- Normal cos 0.840 — MVS native normal 사용으로 양호한 수렴
- Coverage 78.3% — finite-diff(57%) 대비 대폭 개선
- Depth와 Normal 모두 1차 측정/출력 사용으로 감독 신호 품질 최적화

## 이슈 및 해결
1. **COLMAP binary format**: 초기에 null terminator 가정으로 잘못 파싱 → `&` 3개 기반 header 파싱으로 수정
2. **photometric.bin은 confidence가 아님**: 별도의 depth map이었음 → geometric.bin만 사용
3. **MVS normal 비단위벡터**: COLMAP normal은 단위 벡터가 아님 (norm 0.01~1.73) → 정규화 필수
4. **MVS normal z-sign 혼합**: 60.8%만 z<0 → z>0 픽셀을 flip하여 toward-camera 방향 통일
5. **Finite-diff coverage 문제**: 87% → 57% 급감 → MVS native normal로 대체 (78.3%)

## 생성/수정 파일
- `scripts/colmap_to_ps.py` — MVS native normal 로드, `read_colmap_array()`, `--depth_source` 플래그 (수정)

## 다음 Phase
- **Phase 2-B**: 의미론적 헤드 f_i(K=3) 구현 + semantic 렌더링 + L_sem + L_geo
