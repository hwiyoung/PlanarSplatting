# Phase 1 Normal Source 비교: Finite-Diff vs MVS Native

## 배경

### Planar 읽기 버그 발견 및 수정
COLMAP의 `Mat` 클래스는 다채널 데이터를 **planar layout**으로 저장한다:
```
data[slice * W * H + row * W + col]
```
즉 채널이 가장 느리게 변한다 (nx 전체 → ny 전체 → nz 전체).

기존 `read_colmap_array()`는 이를 interleaved로 읽었다:
```python
data.reshape(h, w, c)  # WRONG: assumes interleaved (nx,ny,nz,nx,ny,nz,...)
```

수정 후:
```python
data.reshape(c, h, w).transpose(1, 2, 0)  # CORRECT: planar → (h,w,c)
```

### 버그의 증거
interleaved 읽기 시 3채널(nx, ny, nz)의 통계가 소수점 4자리까지 동일:
- mean: 0.0046, 0.0046, 0.0045
- std: 0.2872, 0.2872, 0.2871

이는 물리적으로 불가능하며, 채널 간 데이터가 섞여 있음을 의미한다.
수정 후: nx≈0, ny≈+0.26, nz≈-0.55 (물리적으로 합리적).

### 버그의 영향 범위
| 항목 | 영향 |
|------|------|
| Phase 1 (이전) | MVS native normal로 학습 → 깨진 데이터로 supervision → 재학습 필요 |
| Phase 2-A segmentation | 영향 없음 (depth-derived normals 사용, MVS normal 미참조) |
| Phase 2-B (코드) | 영향 없음 (normal 소스와 무관) |
| Phase 2-C (이전) | 이전 Phase 1 기반 → 재학습 필요 |
| compare_normals.py (이전) | MVS vs FD 67° 차이 → 실제 25° (수정 후) |

## 실험 설계

### 비교 대상
| 소스 | 설명 | Coverage |
|------|------|----------|
| **Finite-Diff (FD)** | MVS depth에서 forward difference로 유도. 2차 파생. | ~57% |
| **MVS Native** | COLMAP PatchMatch stereo 직접 출력. 1차 추정. planar 읽기 수정 적용. | ~91% |

### 학습 조건
- 동일 설정: 5000 iter, base_conf + demo.conf, 100 views
- 차이점: `input_data.pth`의 normal만 다름 (depth는 동일한 MVS depth)
- `colmap_to_ps.py --normal_source finite_diff` vs `--normal_source mvs`

### 평가 방법: Cross-Evaluation
각 모델의 렌더링 결과를 **양쪽 GT 모두**에 대해 평가하여 2×2 행렬 생성.
Self-evaluation(대각선)은 자기 GT에 유리하므로, cross-evaluation(비대각선)도 함께 비교.

## 결과

### Normal Cosine Similarity (2×2)

|  | FD GT | MVS GT |
|---|---|---|
| **FD model** | **0.7281** | 0.7761 |
| **MVS model** | 0.7264 | **0.7811** |

- FD GT 기준: FD model 승 (+0.0017)
- MVS GT 기준: MVS model 승 (+0.0050)
- **열(GT) 차이 ≈ 0.05 >> 행(model) 차이 < 0.005**

### Depth MAE (2×2)

|  | FD GT mask | MVS GT mask |
|---|---|---|
| **FD model** | 0.0246 | 0.0262 |
| **MVS model** | **0.0229** | **0.0245** |

- MVS model이 모든 조건에서 depth MAE 우수 (-6.9% ~ -6.5%)
- 열 간 차이는 valid_mask 범위 차이 (MVS coverage 91% vs FD 57%)

### 모델 요약

| 지표 | FD model | MVS model | 차이 |
|------|----------|-----------|------|
| Plane 수 | 2,678 | 2,665 | -13 |
| Depth MAE (FD mask) | 0.0246 | **0.0229** | **-6.9%** |
| Normal cos (FD GT) | **0.7281** | 0.7264 | -0.2% |
| Normal cos (MVS GT) | 0.7761 | **0.7811** | +0.6% |

## 분석

### 핵심 발견
1. **Normal cos는 사실상 동일**: 어떤 GT로 평가하든 두 모델 간 차이 < 0.5%.
   렌더링된 법선은 학습 supervision과 거의 무관하게 동일 수준으로 수렴한다.

2. **Depth MAE는 MVS model이 일관적으로 우수**: 동일 mask(동일 픽셀)에서 -6.9%.
   이는 masking artifact가 아닌 실제 기하학적 개선이다.

3. **GT 소스 자체가 점수에 더 큰 영향**: model 간 차이(< 0.005)보다
   GT 간 차이(≈ 0.05)가 10배 이상 크다.

### Depth 개선의 원인 추정
MVS native coverage(91%) vs FD coverage(57%) → ~60% 더 많은 픽셀에서 normal loss gradient 수신.
Depth 개선은 "normal 품질 차이"가 아니라 **"supervision 범위 차이"**에서 비롯될 가능성이 높다.

### 의미
- 학습 normal 소스 선택은 normal reconstruction 품질에 거의 영향 없음
- 단, coverage 차이를 통해 depth reconstruction에 간접적 영향
- MVS native의 장점은 "더 정확한 normal"이 아니라 "더 넓은 supervision"

## 결정

**MVS native normal 채택** (학습 supervision용)

### 근거
1. Depth MAE -6.9% 개선 (일관적, 모든 평가 조건)
2. Normal cos 동등 (하방 리스크 없음)
3. 높은 coverage (91% vs 57%) → 더 많은 supervision signal
4. `--normal_source mvs` 옵션 이미 구현 완료

### 파일 위치
- FD 실험: `planarSplat_ExpRes/seongsu_normal_test_fd/`
- MVS 실험: `planarSplat_ExpRes/seongsu_normal_test_mvs/`
- Cross-eval 스크립트: `scripts/cross_eval_normals.py`
- Normal 소스 옵션: `scripts/colmap_to_ps.py --normal_source {auto|mvs|finite_diff}`

## 정성적 비교: Buggy vs Corrected MVS Reading

아래 이미지는 동일 `.geometric.bin` 파일을 interleaved(buggy)와 planar(corrected) 두 방식으로 읽어 비교한 것이다.
행 구성: Finite-Diff | MVS-buggy(interleaved) | MVS-corrected(planar)

![View 0](images/normal_compare_view000.png)
![View 15](images/normal_compare_view015.png)
![View 40](images/normal_compare_view040.png)
![View 60](images/normal_compare_view060.png)
![View 80](images/normal_compare_view080.png)

**Buggy 채널 통계 (대표: View 0)**:
- nx: mean=-0.4477, std=0.3739
- ny: mean=-0.4813, std=0.2976
- nz: mean=-0.5501, std=0.1924
→ 채널 간 mean/std가 유사 (데이터 혼합의 증거)

**Corrected 채널 통계 (대표: View 0)**:
- nx: mean=-0.0745, std=0.3113
- ny: mean=0.0605, std=0.6554
- nz: mean=-0.6607, std=0.1670
→ 채널 간 명확한 차이 (물리적으로 합리적: nz<0 = 카메라 향함)

## Planar 읽기 버그가 Phase 2-A 진단에 미친 영향

Phase 2-A REPORT.md에 기록된 "MVS PatchMatch normals 퇴화" 진단:
> "MVS PatchMatch normals는 텍스처 없는 외벽에서 카메라 Z축 방향으로 퇴화한다"

이 관찰은 planar 읽기 버그가 있는 상태에서 수행되었다. 수정 후 MVS native normals가
실제로 textureless facade에서 퇴화하는지는 재검증이 필요하다.

단, Phase 2-A segmentation은 MVS native normals를 사용하지 않고 smoothed depth-derived
normals를 내부 계산하므로, 버그 수정이 기존 seg_maps에 영향을 미치지 않는다.
