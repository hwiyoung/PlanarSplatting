# 실험 계획

## 이 문서의 역할
이 문서는 Claude Code가 자동으로 읽지 않는다 (자동으로 읽히는 것은 CLAUDE.md뿐).
각 Phase의 프롬프트를 Claude Code 채팅창에 복붙하여 사용한다.
Docker 맥락은 CLAUDE.md의 "개발 환경" 섹션(자동 참조)과 각 프롬프트의 "컨테이너 내부에서 작업이야"(복붙 시 전달)로 전달된다.

## Phase 의존성

```
Phase 0-Setup ─→ Phase 0 (SfM/MVS) ─→ Phase 1 (MVS depth) ─→ Phase 2-A (Segmentation)
                                                                  ↓
                                                   Phase 2-B (f_i) ─→ Phase 2-C (L_sem)
                                                                            ↓
                                                              Phase 3-A (L_mutual) ─→ Phase 3-B (Ablation)
                                                                                            ↓
                                                                                  Phase 3-C (L_photo, 선택적)
                                                                                            ↓
                                              Phase 4 (CityGML) ←── 코드는 병렬 작성 가능, 실행은 Phase 3 결과 필요
```

### COLMAP 단계와 Phase 의존성
| COLMAP 단계 | 완료 시 가능한 작업 |
|---|---|
| mapper (sparse) | Phase 0 학습 (sparse 기반 초기화). 변환 스크립트 코드 작성은 COLMAP 실행 중에도 가능. |
| patch_match_stereo + stereo_fusion | Phase 1 (MVS depth supervision) |

---

## 데이터셋

| 데이터셋 | 출처 | 용도 | 비고 |
|----------|------|------|------|
| 성수동 드론 이미지 | 자체 촬영 | 주요 실험 (Phase 0~4) | 180장, oblique, 70m, 2048x1365 |
| UrbanScene3D | ECCV 2022 | 추가 평가 (다양한 도시 환경) | 드론 이미지 + GT 메쉬 |
| Building3D | ICCV 2023 | 추가 평가 (다양한 건물 유형) | 160K+ 건물 모델 |
| ScanNet/ScanNet++ | — | PlanarSplatting 재현성 검증 | 실내 데이터셋 (원논문 기준) |

현재 Phase에서는 성수동 데이터로 전체 파이프라인을 검증한 후, 논문 작성 시 공개 데이터셋으로 일반화 실험을 수행한다.

## 비교 방법

| 분류 | 방법 | 비고 |
|------|------|------|
| 기하학적 | City3D (Huang et al., 2022) | 사후 휴리스틱 분류 |
| 기하학적 | PolyFit (Nan & Wonka, 2017) | 면 선택 기반 |
| Neural | PlanarSplatting (CVPR 2025) | 원본, 의미론 없음 |
| Neural | PGSR (TVCG 2024) | 평면 기반 GS |
| Neural | 2DGS | Gaussian Splatting 표면 재구축 |
| Baseline | 기하학적 재구축 + 사후 휴리스틱 분류 | City3D 방식 |
| Baseline | 독립 semantic segmentation + 재구축 | 의미론-기하학 독립 처리 |

Phase 3-B ablation은 제안 방법 내부의 구성요소 효과 검증이며, 비교 방법 실험은 논문 작성 단계에서 수행한다.

## 평가 지표

### 학습 모니터링 지표 (Phase 0~3)
| 지표 | 용도 | Phase |
|------|------|-------|
| Depth MAE | 렌더링 깊이 정확도 | Phase 0+ |
| Normal cos | 렌더링 법선 정확도 | Phase 0+ |
| mIoU | 의미론적 분류 정확도 | Phase 2+ |

### 최종 평가 지표 (논문용, Phase 4+)
| 지표 | 용도 |
|------|------|
| Chamfer Distance | 메쉬 기하학적 정확도 (GT 대비) |
| Completeness / Accuracy | 포인트 클라우드 기반 기하 평가 |
| 평면 정확도 (Fidelity, L1-Chamfer) | 평면 프리미티브 정확도 |
| 면 단위 분류 Accuracy, IoU per class | 의미론적 분류 정확도 |
| val3dity 통과율 | CityGML 기하학적 유효성 |
| 건물 당 최적화 시간 | 효율성 |

학습 모니터링 지표는 Go/No-Go 판단에 사용하고, 최종 평가 지표는 비교 방법 실험 및 논문 표 작성에 사용한다.

---

## Phase 0-Setup: 모니터링 환경 구축

**목표:** TensorBoard 로깅, 시각화, 평가 스크립트 구축. Docker 환경 확인/보강.

**프롬프트:**
```
CLAUDE.md를 읽고, docs/EXPERIMENT_PLAN.md의 Phase 0-Setup을 진행해줘.
모든 작업은 Docker 컨테이너 내부 기준이야.

먼저 이 프로젝트의 코드 구조를 파악해줘:
1. 디렉토리/파일 구조 (tree)
2. train.py 학습 루프 위치와 구조
3. 손실 함수 정의/계산 위치
4. 프리미티브 파라미터 관리 (클래스, 텐서)
5. 렌더러의 alpha-blending 가중치 계산 위치
6. adaptive density control 위치
7. 데이터 로딩 형식
8. TensorBoard 연동 여부
9. Dockerfile, docker-compose.yml 설정 (서비스 이름, volumes, ports, GPU)
10. 웹 기반 뷰어가 있으면 그 구조도

파악 후:
- CLAUDE.md의 "코드 구조" 섹션을 실제에 맞게 수정
- CLAUDE.md의 "개발 환경" 섹션에서 docker-compose 서비스 이름, 볼륨 경로, 웹 뷰어 접속 방법을 실제에 맞게 수정
- docs/RESEARCH_CONTEXT.md의 "프리미티브 파라미터 전체 구조" 표를 실제 코드 기준으로 수정 (어떤 파라미터가 learnable이고 어떤 것이 아닌지 확인)

그 다음 모니터링 인프라 구축 (컨테이너 내부에서 동작하도록):
1. train.py에 TensorBoard 로깅 (이미 있으면 항목만 추가)
   - 매 iter: 각 손실 값, 프리미티브 수
   - 매 500 iter: 렌더링 이미지 (RGB vs GT, Depth 컬러맵, Normal 맵)
2. scripts/visualize_primitives.py
   - 체크포인트 → PLY export (Open3D headless, 데스크톱 GUI 띄우지 말 것)
   - --checkpoint, --color_by (normal/rgb), --export_ply
3. scripts/evaluate.py
   - 체크포인트 → Depth MAE, Normal cos → JSON
   - 기본 metrics: depth_mae, normal_cos (Phase 0~3-B)
   - PSNR은 --enable_photo (Phase 3-C) 시에만 활성화 — color가 random이므로 그 전에는 계산하지 않음
   - --checkpoint, --metrics, --output, --compare_with
4. 패키지 필요하면 Dockerfile에도 추가해줘

검증: 기존 사전실험 체크포인트가 있으면 그걸로 스크립트 작동 확인.
TensorBoard에 ssh 터널링으로 접속 가능한지 확인.

완료 후 CLAUDE.md의 진행 상태에서 Phase 0-Setup을 체크해줘.
```

**완료 노트:** TensorBoard 로깅, visualize_primitives.py, evaluate.py, 코드 리뷰 완료. `results/phase0_setup/` 참조.

---

## Phase 0: SfM/MVS 입력 확보

**목표:** VGGT 10% 정합 문제 해결, 공통 3D 입력 확보.

**입력:** 리사이즈된 드론 이미지 (2048x1365)

**Go/No-Go:**
| 지표 | Go | Retry | Switch |
|------|-----|-------|--------|
| 정합 이미지 | ≥ 100장 | 50~99 → 매칭 조정 | < 50 → Metashape |
| Depth MAE | ≤ 0.10 | 0.10~0.15 → iter 증가 | > 0.15 → 입력 변환 디버깅 |
| Normal cos | ≥ 0.85 | 0.80~0.85 → iter 증가 | < 0.80 → 법선 유도 확인 |
| 건물 형태 | Depth/Normal 렌더링에서 건물 윤곽+면 방향 식별 | 뭉개짐 → 초기화 조정 | — |

**COLMAP 실행** (수십 분~수 시간 소요, 돌려놓고 다른 작업 가능):
```bash
# 컨테이너 내부에서 (COLMAP이 설치된 경우) 또는 호스트에서 실행
colmap feature_extractor --database_path [경로]/db.db --image_path [이미지 경로]/
colmap exhaustive_matcher --database_path [경로]/db.db
colmap mapper --database_path [경로]/db.db --image_path [이미지 경로]/ --output_path [경로]/sparse/
colmap image_undistorter --image_path [이미지 경로]/ --input_path [경로]/sparse/0 --output_path [경로]/dense/
colmap patch_match_stereo --workspace_path [경로]/dense/
colmap stereo_fusion --workspace_path [경로]/dense/ --output_path [경로]/dense/fused.ply
```

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 0을 진행해줘. 컨테이너 내부에서 작업이야.

COLMAP 결과가 [실제 경로]에 있어.
scripts/colmap_to_ps.py를 만들어줘.

현재 코드의 데이터 로딩이 어떤 형식을 기대하는지 확인하고,
COLMAP 출력을 그 형식으로 변환해줘:
- cameras.bin, images.bin, points3D.bin → 카메라 포즈
- points3D → 초기 포인트 클라우드
- MVS depth maps → depth supervision
- --init_method colmap/vggt 플래그로 선택

변환 완료 후 컨테이너 내부에서 학습 실행하고 evaluate.py, visualize_primitives.py로 결과 확인.
렌더링 결과 이미지도 저장해줘 (Depth, Normal 각 2~3장 + GT RGB 참고용).
results/phase0/REPORT.md를 EXPERIMENT_PLAN.md 하단의 REPORT 템플릿에 따라 작성해줘.
CLAUDE.md 진행 상태 업데이트.
```

**완료 노트:** COLMAP 180장 정합, 100장 학습. Depth MAE=0.067, Normal cos=0.911. `results/phase0/` 참조.

---

## Phase 1: MVS Depth Supervision 교체

**목표:** depth supervision을 MVS 깊이로 교체. 기하가 정확해야 Phase 2 의미론 학습의 감독 신호가 정확함.

**입력:** Phase 0 출력. **COLMAP stereo_fusion까지 완료 필요.**

**Go/No-Go:**
| 지표 | Go | Retry |
|------|-----|-------|
| 손실 수렴 | 모두 우하향 | 발산 → λ 조정 |
| Depth MAE | Phase 0 대비 개선/동등 | 악화 → 마스킹 threshold |
| Normal | 벽 수직, 지붕 수평 (웹뷰어/PLY) | 무작위 → normal 유도 코드 확인 |

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 1을 진행해줘. 컨테이너 내부에서 작업이야.

먼저 현재 depth loss 관련 코드의 구조를 설명해줘.

변경:
1. Depth Anything → MVS depth (절대 깊이, L1 loss)
2. MVS 신뢰도 마스킹
3. L_normal도 MVS depth 유도 법선으로 교체
4. --depth_source mvs/mono 플래그
5. 패키지 필요하면 Dockerfile에도 추가

학습(5000 iter) → evaluate.py로 Phase 0과 비교 → visualize (color_by normal) PLY export.
렌더링 결과 이미지 저장 (Depth, Normal 각 2~3장 + GT RGB 참고용, Phase 0과 비교 가능하게).
results/phase1/REPORT.md 작성 (정량 지표 + 정성적 비교 이미지 포함).
CLAUDE.md 진행 상태 업데이트.
```

**완료 노트:** MVS depth + MVS native normal 적용. Depth MAE=0.053, Normal cos=0.840 (vs MVS GT). `results/phase1/` 참조.

---

## Phase 2-A: 2D Segmentation 생성

**목표:** Grounded SAM 2 + MVS normal + DJI gimbal pitch로 roof/wall/ground segmentation map 생성.

**Go/No-Go:** Wall/Ground 시각 검수 80%+ → Go (Roof는 낮아도 허용 — L_mutual이 보완)

**최종 구현 (v10: Confident Labels Only + Ambiguous as Background):**
- Grounded SAM (building + ground) → DJI gimbal pitch로 camera frame gravity 계산 → depth-derived normal dot product + height-based classification으로 roof/wall/ground 분류
- Normal source: **smoothed MVS depth → 3D unprojection → finite-diff normals** (camera frame, [-1,1])
  - MVS PatchMatch normals는 텍스처 없는 facade에서 퇴화 (|dot|≈0.755 uniform) → depth-derived로 전환
  - MVS depth는 multi-view 삼각측량으로 기하학적 검증됨 → 신뢰할 수 있음
- Gravity source: DJI EXIF `drone-dji:GimbalPitchDegree` → `gravity_up_cam = [0, -cos(pitch), sin(pitch)]`
- Two-threshold system: horiz_thresh=0.85 (strong horizontal), wall_thresh=0.3 (strong vertical)
- **Ambiguous normals (0.3 < |dot| ≤ 0.85) → background(0)**: multi-view 일관적 오분류 방지, L_mutual에 위임
- Height-based roof/ground: strong horizontal + elevated → roof, + ground level → ground (Y_ref=0.308)
- Neighbor propagation: no-depth building 픽셀 → 주변 라벨 majority vote (ambiguous는 propagation 대상 아님)
- Score-based overlap: building_score > ground_score → building zone
- COLMAP world frame 불사용 (gravity를 camera frame에서 직접 계산)

**결과:** 100 hybrid + 80 text-only. Roof 5.9%, Wall 23.4%, Ground 19.5%, Coverage 48.8%.
- Wall/Ground 분류 정확 (facade=wall, 도로=ground 일관성 확인)
- **Confident labels only**: ambiguous 픽셀 강제 분류하지 않아 multi-view consistency trap 방지
- 물리적 정확: 고층(facade만)→roof 낮음, 저층(옥상 노출)→roof 높음
- Ambiguous 영역 → background → L_mutual의 L_slope가 학습 중 기하학적으로 결정 (설계 의도)
- 시각적 확인: `user_inputs/testset/0_25x/seg_vis/` (빨강=roof, 파랑=wall, 초록=ground 오버레이)

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 2-A를 진행해줘. 컨테이너 내부에서 작업이야.

scripts/generate_segmentation.py를 만들어줘.
Grounded SAM 2 설치가 필요하면 Dockerfile에도 추가해줘.

MVS Hybrid 접근법으로 구현:
1. Grounded SAM 2 (Grounding DINO + SAM 2.1)로 "building"과 "ground" 영역 검출
2. DJI EXIF gimbal pitch로 camera frame에서 gravity 방향 계산:
   - gravity_up_cam = [0, -cos(pitch), sin(pitch)] (OpenCV 카메라 규약)
   - COLMAP world frame은 사용하지 않음
3. Building 영역 내에서 |dot(normal_cam, gravity_up_cam)|로 roof/wall 분류:
   - Two-threshold: > 0.85 → roof(1)/ground(3) (height 기준), ≤ 0.3 → wall(2), 사이 → background(0, L_mutual에 위임)
   - Ambiguous→background: multi-view consistency trap 방지 (v10)
4. Score-based overlap: building_score > ground_score → building zone으로 배정
5. Normal 없는 픽셀 → background(0, ignore_index=0)
6. MVS normal/DJI pitch 없는 이미지 → text-only fallback
7. Normal source: **depth-derived** (smoothed MVS depth → 3D → finite-diff, compute_depth_normals() in generate_segmentation.py)

- 입력: 이미지 폴더 + input_data.pth + raw DJI 이미지 (EXIF용)
- 출력: seg_maps/ (class index png) + seg_vis/ (오버레이 확인용)

실행:
python scripts/generate_segmentation.py \
    --image_dir user_inputs/testset/0_25x/dense/images \
    --output_dir user_inputs/testset/0_25x/seg_maps \
    --vis_dir user_inputs/testset/0_25x/seg_vis \
    --input_data planarSplat_ExpRes/seongsu_phase1_mvsnormal/input_data.pth \
    --raw_image_dir user_inputs/testset/raw/images

실행 후 seg_vis/에서 여러 장 확인. 특히 건물 facade가 wall(파랑)로 분류되는지 확인.

results/phase2a/REPORT.md 작성:
- 정량 지표: 전체 180장 coverage, 클래스별 비율 (mean/min/max)
- 정성적 결과: 데이터셋 전체에서 다양한 장면 8~9장 선별 (초반/중반/후반 균등 분포)
  - seg_vis/ 이미지를 results/phase2a/images/에 복사
  - 각 이미지에 구체적 캡션 (어떤 구조물이 어떤 클래스로 분류되었는지)
  - 이미지별 클래스 비율 표 포함
- 이전 접근법들과의 비교 표 (시도한 방법들의 장단점)
- Go/No-Go 근거 (다양한 장면 기반)
- 이슈 및 해결 (발생한 문제와 해결 방법)
CLAUDE.md 진행 상태 업데이트.
```

**완료 노트:** v10 confident-labels-only. Roof 5.9%, Wall 23.4%, Ground 19.5%, Coverage 48.8%. `results/phase2a/` 참조.

---

## Phase 2-B: 의미론적 헤드 구현

**목표:** f_i(K=4) 추가, semantic 렌더링, L_sem 구현, L_geo(L_normal_consistency) 추가. 구현 난이도가 가장 높은 Phase.
K=4인 이유: seg_maps class ID가 0(bg), 1(roof), 2(wall), 3(ground)이고, CrossEntropyLoss(ignore_index=0)는 target=0 픽셀만 무시할 뿐 출력 채널은 4개 필요.

**Go/No-Go (구현 검증):**
| 검증 항목 | Go | Retry |
|-----------|-----|-------|
| Gradient check | ∂L_sem/∂f_i ≠ 0, ∂L_sem/∂R_i == 0 | 실패 → 렌더링 파이프라인 점검 |
| Forward pass | semantic 렌더링 출력이 유효 (NaN/Inf 없음, 클래스별 확률합=1) | 오류 → 코드 디버깅 |
| Density control | split/prune 후 f_i 차원 일치, optimizer state 정상 | 불일치 → sync 로직 수정 |
| 기존 기능 보존 | --enable_semantic 없이 기존과 동일 결과 | 간섭 → 플래그 분기 점검 |

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 2-B를 진행해줘.
docs/RESEARCH_CONTEXT.md의 "프리미티브 파라미터 전체 구조" 섹션도 참고해줘.
컨테이너 내부에서 작업이야.

먼저 확인:
1. 프리미티브 파라미터 정의 파일/클래스
2. 렌더러에서 alpha-blending 가중치 접근 방법
3. adaptive density control 위치
4. CUDA rasterizer의 colors_precomp 채널 수 제한 확인 (config.h의 NUM_CHANNELS)
   - NUM_CHANNELS=3 하드코딩이면 K=4 feature 전달 불가
   - 해결: config.h에서 NUM_CHANNELS를 4로 변경 후 재빌드 (커널 로직 수정 아닌 컴파일 상수 변경)
   - 또는 K=3 리매핑: roof→0, wall→1, ground→2, bg→ignore_index=-100 (CUDA 수정 완전 회피)

데이터 로딩 (seg_maps 학습 파이프라인 통합):
- seg_maps 위치: user_inputs/testset/0_25x/seg_maps/ (class index PNG, 0-3)
- input_data.pth에 'segmentation' 키 추가 (colmap_to_ps.py 수정) 또는 scene_dataset_demo.py에서 직접 로드
- scene_dataset_demo.py: ViewInfo.gt_info에 'seg_map' (hw,) long tensor 추가
- 리사이즈 시 cv2.INTER_NEAREST 사용 (class index이므로 보간 금지)
- seg_maps는 100장(0615-0714)만 학습 이미지와 매칭됨 (나머지 80장은 input_data.pth 범위 밖)
- trainer.py에서 view_info.seg_map 접근하여 L_sem 계산

구현:
1. f_i (K=4, learnable) 추가 — seg_maps class ID가 0-3이고 CrossEntropyLoss(ignore_index=0) 사용하므로 출력 4채널 필요. 균등 분포 초기화, optimizer 등록, density control (split→복사, prune→제거)
2. semantic 렌더링 (Option A — raw feature blend → softmax):
   - raw f_i를 colors_precomp에 전달 → rasterizer가 alpha-blend → 결과에 softmax 적용
   - 즉 softmax는 렌더링 이후(2D pixel 레벨)에서 수행
   - CUDA 커널 로직(forward.cu/backward.cu) 수정 금지. config.h NUM_CHANNELS 변경은 허용 (컴파일 상수)
   - NUM_CHANNELS=4로 변경 시: 기존 colors_precomp가 (N,3)이므로 forward pass 수정 필요
     → --enable_semantic ON: f_i (N,4) 전달, OFF: torch.rand(N,4) 전달 (기존 기능 보존)
   - L_mutual은 렌더링을 거치지 않는다 — per-primitive softmax(f_i)와 n_i를 직접 사용 (RESEARCH_CONTEXT.md 참조)
3. L_sem = CrossEntropyLoss(ignore_index=0), --enable_semantic --lambda_sem 0.1
4. L_geo = L_normal_consistency (렌더링 normal vs depth 유도 normal 일치), --lambda_geo 0.1 (기본값 0 → 기존 동작 보존)
   - docs/RESEARCH_CONTEXT.md의 "L_geo" 섹션 참조
   - allmap[0](depth)에서 finite diff로 normal 유도 → allmap[2:5](rendered normal)과 비교
   - Depth discontinuity에서 노이즈 방지: depth gradient가 큰 edge 픽셀 제외 (2DGS/PGSR 표준 구현 참조)
   - L_planar, L_adj는 CUDA primitive ID 채널 필요 → Phase 4에서 구현
5. TensorBoard: L_sem, L_geo, 클래스별 프리미티브 수(매 100 iter), Semantic vs GT(매 500 iter)
6. visualize_primitives.py에 --color_by class (roof=빨강, wall=파랑, ground=회색)
7. evaluate.py에 --metrics semantic_miou
8. gradient check:
   - torch.autograd.grad(L_sem, f_params) non-zero 확인 (f_i에 gradient 전달됨)
   - torch.autograd.grad(L_sem, R_params) == zero 확인 (L_sem이 R_i를 건드리지 않음 → Phase 3-A에서 L_mutual 고유 효과 근거)
9. --enable_semantic 플래그로 기존 기능 보존
10. 패키지 필요하면 Dockerfile에도 추가

참고 (항공 이미지 특성):
- seg_maps GT는 Grounded SAM + depth-derived normals + DJI pitch 기반으로 생성됨 (Phase 2-A). 완벽하지 않은 noisy GT이다.
- 클래스 불균형: Roof 5.9%, Wall 23.4%, Ground 19.5%, Background 51.2% (180장 평균, v10)
- Roof가 특히 적음 (유효 라벨 중 ~12%). Oblique view에서 roof가 적게 보이는 본질적 한계
- Ambiguous normal(0.3<|dot|≤0.85) 픽셀은 background → L_mutual에 위임 (multi-view consistency trap 방지)
- L_sem의 multi-view consistency가 2D seg GT 노이즈를 자연스럽게 희석 (같은 3D 프리미티브가 여러 view에서 supervision 받음)

results/phase2b/REPORT.md 작성 (구현 검증 결과):
- gradient check 결과 값 (∂L_sem/∂f_i, ∂L_sem/∂R_i)
- L_geo(L_normal_consistency) 구현 확인 (depth→finite diff normal vs rendered normal)
- 수정/추가한 파일 목록
- density control 동작 확인 결과
- 기존 기능 보존 확인 결과
- 이슈 및 해결
CLAUDE.md 진행 상태 업데이트.
```

**완료 노트 (2026-02-19):**
- 10/10 구현 항목 PASS. 상세: `results/phase2b/REPORT.md`
- **스펙 이탈 1건**: L321 "CUDA 커널 로직 수정 금지" 위반 — backward.cu에 color gradient atomicAdd 2줄 추가. 원본이 `dL_dcolors` 축적을 생략했으므로(random color → 불필요) 이 수정 없이는 `∂L_sem/∂f_i = 0`. Color→alpha gradient path는 의도적 미구현 (L_sem→geometry 격리 유지).
- **Gradient 격리 검증**: L_sem→f_i only (6 geometry params=0), L_geo→geometry only (f_i=0)
- **Smoke test**: 3 iter, L_sem 1.386→1.374, f_i 0→6.85, classes emerging
- **Phase 3-A 전방 호환 확인**: f_i(N,4) + `get_plane_geometry()`→`plane_normal`(N,3) → L_mutual per-primitive 직접 계산 가능

---

## Phase 2-C: L_sem 독립 학습

**목표:** L_mutual 없이 baseline 확보. Ablation (c) Independent 조건의 결과.

**Go/No-Go:**
| 지표 | Go | Retry |
|------|-----|-------|
| mIoU | ≥ 0.50 | 0.30~0.50 → λ_s 조정 |
| Depth MAE | Phase 1 대비 ≤ 10% 악화 | > 10% → λ_s 감소 |
| Normal cos | Phase 1 대비 ≤ 5% 악화 | > 5% → λ_s 감소 |

**Segmentation 초기값 영향 분석 (Phase 2-A → 2-C 연결):**
Phase 2-A에서 생성한 seg_maps는 noisy GT이다 (v10 confident-labels-only: wall/ground 정확, roof 5.9%로 낮지만 coherent, ambiguous→background). Phase 2-C 결과 분석 시 다음을 확인:
- Seg map에서 미분류(background=0, 51.2%)인 영역이 학습에 영향 없는지 (`ignore_index=0` 동작 확인)
- Roof 클래스의 낮은 비율(5.9%)이 학습 편향을 유발하는지 (class별 프리미티브 수 TensorBoard 확인)
- 결과가 불만족스러우면 아래 순서로 seg_maps 개선 검토

**Seg map 개선 전략 (mIoU 미달 시 순서):**
1. Class-balanced weighting (inverse frequency 또는 focal loss) — 코드 수정만으로 가능
2. λ_sem 조정 (0.05~0.5 범위)
3. Text-only 80장에 multi-prompt voting ("building roof" + "building wall" 별도 검출 후 score 합산)
4. Pitch-adaptive threshold: 이미지별 |sin(pitch)| + margin으로 degenerate 경계 자동 조정
5. Seg map 자체가 병목이면 Phase 2-A로 돌아가 재생성 후 재학습

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 2-C를 진행해줘. 컨테이너 내부에서 작업이야.

--enable_semantic --lambda_sem 0.1 --lambda_geo 0.1로 5000 iter 학습.
L_geo는 Phase 2-B에서 구현한 L_normal_consistency. 이후 모든 실험에 동일하게 포함.
evaluate.py로 Phase 1과 비교 (depth_mae, normal_cos, semantic_miou).
visualize (color_by class) → PLY export.
렌더링 결과 이미지 저장 (Depth, Normal, Semantic 각 2~3장 + GT RGB 참고용).
웹 뷰어 또는 PLY에서 3D 클래스별 시각화 캡처도.

mIoU가 낮으면(< 0.50) 다음 순서로 개선 시도:
1. class-balanced weighting 적용: inverse frequency weight 또는 focal loss (Roof 비율이 5.9%로 매우 낮음)
2. λ_sem 조정 (0.05~0.5 범위)
3. seg_maps 자체 품질 개선 (Phase 2-A의 threshold 조정, confidence filtering 등 — MEMORY.md 참조)

results/phase2c/REPORT.md 작성 (정량 + 정성 이미지 포함).
CLAUDE.md 진행 상태 업데이트.
```

---

## Phase 3-A: L_mutual 구현

**목표:** 핵심 기여인 L_mutual 구현, 양방향 gradient 검증.

**핵심:** docs/RESEARCH_CONTEXT.md의 L_mutual 수식, gradient 분석, 파라미터 구조 참조.
L_mutual은 R_i(→n_i)와 f_i(→p_c)만 연결. 다른 파라미터(c_i, r_i, color, opacity)는 무관.

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 3-A를 진행해줘.
docs/RESEARCH_CONTEXT.md의 L_mutual 수식과 gradient 분석을 반드시 참고해줘.
컨테이너 내부에서 작업이야.

구현:
1. L_mutual (RESEARCH_CONTEXT.md 수식 참조)
   - f_i와 R_i(→n_i) 모두에 미분 가능 (detach 금지)
   - 다른 파라미터(c_i, r_i, color, opacity)는 L_mutual과 무관
2. Warmup: 3단계 curriculum (RESEARCH_CONTEXT 참조)
   - --mutual_warmup_start 0.33 --mutual_warmup_end 0.67 --lambda_mutual 0.05
   - 0~33%: λ_m=0, 33%~67%: 점진적 증가, 67%~100%: 목표값 유지
3. --mutual_mode full/sem2geo/geo2sem/none
   - sem2geo: softmax(f_i).detach() → R_i만 gradient
   - geo2sem: n_i.detach() → f_i만 gradient
4. TensorBoard: L_mutual, gradient norm (||∂L/∂f||, ||∂L/∂R||) 매 100 iter
5. 학습 시작 시 gradient check 자동 실행 (양방향 non-zero 확인)
6. configs/에 ablation .conf 7개 생성 (pyhocon, base config에 merge 가능하도록)
   - RESEARCH_CONTEXT.md Ablation 설계 참조: core (a)~(d), directional (e)~(f), warmup (d-nowarmup)
7. 패키지 필요하면 Dockerfile에도 추가

results/phase3a/REPORT.md 작성 (구현 검증 결과):
- 양방향 gradient check 결과 값 (∂L_mutual/∂f_i, ∂L_mutual/∂R_i)
- mutual_mode별 gradient 흐름 확인 (full: 양방향, sem2geo: R_i만, geo2sem: f_i만)
- 수정/추가한 파일 목록
- ablation .conf 7개 내용 요약
- 이슈 및 해결
CLAUDE.md 진행 상태 업데이트.
```

---

## Phase 3-B: Ablation 학습

**목표:** 논문 핵심 실험. 양방향 상호 보강 효과 검증. docs/RESEARCH_CONTEXT.md의 Ablation 설계 참조.

**실험 조건 (총 7개):**
- Core Ablation (a)~(d): 전체 접근법 검증 — 핵심 비교는 (c) Independent vs (d) Joint
- Directional Ablation (e)~(f): L_mutual의 방향별 효과 검증
- Warmup Ablation (d-nowarmup): warmup 필요성 검증

**Go/No-Go:** Joint(d)가 Independent(c) 대비 ≥ 2개 지표에서 개선 → 핵심 기여 확인

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 3-B를 진행해줘. 컨테이너 내부에서 작업이야.
docs/RESEARCH_CONTEXT.md의 Ablation 설계를 반드시 참고해줘.

scripts/compare_ablation.py를 만들어줘 (여러 evaluation.json → 비교 표 CSV + 터미널 출력).

순차 실행 (GPU 1개). L_geo는 모든 조건에 동일 포함 (해당 조건 제외):

Core Ablation:
- (a) geo_only: L_depth + L_normal + L_geo (semantic 헤드 없음)
- (b) sem_only: L_depth + L_normal + L_sem (L_geo 없음)
- (c) independent: L_depth + L_normal + L_geo + L_sem (λ_m=0) — Phase 2-C와 동일
- (d) joint: L_depth + L_normal + L_geo + L_sem + L_mutual (warmup 적용)

Directional Ablation:
- (e) sem2geo: (d)에서 softmax(f_i).detach() → R_i만 gradient
- (f) geo2sem: (d)에서 n_i.detach() → f_i만 gradient

Warmup Ablation:
- (d-nowarmup): (d)에서 warmup 없이 λ_m 즉시 적용

configs/에 ablation .conf 7개 생성 (pyhocon, base config에 merge 가능하도록).
각각 evaluate.py 실행 후 compare_ablation.py로 비교 표.
TensorBoard에서 조건 동시 비교 설정도 알려줘.
각 조건의 렌더링 결과 이미지 저장 (Depth/Normal/Semantic + GT RGB 참고용).
3D 클래스 시각화도 조건 비교 (PLY 또는 웹 뷰어).
results/phase3b/REPORT.md 작성 (ablation 표 + 비교 이미지 + 해석).
CLAUDE.md 진행 상태 업데이트.
```

---

## Phase 3-C: L_photo 추가 실험 (선택적, core ablation 이후)

**목표:** PlanarSplatting 원래 설계에는 L_photo가 없다. color를 학습 파라미터로 추가하고 L_photo를 포함했을 때 기하/의미론 품질이 달라지는지, L_mutual의 ablation 효과가 달라지는지 확인한다. 이론적 배경은 docs/RESEARCH_CONTEXT.md의 "L_photo 포함 여부" 섹션 참조.

**입력:** Phase 3-B 완료 상태의 코드

**실험 조건:**
| 조건 | L_photo | L_mutual | 비교 기준 |
|------|---------|----------|-----------|
| Phase 3-B (d) | 없음 | Full | (기존 core ablation 결과, 참고용) |
| 3-C-1 | 추가 | 없음 (λ_m=0) | vs Phase 2-C (L_photo 없음, L_mutual 없음) |
| 3-C-2 | 추가 | Full | vs 3-C-1 및 Phase 3-B(d) |

**확인 사항:**
- 3-C-1 vs Phase 2-C: L_photo 추가만으로 기하 지표(Depth MAE, Normal Error)가 개선되는가?
- 3-C-2 vs 3-C-1: L_photo가 있을 때도 L_mutual이 추가 개선을 주는가?
- 3-C-2 vs Phase 3-B (d): L_photo 추가가 전체적으로 더 나은가?

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 3-C를 진행해줘.
docs/RESEARCH_CONTEXT.md의 "L_photo 포함 여부" 섹션도 참고해줘.
컨테이너 내부에서 작업이야.

PlanarSplatting에 L_photo(RGB photometric loss)를 추가하는 실험이야.
현재 코드에는 L_photo가 없어.

구현:
1. color를 학습 가능 파라미터로 추가 (SH 계수 또는 RGB, 코드 구조에 맞게)
   - color 초기화: SfM 포인트 색상에서 가져오거나, GT 이미지에서 투영하여 초기값 설정
   - SH라면 0차만 초기화, RGB라면 nearest-neighbor projection
2. L_photo = L1 or SSIM loss(rendered RGB, GT image) 구현
3. --enable_photo --lambda_photo 플래그
4. density control에서 color도 처리 (split→복사, prune→제거)

실험 (L_geo는 모든 조건에 동일 포함):
- 3-C-1: --enable_photo --lambda_photo 1.0 --lambda_geo 0.1 (L_mutual 없음)
- 3-C-2: --enable_photo --lambda_photo 1.0 --lambda_geo 0.1 --mutual_mode full

각각 evaluate.py 실행. Phase 3-B 결과와 비교.
results/phase3c/REPORT.md 작성 (L_photo 유무 비교 표 + 이미지).
CLAUDE.md 진행 상태 업데이트.
```

---

## Phase 4: CityGML 변환 + 검증

**목표:** CUDA rasterizer에 primitive ID 채널 추가, L_planar/L_adj 구현, 프리미티브 → CityGML LOD2 + val3dity 검증.

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 4를 진행해줘. 컨테이너 내부에서 작업이야.
docs/RESEARCH_CONTEXT.md의 "L_geo" 섹션도 참고해줘.

Part 1: CUDA rasterizer 확장 + L_geo 완성
1. CUDA rasterizer에 primitive ID 채널 추가 (allmap 8번째 채널)
   - forward.cu에서 최대 기여 프리미티브 ID를 기록
   - backward은 이 채널에 대해 gradient 불필요 (argmax이므로)
2. L_planar 구현 (역투영 3D점 → 프리미티브 평면 거리)
3. L_adj 구현 (인접 프리미티브 경계 연속성)
4. L_geo = L_nc + L_planar + L_adj로 완성, --lambda_geo_planar, --lambda_geo_adj 플래그

Part 2: CityGML 변환
scripts/export_citygml.py를 만들어줘.
먼저 Phase 3 checkpoint 구조를 파악해줘.

처리:
1. argmax(softmax(f_i)) → roof/wall/ground
2. 같은 클래스 + cos(normal) > 0.95 + 중심 거리 < 2*r_i → 병합
3. alpha shape → 경계 다각형
4. CityGML XML (RoofSurface, WallSurface, GroundSurface)
5. OBJ도 생성 (색칠)

val3dity로 검증. 패키지 필요하면 Dockerfile에도 추가.
CityGML 3D 시각화 캡처 (QGIS, 웹 뷰어, 또는 OBJ 렌더링).
results/phase4/REPORT.md 작성.
CLAUDE.md 진행 상태 업데이트.
```

---

## 결과 기록

### REPORT.md (Phase별 별도 파일)
각 Phase 완료 시 `results/phaseX/REPORT.md`를 생성한다. 하나의 파일에 누적하지 않고, Phase마다 독립된 파일을 만든다.

```markdown
# Phase X: [Phase 이름] 결과 보고

## 수행 일시
YYYY-MM-DD

## 수행 작업 요약
(무엇을 했는지 간결하게)

## 정량 지표
| 지표 | 값 | 이전 Phase | 변화 |
|------|-----|-----------|------|
| Depth MAE | x.xxxx | x.xxxx | -x.xxxx |
| Normal cos | x.xxxx | x.xxxx | +x.xxxx |
| mIoU | x.xx (Phase 2+) | x.xx | +x.xx |
| PSNR | xx.x dB (Phase 3-C only) | — | — |

## 정성적 결과
데이터셋 전체에서 다양한 장면을 선별하여 제시 (초반/중반/후반 균등 분포, 8~9장 이상).
이미지는 `results/phaseX/images/`에 저장하고 상대 경로로 참조.
각 이미지에 구체적 캡션을 포함: 어떤 구조물이 어떤 결과로 나왔는지 설명.

### 렌더링 결과
![Depth 맵](images/depth_render.png)
![Normal 맵](images/normal_render.png)
![GT RGB 참고](images/gt_rgb.png)

### 이전 Phase와 비교
![Phase 0 vs Phase 1 비교](images/comparison.png)

### 3D 시각화 (Phase 2 이후)
![클래스별 색칠](images/3d_class_view.png)

## Go/No-Go 판단
- [x] Go / [ ] Retry / [ ] Switch
- 근거: ...

## 이슈 및 해결
(문제가 있었으면 무엇이었고 어떻게 해결했는지)

## 다음 Phase
(다음에 할 일)
```

※ 이미지는 `results/phaseX/images/` 에 저장하고 상대 경로로 참조.
※ 정성적 결과가 중요한 이유: 숫자만으로는 벽면/지붕 분리가 제대로 되는지 판단 불가. 시각적 확인이 필수.

### SUMMARY.md (전체 진행 현황)
모든 Phase의 핵심 수치를 한눈에 보려면 `results/SUMMARY.md`를 요청할 수 있다:
```
지금까지 완료된 모든 Phase의 REPORT.md를 읽고,
results/SUMMARY.md에 Phase별 핵심 지표를 요약하는 표를 만들어줘.
```