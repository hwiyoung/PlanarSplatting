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
   - 체크포인트 → PSNR, Depth MAE → JSON
   - --checkpoint, --metrics, --output, --compare_with
4. 패키지 필요하면 Dockerfile에도 추가해줘

검증: 기존 사전실험 체크포인트가 있으면 그걸로 스크립트 작동 확인.
TensorBoard에 ssh 터널링으로 접속 가능한지 확인.

완료 후 CLAUDE.md의 진행 상태에서 Phase 0-Setup을 체크해줘.
```

---

## Phase 0: SfM/MVS 입력 확보

**목표:** VGGT 10% 정합 문제 해결, 공통 3D 입력 확보.

**입력:** 리사이즈된 드론 이미지 (2048x1365)

**Go/No-Go:**
| 지표 | Go | Retry | Switch |
|------|-----|-------|--------|
| 정합 이미지 | ≥ 100장 | 50~99 → 매칭 조정 | < 50 → Metashape |
| PSNR | ≥ 20dB | 18~20 → iter 증가 | < 18 → 입력 변환 디버깅 |
| 건물 형태 | PLY/웹뷰어에서 식별 가능 | 뭉개짐 → 초기화 조정 | — |

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
렌더링 결과 이미지도 저장해줘 (RGB, Depth, Normal 각 2~3장).
/results/phase0/REPORT.md를 EXPERIMENT_PLAN.md 하단의 REPORT 템플릿에 따라 작성해줘.
CLAUDE.md 진행 상태 업데이트.
```

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
렌더링 결과 이미지 저장 (RGB, Depth, Normal 각 2~3장, Phase 0과 비교 가능하게).
/results/phase1/REPORT.md 작성 (정량 지표 + 정성적 비교 이미지 포함).
CLAUDE.md 진행 상태 업데이트.
```

---

## Phase 2-A: 2D Segmentation 생성

**목표:** Grounded SAM 2로 roof/wall/ground segmentation map 생성.

**Go/No-Go:** 시각 80%+ → Go / 50~80% → 프롬프트 조정

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 2-A를 진행해줘. 컨테이너 내부에서 작업이야.

scripts/generate_segmentation.py를 만들어줘.
Grounded SAM 2 설치가 필요하면 Dockerfile에도 추가해줘.

- "building roof"→1, "building wall"/"facade"→2, "ground"/"road"→3, 나머지→0
- 입력: 이미지 폴더
- 출력: seg_maps/ (class index png) + seg_vis/ (오버레이 확인용)

실행 후 seg_vis/에서 여러 장 확인. 샘플 이미지 3~5장을 /results/phase2a/에 복사해줘.
/results/phase2a/REPORT.md 작성 (샘플 이미지 포함).
```

---

## Phase 2-B: 의미론적 헤드 구현

**목표:** f_i(K=3) 추가, semantic 렌더링, L_sem 구현. 구현 난이도가 가장 높은 Phase.

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 2-B를 진행해줘.
docs/RESEARCH_CONTEXT.md의 "프리미티브 파라미터 전체 구조" 섹션도 참고해줘.
컨테이너 내부에서 작업이야.

먼저 확인:
1. 프리미티브 파라미터 정의 파일/클래스
2. 렌더러에서 alpha-blending 가중치 접근 방법
3. adaptive density control 위치

구현:
1. f_i (K=3, learnable) 추가 — optimizer 등록, density control (split→복사, prune→제거)
2. semantic 렌더링: alpha weight × softmax(f_i) (PyTorch 레벨, CUDA 커널 수정 금지)
3. L_sem = CrossEntropyLoss(ignore_index=0), --enable_semantic --lambda_sem 0.1
4. TensorBoard: L_sem, 클래스별 프리미티브 수(매 100 iter), Semantic vs GT(매 500 iter)
5. visualize_primitives.py에 --color_by class (roof=빨강, wall=파랑, ground=회색)
6. evaluate.py에 --metrics semantic_miou
7. gradient check: torch.autograd.grad(L_sem, f_params) non-zero 확인
8. --enable_semantic 플래그로 기존 기능 보존
9. 패키지 필요하면 Dockerfile에도 추가
```

---

## Phase 2-C: L_sem 독립 학습

**목표:** L_mutual 없이 baseline 확보. Ablation (a) No mutual 조건의 결과.

**Go/No-Go:**
| 지표 | Go | Retry |
|------|-----|-------|
| mIoU | ≥ 0.50 | 0.30~0.50 → λ_s 조정 |
| PSNR | Phase 1 대비 ≤ 5% 악화 | > 5% → λ_s 감소 |

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 2-C를 진행해줘. 컨테이너 내부에서 작업이야.

--enable_semantic --lambda_sem 0.1로 5000 iter 학습.
evaluate.py로 Phase 1과 비교 (psnr, depth_mae, normal_error, semantic_miou).
visualize (color_by class) → PLY export.
렌더링 결과 이미지 저장 (RGB, Depth, Normal, Semantic 각 2~3장).
웹 뷰어 또는 PLY에서 3D 클래스별 시각화 캡처도.
/results/phase2/REPORT.md 작성 (정량 + 정성 이미지 포함).
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
2. --mutual_warmup_ratio 0.33, --lambda_mutual 0.05
3. --mutual_mode full/sem2geo/geo2sem/none
   - sem2geo: softmax(f_i).detach() → R_i만 gradient
   - geo2sem: n_i.detach() → f_i만 gradient
4. TensorBoard: L_mutual, gradient norm (||∂L/∂f||, ||∂L/∂R||) 매 100 iter
5. 학습 시작 시 gradient check 자동 실행 (양방향 non-zero 확인)
6. configs/에 ablation yaml 4개 생성
7. 패키지 필요하면 Dockerfile에도 추가
```

---

## Phase 3-B: Ablation 4조건 학습

**목표:** 논문 핵심 실험. 양방향 상호 보강 효과 검증.

**기대 패턴:** RESEARCH_CONTEXT.md의 Ablation 설계 참조.

**Go/No-Go:** Full mutual이 ≥ 2개 지표에서 No mutual 대비 개선 → 핵심 기여 확인

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 3-B를 진행해줘. 컨테이너 내부에서 작업이야.

scripts/compare_ablation.py를 만들어줘 (여러 evaluation.json → 비교 표 CSV + 터미널 출력).

순차 실행 (GPU 1개):
- (a) none: Phase 2 결과 복사
- (b) full: ablation_full.yaml
- (c) sem2geo: ablation_sem2geo.yaml
- (d) geo2sem: ablation_geo2sem.yaml

각각 evaluate.py 실행 후 compare_ablation.py로 비교 표.
TensorBoard에서 4조건 동시 비교 설정도 알려줘.
각 조건의 렌더링 결과 이미지 저장 (4조건 × RGB/Depth/Normal/Semantic).
3D 클래스 시각화도 4조건 비교 (PLY 또는 웹 뷰어).
/results/phase3/REPORT.md 작성 (ablation 표 + 비교 이미지 + 해석).
CLAUDE.md 진행 상태 업데이트.
```

---

## Phase 3-C: L_photo 추가 실험 (선택적, core ablation 이후)

**목표:** PlanarSplatting 원래 설계에는 L_photo가 없다. color를 학습 파라미터로 추가하고 L_photo를 포함했을 때 기하/의미론 품질이 달라지는지, L_mutual의 ablation 효과가 달라지는지 확인한다. 이론적 배경은 docs/RESEARCH_CONTEXT.md의 "L_photo 포함 여부" 섹션 참조.

**입력:** Phase 3-B 완료 상태의 코드

**실험 조건:**
| 조건 | L_photo | L_mutual | 비교 대상 |
|------|---------|----------|-----------|
| Phase 3-B (b) | 없음 | Full | 기존 core ablation 결과 |
| 3-C-1 | 추가 | 없음 (λ_m=0) | L_photo만의 효과 |
| 3-C-2 | 추가 | Full | L_photo + L_mutual 조합 효과 |

**확인 사항:**
- 3-C-1 vs Phase 2-C: L_photo 추가만으로 기하 지표(Depth MAE, Normal Error)가 개선되는가?
- 3-C-2 vs 3-C-1: L_photo가 있을 때도 L_mutual이 추가 개선을 주는가?
- 3-C-2 vs Phase 3-B (b): L_photo 추가가 전체적으로 더 나은가?

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 3-C를 진행해줘.
docs/RESEARCH_CONTEXT.md의 "L_photo 포함 여부" 섹션도 참고해줘.
컨테이너 내부에서 작업이야.

PlanarSplatting에 L_photo(RGB photometric loss)를 추가하는 실험이야.
현재 코드에는 L_photo가 없어.

구현:
1. color를 학습 가능 파라미터로 추가 (SH 계수 또는 RGB, 코드 구조에 맞게)
2. L_photo = L1 or SSIM loss(rendered RGB, GT image) 구현
3. --enable_photo --lambda_photo 플래그
4. density control에서 color도 처리

실험:
- 3-C-1: --enable_photo --lambda_photo 1.0 (L_mutual 없음)
- 3-C-2: --enable_photo --lambda_photo 1.0 --mutual_mode full

각각 evaluate.py 실행. Phase 3-B 결과와 비교.
/results/phase3c/REPORT.md 작성 (L_photo 유무 비교 표 + 이미지).
```

---

## Phase 4: CityGML 변환 + 검증

**목표:** 프리미티브 → CityGML LOD2 + val3dity 검증.

**프롬프트:**
```
docs/EXPERIMENT_PLAN.md의 Phase 4를 진행해줘. 컨테이너 내부에서 작업이야.

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
/results/phase4/REPORT.md 작성.
CLAUDE.md 진행 상태 업데이트.
```

---

## 결과 기록

### REPORT.md (Phase별 별도 파일)
각 Phase 완료 시 `/results/phaseX/REPORT.md`를 생성한다. 하나의 파일에 누적하지 않고, Phase마다 독립된 파일을 만든다.

```markdown
# Phase X: [Phase 이름] 결과 보고

## 수행 일시
YYYY-MM-DD

## 수행 작업 요약
(무엇을 했는지 간결하게)

## 정량 지표
| 지표 | 값 | 이전 Phase | 변화 |
|------|-----|-----------|------|
| PSNR | xx.x dB | xx.x dB | +x.x |
| ... | | | |

## 정성적 결과
(렌더링 이미지, 3D 시각화 캡처 등을 삽입)

### 렌더링 결과
![RGB 렌더링](images/rgb_render.png)
![Depth 맵](images/depth_render.png)
![Normal 맵](images/normal_render.png)

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

※ 이미지는 `/results/phaseX/images/` 에 저장하고 상대 경로로 참조.
※ 정성적 결과가 중요한 이유: 숫자만으로는 벽면/지붕 분리가 제대로 되는지 판단 불가. 시각적 확인이 필수.

### SUMMARY.md (전체 진행 현황)
모든 Phase의 핵심 수치를 한눈에 보려면 `/results/SUMMARY.md`를 요청할 수 있다:
```
지금까지 완료된 모든 Phase의 REPORT.md를 읽고,
/results/SUMMARY.md에 Phase별 핵심 지표를 요약하는 표를 만들어줘.
```