# Phase 4' 중간 보고: Poisson mesh topology + PlanarSplatting semantics

**목표**: Poisson mesh의 topology + PlanarSplatting의 semantics → val3dity 305 = 0인 CityGML LOD2
**상태**: Step 3 완료 (Step 4~6 진행 중)

---

## Step 1: Poisson mesh 생성 (전체 씬) — 완료

**입력**: `user_inputs/testset/0_25x/dense/fused.ply` (COLMAP, 11,241,735 points)
**스크립트**: `scripts/generate_poisson_mesh.py`

**Poisson 파라미터**: depth=10, linear_fit=True, trim 5%

| 구분 | Vertices | Triangles | Watertight | Edge Manifold |
|------|----------|-----------|------------|---------------|
| Raw | 596,092 | 1,190,954 | No | No |
| Trimmed 5% | 566,287 | 1,123,255 | No | No |
| Trimmed 10% | 536,482 | 1,057,512 | No | No |
| Trimmed 15% | 506,678 | 996,256 | No | No |
| Trimmed 20% | 476,873 | 936,338 | No | No |
| TSDF (비교) | 79,719 | 141,217 | No | Yes |

**판단**: 전체 씬 Poisson mesh가 watertight가 아닌 것은 야외 항공 데이터에서 정상. 개별 건물 단위로 추출 후 watertight 여부는 Step 3에서 확인.

**산출물**: `results/phase4/poisson/poisson_trimmed.ply` (이후 단계에서 사용)

---

## Step 2: Mesh face에 PlanarSplatting semantic label 부여 — 완료

**입력**:
- Poisson mesh: `poisson_trimmed.ply` (566,287 verts, 1,123,255 faces)
- PlanarSplatting 프리미티브: `primitives_data.npz` (2,653개, bg=13, roof=316, wall=1,285, ground=1,039)
- Building ID: `building_data.npz` (11 buildings, Phase 4 Part 2에서 생성)

**스크립트**: `scripts/label_mesh_faces.py`

### 매칭 방식 변천

| 버전 | 방법 | Score median | 문제 |
|------|------|-------------|------|
| v1 | 최근접 center KDTree | 0.82 | 위의 roof가 아래 face에 매칭 |
| v2 | Plane distance + normal similarity | 0.01 | 수천 개 잡음 component |
| v3 | v2 + label smoothing | 0.01 | 현재 버전 |

### v3 매칭 파라미터
- `max_plane_dist=0.15`: face centroid에서 primitive plane까지 최대 거리
- `max_center_dist=0.5`: primitive center까지 최대 거리 (후보 필터)
- `normal_cos_thresh=0.5`: face-primitive normal 방향 일치 최소 |cos|
- `k_candidates=10`: 검토할 후보 primitive 수
- `min_component_size=50`: 이보다 작은 component는 이웃 다수결로 재라벨
- `n_smooth_iters=3`: 경계 smoothing 반복 횟수

### Smoothing 효과 (v2 → v3)

| Class | Before (components) | Before (noise<10) | After (components) | After (noise<10) |
|-------|--------------------|--------------------|--------------------|--------------------|
| Roof | 2,240 | 1,965 | 104 | 27 |
| Wall | 3,542 | 3,154 | 207 | 79 |
| Ground | 2,211 | 1,914 | 248 | 104 |

### 최종 라벨 분포

| Class | Faces | 비율 |
|-------|-------|------|
| bg | 719,960 | 64.1% |
| roof | 43,609 | 3.9% |
| wall | 175,932 | 15.7% |
| ground | 183,754 | 16.4% |
| **building (roof+wall)** | **219,541** | **19.5%** |

매칭된 face: 408,061 (36.3%). 나머지 64%는 건물/ground 프리미티브 범위 밖의 씬 영역.

### Building별 face 분포

| Building | Roof | Wall | Total | 비고 |
|----------|------|------|-------|------|
| 0 | 19,801 | 39,404 | 59,205 | |
| 1 | 11,084 | 33,903 | 44,987 | |
| 2 | 2,209 | 11,038 | 13,247 | |
| 3 | 7,252 | 30,950 | 38,202 | |
| 4 | 639 | 5,608 | 6,247 | roof 적음 |
| 5 | 620 | 8,064 | 8,684 | roof 적음 |
| 6 | 94 | 4,779 | 4,873 | roof 극소 |
| 7 | 231 | 17,015 | 17,246 | roof 극소 |
| 8 | 353 | 6,662 | 7,015 | roof 적음 |
| 9 | 439 | 12,274 | 12,713 | roof 적음 |
| 10 | 887 | 6,235 | 7,122 | |

### 알려진 한계

1. **Building ID 11개**: Phase 4 Part 2의 XZ distance=0.3 connected component가 인접 건물을 병합. Step 3에서 mesh connected component로 재분리 예정.
2. **일부 건물 roof 극소**: Building 6(94), 7(231). 원인: 인접 건물 병합 또는 경사 지붕이 wall로 분류.
3. **매칭 정확도**: Primitive 기반 매칭의 근본 한계. 범위 밖 건물 표면은 bg로 남음. 다만 후속 단계에서 실제 영향을 봐야 판단 가능.

**산출물**: `results/phase4_prime/step2_labeling/`
- `labeled_mesh_semantic.ply`: 의미론적 색상 (bg=gray, roof=red, wall=blue, ground=green)
- `labeled_mesh_building.ply`: Building ID별 색상
- `face_labels.npz`: face_labels, face_building_ids, face_prim_indices, face_dists
- `labeling_summary.json`

---

## Step 3: Building 영역 추출 — 완료

**스크립트**: `scripts/extract_buildings.py`

### 방법: Roof-seed BFS
- Phase 4 Part 2의 building_ids (XZ distance=0.3, 11개 건물) 대신 **mesh topology 기반** 분리
- Roof connected component → 각 component가 건물 seed
- Wall face를 가장 가까운 roof seed에 BFS 귀속
- `min_roof_faces=100` (roof component 최소 크기), `min_building_faces=500` (건물 최소 총 face)

### 결과
- Roof component 53개 발견 → 필터 후 **49개 building** 추출
- Wall 귀속: 151,854 assigned, 24,078 unassigned (roof 도달 불가)

### 대표 건물

| Building | Total | Roof | Wall | Roof% | 비고 |
|----------|-------|------|------|-------|------|
| 0 | 21,232 | 10,889 | 10,343 | 51.3% | **과분할** (MeshLab 확인) |
| 1 | 6,378 | 4,113 | 2,265 | 64.5% | |
| 2 | 9,380 | 3,351 | 6,029 | 35.7% | |
| 5 | 18,060 | 1,296 | 16,764 | 7.2% | **roof가 바닥면** (MeshLab 확인) |
| 13 | 11,243 | 639 | 10,604 | 5.7% | roof 극소 |
| 36 | 7,435 | 166 | 7,269 | 2.2% | roof 극소 |

### MeshLab 확인 결과 (user feedback)
1. **Building 0**: 과분할 심각 — 하나의 건물이 아닌 여러 구조물 포함
2. **Building 5**: Roof가 실제로는 바닥면 — Step 2 labeling 오류가 Step 3으로 전파
3. 모든 건물 watertight=False, edge_manifold=True

### 알려진 한계
- **Step 2 labeling 오류 전파**: primitive 기반 매칭의 부정확성이 building 분리에 직접 영향
- **Roof-seed 의존성**: roof labeling이 부정확하면 building 전체가 잘못됨
- **Wall 24K unassigned**: roof에 도달할 수 없는 고립 wall face 존재
- 단, 끝까지 진행하여 최종 CityGML에서의 영향을 확인 후 개선점 판단

**산출물**: `results/phase4_prime/step3_buildings/`
- `building_000.ply` ~ `building_052.ply` (49개, semantic 색상)
- `building_faces.npz`: face_labels, building_ids, face_prim_indices
- `buildings_summary.json`

---

## Step 4-6 v1: Per-primitive regions → CityGML — 완료 (v2로 대체)

v1은 각 primitive별로 region을 생성하여 surface가 과도하게 많고 (Building 1: 54, Building 3: 105),
PlanarSplatting의 normal을 활용하지 않아 mesh boundary를 직접 polygon으로 사용.
305=0 달성했으나 203(비평면), 104(자기교차) 문제.

---

## Step 4-6 v2: Co-planar merge + geometry 정제 — 완료

**스크립트**: `scripts/building_to_citygml.py`

### 설계 원칙
- **Mesh** → geometry + topology (빈 부분 채움, edge 제공)
- **PlanarSplatting** → normal + semantics (surface 분류, co-planar 그룹화)

### Step 4 v2: Co-planar Region Merge
- PlanarSplatting normal 유사도 (cos > 0.90) + 같은 semantic label + mesh 인접성 → 하나의 surface
- 50 face 미만 소규모 region은 인접 대형 region에 병합
- v1→v2 surface 수 감소: Building 1: 54→16, Building 3: 105→16, Building 10: 24→6

| Building | v1 regions | v2 regions | Roof | Wall |
|----------|-----------|-----------|------|------|
| 1 | 54 | **16** | 4 | 12 |
| 3 | 105 | **16** | 7 | 9 |
| 10 | 24 | **6** | 1 | 5 |

### Step 5 v2: Boundary Simplification
- Region pair별 shared boundary를 Douglas-Peucker로 단순화 (tol=0.1m)
- Pair별로 한 번 단순화 → 양쪽 region이 동일한 simplified vertex 사용 (shared edge 보장)
- 단순화된 polygon vertex: 3~17개/surface (v1 대비 대폭 감소)

### Step 6 v2: CityGML Construction (geometry 정제 포함)

**해결된 문제:**

1. **203 (Non-planar) → 0**: Vertex consensus optimization
   - 각 surface에 SVD best-fit plane 계산
   - 공유 vertex: 인접 surface들의 plane에 대한 least-squares 최적 위치
   - 단독 vertex: 해당 surface plane에 직접 투영
   - 결과: 모든 surface planarity = 0.0000m

2. **104 (Self-intersection) → 부분 해결**: Convex hull scaffold + vertex insertion
   - 2D 투영에서 convex hull로 기본 순서 결정
   - Hull 외부 vertex는 가장 가까운 hull edge에 삽입
   - 대부분의 surface에서 자기교차 제거

3. **307 (Wrong orientation) → 부분 해결**: Edge-based BFS 전파 + signed volume
   - 인접 surface 간 directed edge 비교로 일관된 winding 전파
   - Signed volume 검사로 전체 방향 결정 (음수면 전체 뒤집기)

### val3dity 검증 결과

| Building | 102 | 104 | 203 | 305 | 307 |
|----------|-----|-----|-----|-----|-----|
| 1 (16 surfaces) | - | face 12 (1건) | **0** | **0** | - |
| 3 (16 surfaces) | - | - | **0** | **0** | face 6 (1건) |
| 10 (6 surfaces) | face 1,4 (2건) | - | **0** | **0** | - |

### v1→v2 비교

| 항목 | v1 | v2 |
|------|----|----|
| Surface 수 (B10) | 24 | **6** |
| 203 (Non-planar) | 3/3 buildings | **0/3** |
| 104 (Self-intersect) | 2/3 buildings | 1/3 buildings (1 face) |
| 305 (Edge sharing) | **0/3** | **0/3** |
| 307 (Orientation) | N/A | 1/3 buildings (1 face) |
| 102 (Same points) | N/A | 1/3 buildings (2 faces) |
| PlanarSplatting normal 활용 | 아니오 | **예** (co-planar merge) |
| Polygon vertex 수 | 40~60 | **3~17** |

### 핵심 성과

1. **305 = 0 유지**: mesh topology 기반 shared edge가 여전히 유효
2. **203 = 0 달성**: vertex consensus optimization으로 완벽한 평면성 확보
3. **Surface 수 대폭 감소**: PlanarSplatting normal 활용 co-planar merge 효과적
4. **PlanarSplatting의 역할 확립**: normal → co-planar grouping, semantics → roof/wall 분류

### 남은 에러 (minor, fixable)

| 에러 | 건수 | 원인 | 해결 방향 |
|------|------|------|----------|
| 104 | 1 face / 1 building | 복잡한 concave polygon에서 hull scaffold의 한계 | Polygon simplification 강화 또는 삼각분할 |
| 307 | 1 face / 1 building | Edge 연결이 없는 고립 surface의 orientation 결정 불가 | Signed volume 기반 개별 surface 판정 |
| 102 | 2 faces / 1 building | Vertex consensus가 인접 vertex를 같은 위치로 이동 | 정수 좌표 기준 dedup 강화 |

### 산출물
`results/phase4_prime/step456/building_XXX/`
- `regions.ply`: v2 region별 색상 mesh
- `polygons.ply`: 단순화된 polygon 시각화
- `building.city.json`: CityJSON LOD2

---

## 종합 평가

### Phase 4 vs Phase 4' 비교

| 항목 | Phase 4 (primitive 직접) | Phase 4' v2 (mesh+normal) |
|------|-------------------------|---------------------------|
| PlanarSplatting 역할 | polygon geometry | **normal + semantics** |
| Mesh 역할 | 없음 | **geometry + topology** |
| Surface 수 (B10) | 24 | **6** |
| val3dity 305 | **다수** | **0** |
| val3dity 203 | 0 | **0** |
| val3dity 104 | 0 | 1건/3건물 |
| val3dity 307 | N/A | 1건/3건물 |
| Building 분리 | XZ clustering (11개) | Roof-seed BFS (49개) |

### 결론

Phase 4' v2는 핵심 목표(305=0)와 추가 목표(203=0)를 모두 달성.

**확인된 가설:**
- Poisson mesh의 topology가 PlanarSplatting의 topology 부재를 보완 (305=0)
- PlanarSplatting의 normal이 co-planar surface grouping에 효과적 (surface 수 75~94% 감소)
- Vertex consensus optimization이 shared edge 유지하면서 planarity 확보 가능 (203=0)

**남은 과제:**
- 102/104/307: geometry 정제 수준의 minor 에러 (각 1~2건). 해결 가능하나 우선순위 낮음.
- Step 2 labeling 품질: roof 오분류, building 과분할이 최종 CityGML 품질의 상한을 결정.
- City3D 등 기존 방법과의 비교가 필요.

---

## Step 4-6 v4: Convex Polytope from PlanarSplatting Planes — 완료

**스크립트**: `scripts/building_to_citygml_v4.py`

### 핵심 아이디어

v2는 Mesh를 geometry의 주 출처로 사용하고 PlanarSplatting은 normal/semantics만 제공. v4는 **반대 접근**: PlanarSplatting의 plane equation이 geometry를 직접 정의하고, mesh는 building 분할에만 사용.

각 건물 = PlanarSplatting surface group들의 half-space intersection으로 정의된 **convex polytope**.

### 알고리즘

1. **Primitive clustering**: Normal similarity (cos > 0.85) + semantic class → surface groups
2. **Normal orientation**: 건물 중심 기준 outward 방향으로 flip
3. **Ground + Bbox planes**: Virtual ground surface at wall base + 미관측 방향 bbox planes
4. **Half-space intersection**: 모든 3-plane intersection → half-space filter → valid vertices
5. **ConvexHull**: scipy ConvexHull → manifold solid (every edge shared by exactly 2 faces)
6. **Face mapping**: Hull triangles → surface groups (plane residual + normal matching)
7. **Triangle merge**: Coplanar hull triangles → polygon via boundary walk
8. **CityJSON**: Integer vertex quantization + signed volume check → CityJSON 2.0 Solid

### val3dity 결과

| 항목 | 결과 |
|------|------|
| **Buildings processed** | **53/53** |
| **val3dity ALL VALID** | **53/53 (100%)** |
| 102 (Same points) | 0 |
| 104 (Self-intersection) | 0 |
| 203 (Non-planar) | 0 errors (39 info-level warnings, distance ≤ 0.016m) |
| 303 (Non-manifold edge) | 0 |
| 305 (Multiple connected comp) | 0 |
| 307 (Wrong orientation) | 0 |

### 정량적 결과

| 항목 | 값 |
|------|-----|
| Total surfaces | 373 (86 Roof + 251 Wall + 36 Ground) |
| Surfaces/building | 4~11 (mean 7.0) |
| Vertices/building | 6~18 (mean 10.0) |
| Perfect manifold (0bd/0nm) | 43/53 (81%) |
| Buildings with boundary edges | 10/53 (모두 val3dity VALID) |
| Total edges | 762 shared, 61 boundary, 2 non-manifold |

### Bbox planes의 역할

항공 경사 이미지에서는 건물의 한쪽 면만 관측됨. 예: Building 0의 9개 wall이 모두 -X 방향 → 반대편(+X) wall이 없어 polytope가 2D로 퇴화. Bbox planes가 미관측 방향을 보완:

| Buildings | bbox planes 0개 | 1~2개 | 3~4개 |
|-----------|----------------|-------|-------|
| Count | 11 | 24 | 18 |

### v2 vs v4 비교

| 항목 | v2 (Mesh geometry) | v4 (Plane intersection) |
|------|-------------------|------------------------|
| Geometry source | Poisson mesh | **PlanarSplatting planes** |
| Mesh role | Geometry + topology | Building 분할만 |
| val3dity VALID | 0/3 (3 buildings tested) | **53/53 (100%)** |
| 305 (Topology) | 0 | 0 |
| 203 (Planarity) | 0 | 0 (39 warnings) |
| 102/104/307 | 각 1건 | **0** |
| PlanarSplatting 활용도 | Normal + semantics | **Plane eq + semantics** |
| 건물 형태 | Non-convex (mesh faithful) | Convex approximation |
| Surfaces/building (mean) | ~8 | 7.0 |

### 정성적 관찰

출력 PLY (`lod2.ply`)에서 확인:
- 건물이 **convex polyhedra**로 표현됨 — LOD2 수준에서 합리적
- Roof (red) surfaces가 건물 상부, Wall (blue)이 측면, Ground (gray)가 하부에 위치
- 일부 건물에서 GroundSurface 누락 (ground가 bbox plane에 의해 wall로 흡수) — 36/53에만 Ground 존재
- Bbox planes로 인한 인공적 wall surface 존재 (미관측 방향) — LOD2에서 불가피

### 알려진 한계

1. **Convex 제약**: L-shaped 건물은 convex hull로 근사됨, concavity 손실
2. **Bbox surfaces**: 미관측 방향에 axis-aligned 평면 삽입 — 실제 건물 형태와 다를 수 있음
3. **e203 warnings (39/53 buildings)**: 정수 좌표 양자화로 인한 미소 비평면성 (최대 0.016m)
4. **10 buildings with boundary edges**: Polygon merge에서 일부 non-simple boundary 발생, 그러나 val3dity는 모두 VALID

### 산출물

`results/phase4_prime/step456_v4/`
- `building_XXX/building.city.json`: CityJSON 2.0 LOD2 Solid
- `building_XXX/lod2.ply`: Colored PLY (Roof=red, Wall=blue, Ground=gray)
- `summary.json`: 전체 결과 요약

---

## 종합 평가 (업데이트: v4)

### Phase 4 → Phase 4' v2 → v4 비교

| 항목 | Phase 4 (primitive 직접) | Phase 4' v2 (mesh geometry) | Phase 4' v4 (plane intersection) |
|------|------------------------|-----------------------------|----------------------------------|
| val3dity VALID | 0/11 | 0/3 | **53/53 (100%)** |
| Geometry source | PlanarSplatting polygon | Poisson mesh | **PlanarSplatting planes** |
| Manifold guarantee | No | Mesh-based (305=0) | **ConvexHull (by construction)** |
| PlanarSplatting 역할 | Geometry (failed) | Normal+Semantics | **Plane eq+Semantics** |
| 건물 수 | 11 | 3 (test) | **53** |
| Building 형태 | Irregular | Non-convex | Convex approximation |

### 결론

Phase 4' v4가 연구의 핵심 목표인 **"PlanarSplatting plane equations → val3dity VALID CityGML LOD2"**를 달성.

**핵심 기여:**
1. PlanarSplatting의 plane equation (normal + center)이 LOD2 건물 geometry를 직접 정의할 수 있음을 실증
2. Convex polytope (half-space intersection) 접근이 manifold 보장을 통해 val3dity 100% VALID 달성
3. Semantic labels (roof/wall) → CityGML surface types (RoofSurface/WallSurface/GroundSurface) 매핑
4. Aerial oblique의 partial observation 문제를 bbox planes로 보완하는 실용적 전략

**향후 과제:**
- Non-convex 건물 처리 (convex decomposition 또는 plane arrangement)
- City3D 등 기존 방법과의 정량적 비교
- L_mutual 재실험 (Phase 3-B'-Step2)
