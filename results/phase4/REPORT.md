# Phase 4 프로토타입: CityGML 변환 + Mesh 기반 위상 보정

## 수행 작업 요약

Phase 4는 PlanarSplatting의 학습 결과(평면 프리미티브 + 의미론)를 CityGML LOD2 건물 모델로 변환하는 **end-to-end 프로토타입**이다. 연구의 최종 산출물인 "항공 드론 이미지 → CityGML LOD2 자동 생성"의 실현 가능성을 검증하는 것이 목표이다.

사용 체크포인트: (c') Independent+Photo (Phase 3-B'-Step1 결과, iter=5000)

파이프라인:
```
Checkpoint → 프리미티브 분석(Part 1) → Building Instance 분리(Part 2)
→ TSDF Mesh 추출+위상 분석(Part 3) → CityGML LOD2 변환(Part 4)
→ val3dity 검증(Part 5) → 위상 보정 프로토타입(Part 6)
```

## Part 1: 프리미티브 분석

### 정량 지표

| Class | Count | Ratio | Total Area | Height(Y) mean | gravity align |
|-------|-------|-------|------------|-----------------|---------------|
| bg | 13 | 0.5% | 0.161 | -1.026 | 0.265 |
| roof | 316 | 11.9% | 3.407 | -0.201 | 0.406 |
| wall | 1285 | 48.4% | 16.636 | -0.224 | 0.515 |
| ground | 1039 | 39.2% | 16.653 | 0.048 | 0.423 |
| **Total** | **2653** | | | | |
| **Building** | **1601** | **60.3%** | | | |

> **gravity align** = mean of abs(n · e_gravity). 법선이 중력 방향(수직축)과 얼마나 정렬되었는지의 평균.
> - 수평면(flat roof, ground): gravity align ≈ 1.0 (법선이 수직 = 중력 방향과 평행)
> - 수직면(wall): gravity align ≈ 0.0 (법선이 수평 = 중력 방향과 직교)

### 해석

- **Wall 우세**: Wall이 전체의 48.4%로 가장 많다. 항공 oblique 이미지 특성상 건물 측면이 많이 관측되기 때문.
- **Semantic confidence**: 평균 0.824, 90% 이상 confidence 46.1% (1223/2653).
- **법선 방향 분포 (L_mutual 미적용 기준)**:
  - Roof의 gravity align 평균 0.406 — flat roof라면 1.0에 가까워야 하므로 **경사 지붕이 대다수**임을 시사. 실제 성수동 건물에 경사 지붕이 많은지 확인 필요. 만약 flat roof 건물인데 0.406이면 법선 추정 품질 문제.
  - Wall의 gravity align 평균 0.515 — 이상적인 수직 벽은 0.0이어야 함. 0.515는 벽 법선이 수평이 아니라 **약 30° 기울어져 있다**는 의미. L_mutual(수직 prior)이 이를 교정할 수 있는 영역.
  - L_mutual은 적용되지 않은 Independent 체크포인트이므로, 이 수치는 **기하학만으로 달성한 baseline** 역할.
- **Ground 분리**: Ground의 평균 Y=0.048 (COLMAP frame에서 -Y가 위), roof/wall보다 실제로 낮은 위치 확인.

### 산출물
- `results/phase4/part1/primitives_{bg,roof,wall,ground}.ply`: 클래스별 PLY
- `results/phase4/part1/primitives_building.ply`: Roof+wall 결합 PLY

## Part 2: Building Instance 분리

XZ 평면 투영 + distance threshold(0.3) connected component → **11개 건물 클러스터** 식별.

| Building | Roof | Wall | Total | Area | Y range |
|----------|------|------|-------|------|---------|
| 0 | 158 | 304 | 462 | 5.30 | [-2.10, 0.54] |
| 1 | 90 | 257 | 347 | 3.65 | [-0.98, 0.56] |
| 2 | 24 | 146 | 170 | 2.27 | [-0.33, 0.24] |
| 3 | 19 | 96 | 115 | 3.07 | [-2.24, -0.73] |
| 4-10 | 25 | 482 | 507 | 5.75 | various |

Wall/Roof 비율이 2:1 ~ 10:1. 항공 oblique 촬영에서 측면 관측 빈도가 높기 때문.

### 산출물
- `results/phase4/part2/buildings_colored.ply`: 건물별 색상 구분 PLY
- `results/phase4/part2/building_*.ply`: 개별 건물 PLY

## Part 3: TSDF Mesh 추출 + 위상 분석

### 전체 Mesh 토폴로지

| 항목 | 값 |
|------|------|
| Vertices | 79,719 |
| Faces | 141,217 |
| Watertight | No |
| Euler number | -22 |
| Components | 637 |
| Holes | 1,167 |
| Largest hole | 5,973 edges |

### 건물 인스턴스별 Watertight 분석

전체 mesh watertight 여부는 야외 씬에서 무의미 — **건물별로** 분석해야 한다.
TSDF mesh를 building primitive에 가장 가까운 face (threshold=0.1m) 기준으로 분할:

| Building | Verts | Faces | Watertight | Components | WT comps |
|----------|-------|-------|------------|------------|----------|
| 0 | 13,900 | 23,858 | No | 124 | 18 |
| 1 | 10,845 | 19,022 | No | 74 | 7 |
| 2 | 4,737 | 8,751 | No | 16 | 6 |
| 3 | 6,773 | 10,908 | No | 82 | 13 |
| 4 | 1,473 | 2,486 | No | 20 | 1 |
| 5 | 2,412 | 4,225 | No | 8 | 1 |
| 6 | 2,673 | 4,416 | No | 18 | 0 |
| 7 | 4,653 | 8,123 | No | 29 | 10 |
| 8 | 966 | 1,716 | No | 1 | 0 |
| 9 | 4,297 | 7,115 | No | 63 | 21 |
| 10 | 1,103 | 1,708 | No | 12 | 1 |

**결론**: 건물 인스턴스별로도 모두 NOT watertight. TSDF mesh는 야외 항공 촬영의 sparse view, 가림(occlusion), 미관측 영역으로 인해 건물별로도 완전한 닫힌 표면을 생성하지 못한다. 따라서 "방향 C: TSDF mesh로 topology 보정"의 한계가 여기서 확인됨 — TSDF mesh 자체가 watertight가 아니므로 이를 기반으로 CityGML Solid의 305 오류를 해결하기는 어렵다.

### 프리미티브-Mesh 대응

| 항목 | 값 |
|------|------|
| Coverage (5cm threshold) | 62.1% |
| Nearest dist mean | 0.049m |
| Nearest dist median | 0.043m |
| Nearest dist 95pct | 0.105m |

### 산출물
- `results/phase4/part3/tsdf_mesh.ply`: TSDF mesh
- `results/phase4/part3/tsdf_mesh_class_colored.ply`: 클래스 색상 mesh
- `results/phase4/part3/tsdf_mesh_gap_heatmap.ply`: 프리미티브 거리 히트맵

## Part 4: CityGML LOD2 변환

### Merging 알고리즘

**v1 (greedy single-pass, cos_thresh=0.95)**: 첫 프리미티브 기준으로만 비교하여 transitive 관계를 포착하지 못함 → 620 surfaces (과도한 분할).

**v2 (connected components, cos_thresh=0.85)**: 동일 클래스 + cos(normal)>0.85 + 공간 근접 조건으로 그래프 구성 → scipy connected_components로 연결 요소 추출. Transitive merging 지원.

| Building | v1 surfaces | v2 surfaces | 원본 primitives |
|----------|-------------|-------------|-----------------|
| 0 | 189 | 28 | 462 |
| 1 | 129 | 31 | 347 |
| 2 | 47 | 10 | 170 |
| 3 | 45 | 19 | 115 |
| 4 | 21 | 7 | 42 |
| 5 | 31 | 16 | 53 |
| 6 | 29 | 7 | 56 |
| 7 | 52 | 13 | 176 |
| 8 | 11 | 3 | 35 |
| 9 | 51 | 14 | 123 |
| 10 | 15 | 5 | 22 |
| **Total** | **620** | **153** | **1601** |

v2에서 surface 수가 75% 감소 (620→153). 평균 surface당 ~10 primitives가 merge되어 convex hull polygon 크기가 대폭 증가.

### 산출물
- `results/phase4/part4/all_buildings.city.json`: CityJSON 1.1 (v2)
- `results/phase4/part4/all_buildings.gml`: CityGML 2.0 XML (v2)
- `results/phase4/part4/citygml/building_*.gml`: 개별 CityGML
- `results/phase4/part4/obj/building_*.obj`: OBJ (시각 확인용)

## Part 5: val3dity 검증

### v2 결과 (connected components merging)

| 항목 | 값 |
|------|------|
| val3dity version | 2.6.0 |
| Features validated | 11 |
| Valid | 0 (0%) |
| Invalid | 11 (100%) |
| 305 (MULTIPLE_CONNECTED_COMPONENTS) | 10 buildings |
| 301 (TOO_FEW_POLYGONS) | 1 building (Building 8, 3 surfaces) |

**Polygon-level 오류(1xx, 2xx)는 없음** — 개별 surface polygon은 모두 유효. 오류는 Shell-level(3xx)에만 존재:
- **305**: Surface들이 edge를 공유하지 않아 closed shell을 구성하지 못함. 프리미티브 기반 polygon들은 각각 독립적으로 생성되므로, 인접 surface 간 edge 공유가 자연스럽게 발생하지 않음.
- **301**: Building 8은 aggressive merging으로 3개 surface(1 roof + 1 wall + 1 ground)만 남아 Solid 최소 요건 미충족.

### v1 대비 변화
| 항목 | v1 | v2 |
|------|-----|-----|
| Total surfaces | 620 | 153 |
| 305 errors | 11 buildings | 10 buildings |
| 301 errors | 0 | 1 (Building 8) |
| 1xx/2xx errors | 0 | 0 |

305는 merging과 무관한 근본적 문제 — surface 경계 확장 (plane-plane intersection) 또는 CSG 기반 접근이 필요.

## Part 6: 위상 보정 프로토타입

단순 connecting quad 삽입 시도 → 11→357 errors로 악화 (104 self-intersection + 203 non-planar).

효과적 보정 방향:
1. Plane-plane intersection line 기반 surface 경계 확장
2. CSG boolean intersection으로 정밀 shell 구성
3. Per-building TSDF mesh도 watertight가 아니므로, mesh 기반 보정은 한계 (Part 3에서 확인)

## Go/No-Go

### 달성
- End-to-end 파이프라인: Checkpoint → CityGML LOD2 + CityJSON
- val3dity 인프라: 자동 검증 + 오류 분류
- Polygon-level 유효 (1xx/2xx 오류 없음), Shell-level 305만 잔존
- Connected components merging으로 surface 수 75% 감소 (620→153)
- Per-building TSDF watertight 분석 → mesh 기반 보정 한계 확인

### 주요 한계
- **305 (shell connectivity)**: surfaces가 edge를 공유하지 않아 closed solid 불가. Phase 4 고도화 과제.
- **법선 품질**: Wall gravity align=0.515 (이상: 0.0), Roof gravity align=0.406 (flat이면 이상: 1.0). L_mutual로 개선 가능 영역.
- **TSDF mesh**: 건물별로도 watertight 아님 → mesh 기반 topology 보정에 직접 활용 어려움.

### Go: Phase 3-B'-Step2 + Phase 4 고도화 병행
1. Phase 3-B'-Step2: L_mutual 재실험 → wall/roof 법선 정렬 → CityGML 품질 직접 영향
2. Phase 4 고도화: plane intersection 기반 위상 보정 → 305 해결
3. City3D 비교
