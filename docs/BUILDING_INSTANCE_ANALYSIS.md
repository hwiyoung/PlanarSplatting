# Building Instance Segmentation 분석

## 문제 정의

현재 파이프라인은 primitive 단위로 semantic class (roof/wall/ground)를 분류하지만,
**어느 건물의 roof/wall인지**는 구분하지 않는다.

CityGML LOD2는 `<Building>` 단위로 구성되므로, building instance 분리 없이는
의미 있는 LOD2 출력이 불가능하다.

```xml
<!-- 현재 가능: 장면 전체 = 하나의 Building (무의미) -->
<Building gml:id="scene">
  <RoofSurface>... 모든 지붕 ...</RoofSurface>
  <WallSurface>... 모든 벽 ...</WallSurface>
</Building>

<!-- 필요: 건물별 분리 -->
<Building gml:id="bldg_001">
  <RoofSurface>... 이 건물 지붕 ...</RoofSurface>
  <WallSurface>... 이 건물 벽 ...</WallSurface>
</Building>
<Building gml:id="bldg_002">
  ...
</Building>
```

---

## 현재 파이프라인에서 빠진 부분

```
이미지 → COLMAP → PlanarSplatting → semantic class per primitive → ???  → CityGML
                                          ↑ 여기까지 있음          ↑ 이게 없음
```

| 단계 | 현재 상태 | 필요한 것 |
|------|-----------|-----------|
| Phase 2-A | building zone (binary) | per-building instance mask |
| Phase 2-C/3 | roof/wall/ground class | class + building ID |
| Phase 4 | primitive 병합 → CityGML | building별 그룹핑 → per-building CityGML |

---

## 접근법 비교

### 접근 A: Phase 2-A에서 SAM instance 보존 + Phase 4 투영

**원리**: SAM이 이미 개별 건물을 검출하고 있으나 현재 binary merge로 정보 손실.
Instance mask를 보존하고, Phase 4에서 primitive에 투영.

**구현**:
1. `generate_segmentation.py`: per-building instance mask도 저장
   - `seg_maps/` (현재): 0=bg, 1=roof, 2=wall, 3=ground
   - `instance_maps/` (신규): 0=bg, 1=building_1, 2=building_2, ...
2. Phase 4: primitive center를 각 view에 투영 → instance_map lookup → multi-view majority vote

**장점**:
- 이미 가진 정보 활용 (SAM 추가 실행 불필요)
- 학습 파이프라인 수정 없음
- 구현 단순

**빈틈**:
1. **Multi-view instance 불일치**: view_0의 building_1과 view_15의 building_3이 같은 건물인지 모름
   - 해법: camera pose + depth로 2D mask를 3D에 역투영 → 3D 공간에서 IoU 매칭
   - 하지만 시점 변화가 크면 overlap이 적어 매칭 실패 가능
2. **SAM 검출 불안정**: 같은 건물이 view별로 1개/2개로 분할, 인접 건물 합쳐짐
3. **Primitive 투표 불안정**: wall primitive가 5 view에서 보이는데 각각 다른 instance

### 접근 B: Phase 4 기하학 기반 building 구성

**원리**: 학습된 primitive의 3D 위치 + semantic class만으로 건물을 구성.
SAM instance 없이 순수 기하학.

**구현**:
1. Roof primitive 그룹핑: class=roof → XZ 평면(=지면) 투영 → connected component
   - COLMAP 좌표계에서 gravity ≈ -Y이므로 XZ가 수평면
   - 각 connected component = 하나의 건물 지붕
2. Wall primitive 귀속: 각 wall의 상단 edge → 가장 가까운 roof component에 연결
3. Ground 투영: roof component boundary → 지면에 투영

**장점**:
- SAM instance 매칭 불필요
- 3D 기하 직접 활용 → multi-view 일관성 문제 없음
- EXPERIMENT_PLAN Phase 4의 기존 병합 로직과 자연스럽게 연결

**빈틈**:
1. **같은 높이 인접 건물**: 지붕이 XZ에서 연결되어 보이면 하나로 합쳐짐
   - 성수동: 건물 높이가 다양하여 대부분 분리 가능, 하지만 연립주택은 문제
2. **복합 건물**: 높이 다른 여러 지붕을 가진 하나의 건물 → 여러 개로 쪼개짐
3. **Roof classification 정확도 의존**: roof가 wall로 오분류되면 building 누락
   - 현재 Phase 2-C mIoU=0.81, Roof IoU=0.70 → 개선 여지 있음

### 접근 C: Learnable instance embedding (3-way mutual)

**원리**: f_i를 확장하여 semantic class + instance embedding을 동시 학습.

**구현**:
```
현재:  f_i ∈ R^4      (semantic class logits)
확장:  f_i ∈ R^(4+D)  (semantic 4 + instance embedding D차원)
```
- Instance embedding에 discriminative loss 적용:
  - pull: 같은 건물 primitive embedding → 가깝게
  - push: 다른 건물 primitive embedding → 멀게
- Training 후: instance embedding에 k-means 또는 mean-shift → building ID

**3-way Mutual 관계**:
```
         Geometry (n_i, position)
           ↗          ↖
     기하→의미       기하→인스턴스
     의미→기하       인스턴스→기하
          ↙            ↘
Semantics (c_i) ←→ Instance (b_i)
              의미↔인스턴스
```

각 방향의 물리적 의미:
| 방향 | Prior | 손실 형태 |
|------|-------|-----------|
| Geometry → Instance | 공간적으로 가까운 primitive → 같은 건물 | spatial affinity |
| Instance → Geometry | 같은 건물 표면 → 닫힌 다면체 | L_adj 확장 |
| Semantics → Instance | 하나의 건물 = roof + walls | topological constraint |
| Instance → Semantics | 건물 최상부 = roof, 측면 = wall | height-based class prior |

**장점**:
- End-to-end, multi-view 일관성 자동 확보
- Ablation에서 "instance embedding 추가 효과" 검증 가능 → 추가 기여
- 3-way mutual이 작동하면 강력한 연구 기여

**빈틈**:
1. **Instance GT 필요**: pull/push loss에 "어떤 primitive가 같은 건물인지" 라벨 필요
   - 해법 A: SAM instance를 GT로 사용 (접근 A의 문제 계승)
   - 해법 B: Self-supervised (spatial proximity를 pseudo-label로) → 하지만 인접 건물 문제
2. **Number of instances 미지**: scene마다 건물 수가 다름 → 고정 class CE 사용 불가
   - Discriminative loss가 이를 해결하지만, 클러스터링 단계 필요
3. **연구 범위 확대**: L_mutual (2-way)이 핵심 기여인데 3-way로 확장하면 focus 분산
   - 하나의 논문에 너무 많은 기여 → 리뷰어 skepticism
   - 혹은 별도 챕터/논문으로 분리 가능
4. **구현 복잡도**: CUDA rasterizer 추가 수정, instance rendering path, discriminative loss

---

## 접근별 의존성 분석

```
                    현재 파이프라인 (변경 없음)
                    Phase 2-A → 2-B → 2-C → 3-A → 3-B → 3-C
                                                    |
                    ┌───────────────────────────────┘
                    ↓
            Phase 4 (CityGML)
                    |
        ┌───────────┼──────────────┐
        ↓           ↓              ↓
    접근 A        접근 B         접근 C
  SAM instance   기하 구성      Learnable
  보존+투영       (후처리)      instance
        |           |              |
        ↓           ↓              ↓
     [Phase 2-A   [Phase 4만    [Phase 3 수정
      수정 필요]   수정 필요]     + Phase 4]
```

**핵심**: 접근 A와 B는 현재 학습 파이프라인을 건드리지 않는다.
접근 C만 Phase 3 학습 자체를 수정한다.

---

## 성수동 테스트셋 특성과 적합성

드론 항공 이미지 (DJI, oblique ~49°):
- 저층~중층 건물 혼재 (3F~7F)
- 건물간 거리: 도로 폭 4~8m, 일부 연립/다세대는 벽 공유
- 지붕 형태: 대부분 평지붕 (flat roof), 일부 경사 지붕
- 100 학습 views, 3000 initial primitives → ~2600 최종

| 특성 | 접근 A (SAM) | 접근 B (기하) | 접근 C (Learnable) |
|------|-------------|-------------|-------------------|
| 분리된 건물 | 가능 | 가능 | 가능 |
| 인접 건물 (다른 높이) | SAM 의존 | **가능** (높이 차이) | 가능 |
| 인접 건물 (같은 높이) | SAM 의존 | **어려움** | GT 의존 |
| 연립주택 (벽 공유) | SAM 의존 | **어려움** | GT 의존 |
| 복합 건물 (여러 지붕) | SAM이 하나로 | 분리 위험 | GT 의존 |

---

## 단계적 전략 제안

### 전략 1: 현재 실험 유지 + Phase 4에서 후처리 (최소 변경)

```
Phase 3-A/B/C (현재 계획 그대로)
    ↓
Phase 4:
  Part 1: L_planar + L_adj (현재 계획)
  Part 2: Building grouping (접근 B: 기하학 기반)
    - Roof XZ 클러스터링 → building 후보
    - Wall 귀속 → building 완성
  Part 3: CityGML 변환 (per-building)
```

- **변경점**: Phase 4 Part 2에 building grouping 추가
- **리스크**: 인접 건물 분리 어려움
- **연구 기여**: L_mutual (2-way), CityGML은 응용

### 전략 2: SAM instance 보존 + 후처리 보강

```
Phase 2-A 수정: instance_maps 저장 (코드 수정 작음)
Phase 3-A/B/C (현재 계획 그대로)
    ↓
Phase 4:
  Part 2: Building grouping (접근 B + A 하이브리드)
    - 기본: 기하학 기반 (roof XZ 클러스터링)
    - 보강: SAM instance vote로 ambiguous case 해소
```

- **변경점**: Phase 2-A instance 저장 + Phase 4 하이브리드
- **리스크**: multi-view SAM 매칭의 불안정성
- **연구 기여**: L_mutual + 실용적 building extraction pipeline

### 전략 3: 3-way mutual (연구 확장)

```
Phase 3-A: L_mutual 구현 (현재 계획)
Phase 3-B: Ablation (현재 계획)
    ↓ 결과 확인 후 판단
Phase 3-D (신규): Instance embedding 추가
  - f_i 확장 (4 → 4+D)
  - Discriminative loss
  - Instance GT: SAM instance (Phase 2-A 수정 선행)
Phase 3-E (신규): Instance ablation
  - (j) Joint + Instance
  - (k) Joint only (instance 없음)
    ↓
Phase 4: CityGML (instance 정보 활용)
```

- **변경점**: Phase 3-D/E 신규 추가
- **리스크**: 범위 확대, 일정 리스크
- **연구 기여**: L_mutual (2-way) + instance-aware mutual (3-way) = 강한 기여

---

## Phase 4 CityGML 변환 시 Building Grouping 상세 (접근 B)

현재 EXPERIMENT_PLAN Phase 4 Part 2 (L595-600)의 변환 로직에 building grouping을 삽입:

```
현재:
1. argmax(softmax(f_i)) → roof/wall/ground
2. 같은 클래스 + cos(normal) > 0.95 + 중심 거리 < 2*r_i → 병합
3. alpha shape → 경계 다각형
4. CityGML XML

수정:
1. argmax(softmax(f_i)) → roof/wall/ground                     ← 동일
2. 같은 클래스 + cos(normal) > 0.95 + 중심 거리 < 2*r_i → 병합  ← 동일
3. ★ Building grouping:                                         ← 신규
   3a. 병합된 roof surface → XZ 투영 → connected component → building_id
   3b. 각 wall surface → 가장 가까운 roof component에 귀속
   3c. 각 building의 roof boundary → 지면 투영 → GroundSurface
4. Per-building alpha shape → 경계 다각형                        ← 수정
5. Per-building CityGML XML                                     ← 수정
   <Building gml:id="bldg_{id}">
     <RoofSurface>...</RoofSurface>
     <WallSurface>...</WallSurface>
     <GroundSurface>...</GroundSurface>
   </Building>
```

---

## 결정 필요 사항

1. **어느 전략으로 갈 것인가?** (전략 1/2/3)
   - 전략 1: 최소 변경, Phase 4 후처리만
   - 전략 2: SAM instance 보존 추가 (중간)
   - 전략 3: Learnable instance (연구 확장)

2. **언제 결정할 것인가?**
   - 지금: 전략 2 정도로 확정하고 Phase 2-A 수정 선행
   - Phase 3-B 이후: L_mutual 결과를 보고 전략 3 여부 판단
   - Phase 4 진입 시: 실제 CityGML 변환 시 필요한 수준 판단

3. **논문 구조에 어떻게 반영할 것인가?**
   - 핵심 기여 = L_mutual (2-way geometry ↔ semantics)
   - Building instance = "응용" 챕터 vs "확장" 챕터 vs 별도 논문

---

## 참고: 관련 연구

- **Mask3D** (ICRA 2023): 3D instance segmentation on point clouds, superpoint-based
- **SAM3D**: SAM을 multi-view로 확장하여 3D instance segmentation
- **PLANES4LOD2**: 이미지 기반 평면 추출 → 건물 모델, instance는 2D detection 기반
- **PolyFit**: 평면 primitive → 다면체 건물 재구축 (instance = connected component)
- **City3D**: Aerial LiDAR → per-building 3D 모델 (KD-tree 기반 building separation)
