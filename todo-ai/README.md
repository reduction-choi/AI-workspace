# Purpose-Action Relevance AI

**목적(Purpose)** 과 **행위(Action)** 사이의 연관성을 0~1 사이의 실수로 평가하는 AI 시스템입니다.

---

## 아키텍처 개요

```
[목적 텍스트] ──► Embedding Backend ──► Purpose Projection Layer ──┐
                                                                    ├──► Cross-Attention Fusion ──► MLP Head ──► score ∈ [0, 1]
[행위 텍스트] ──► Embedding Backend ──► Action  Projection Layer ──┘
```

### Stage 1: Dual Embedding (독립적 임베딩)

| 항목 | 내용 |
|------|------|
| 기본 모델 | `sentence-transformers/all-MiniLM-L6-v2` (로컬, 무료) |
| 대안 모델 | `text-embedding-3-small` (OpenAI API) |
| 특징 | 목적과 행위 각각 **독립적**으로 임베딩 → 서로 다른 Projection Layer 통과 |

두 임베딩은 완전히 분리된 Projection Layer (`Linear → GELU → LayerNorm`)를 거치므로,  
각 tower가 독립적으로 학습됩니다.

### Stage 2: Cross-Attention Fusion + MLP Head

단순한 cosine similarity 대신, **Cross-Attention**으로 두 벡터 사이의 상호작용을 학습합니다:

- 목적 벡터가 행위 벡터에 **어텐션**을 적용하고
- 행위 벡터가 목적 벡터에 **어텐션**을 적용한 뒤
- 두 결과를 concat → MLP → Sigmoid → 연관성 점수

이 방식이 단순 distance 계산보다 우수한 이유:
- 두 임베딩의 **어느 차원이 서로 관련**되는지를 학습
- 비대칭적 관계 표현 가능 ("목적 A에 행위 B가 기여하는가"와 "목적 B에 행위 A가 기여하는가"는 다름)
- 학습 데이터로 미세 조정 가능

---

## 설치

```bash
pip install -r requirements.txt
```

OpenAI 임베딩 사용 시:
```bash
pip install openai
export OPENAI_API_KEY="sk-..."
```

---

## 학습 데이터 형식

`sample_data.json` 참고. JSON 배열, 각 항목의 구조:

```json
{
  "purpose": "일본어 실력 향상",
  "action":  "자막 없이 일본 애니메이션 시청",
  "label":   0.75
}
```

| 필드 | 설명 |
|------|------|
| `purpose` | 목적 설명 (자연어) |
| `action` | 행위 설명 (자연어) |
| `label` | 연관성 점수. `0.0`~`1.0` 실수 또는 `0`/`1` 이진값 모두 허용 |

---

## 사용법

### 학습

```bash
# 기본 (sentence-transformers, 30 epoch)
python main.py train --data sample_data.json

# 세부 설정
python main.py train \
  --data     my_data.json \
  --save     my_model.pt \
  --backend  sentence-transformers \
  --epochs   50 \
  --batch-size 32 \
  --lr       2e-4

# OpenAI 임베딩 사용
python main.py train --backend openai --data my_data.json
```

### 단일 점수 출력

```bash
python main.py score \
  --purpose "일본어 실력 향상" \
  --action  "자막 없이 일본 애니메이션 시청"
```

출력 예시:
```
목적  : 일본어 실력 향상
행위  : 자막 없이 일본 애니메이션 시청
연관성: 0.742  (높음 🟡)
```

### 대화형 모드

```bash
python main.py interactive
```

### Python API

```python
from inference import RelevanceAI

ai = RelevanceAI.load("relevance_model.pt")

# 단일 점수
score = ai.score("체중 감량", "매일 30분 유산소 운동")
print(score)   # 0.89

# 사람이 읽기 편한 설명
print(ai.describe("체중 감량", "야식으로 치킨 주문"))

# 배치 처리
pairs = [
    ("일본어 실력 향상", "일본 원서 소설 읽기"),
    ("일본어 실력 향상", "한국 예능 시청"),
]
scores = ai.score_batch(pairs)
```

---

## 연관성 점수 해석

| 범위 | 의미 |
|------|------|
| 0.8 ~ 1.0 | 매우 높음 🟢 – 행위가 목적에 직접적으로 기여 |
| 0.6 ~ 0.8 | 높음 🟡 – 간접적으로 상당히 기여 |
| 0.4 ~ 0.6 | 보통 🟠 – 어느 정도 연관 있음 |
| 0.2 ~ 0.4 | 낮음 🔴 – 약한 연관성 |
| 0.0 ~ 0.2 | 매우 낮음 ⚫ – 사실상 무관 |

---

## 파일 구조

```
purpose_action_ai/
├── model.py          # 모델 아키텍처 (EmbeddingBackend, RelevanceNet)
├── train.py          # 데이터셋 & 학습 파이프라인
├── inference.py      # 추론 엔진 (RelevanceAI 클래스)
├── main.py           # CLI 진입점
├── sample_data.json  # 예시 학습 데이터 (44개 샘플)
├── requirements.txt  # 의존성 목록
└── README.md         # 이 문서
```

---

## 학습 팁

- **데이터 품질이 핵심**: 라벨을 `0.0`, `0.25`, `0.5`, `0.75`, `1.0` 5단계로 나눠서 일관성 있게 부여하면 학습이 안정적
- **최소 데이터**: 목적 카테고리당 10~20개 이상 권장; 카테고리 다양성이 일반화에 중요
- **에폭**: 데이터가 100개 이하면 50~100 에폭, 1000개 이상이면 20~30 에폭으로 시작
- **임베딩 캐싱**: `emb_cache/` 디렉토리에 임베딩이 캐싱되어 재학습 시 API 비용/시간 절약
