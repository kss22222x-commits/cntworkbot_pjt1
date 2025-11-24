"""
query_classifier.py
GPT 기반 질문 유형 분류기
"""

from openai import OpenAI
import json
from typing import Dict


class QueryClassifier:
    """GPT 기반 질문 유형 분류기"""
    
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)
        self.classification_prompt = self._create_classification_prompt()
    
    def _create_classification_prompt(self) -> str:
        """질문 분류용 시스템 프롬프트"""
        return """당신은 건설/법령 분야 질문을 정확하게 분류하는 전문가입니다.

# 질문 유형 정의

## 1. 법조문_조회
- **설명**: 특정 조항의 내용을 직접 요청하는 질문
- **특징**: 조항 번호를 명시하거나 특정 법조문을 요청
- **예시**:
  * "제36조가 뭐야?"
  * "산업안전보건기준 제5장 보여줘"
  * "건축법 제19조 내용 알려줘"

## 2. 일반_정보_검색
- **설명**: 특정 주제에 대한 법적 기준이나 일반 정보를 묻는 질문
- **특징**: "기준", "규정", "요구사항" 등 일반적 정보 요청
- **예시**:
  * "비계 안전 기준은?"
  * "굴착 작업 시 주의사항"
  * "화재 예방 관련 규정"

## 3. 상황별_컨설팅
- **설명**: 구체적인 현장 상황에 대한 법적 판단이나 조언을 요청
- **특징**: "우리", "현장", 구체적 수치, "해도 되나", "가능한가" 등 포함
- **예시**:
  * "우리 현장에서 비계를 3m 높이로 설치하는데 문제없나?"
  * "안전 관리자 1명으로 충분한가요? 현장 인원 50명입니다"

## 4. 절차_안내
- **설명**: 행정 절차나 프로세스를 단계별로 안내 요청
- **특징**: "절차", "방법", "어떻게", "순서", "신청" 등 포함
- **예시**:
  * "건축물 용도변경 절차 알려줘"
  * "안전 관리 계획서 제출은 어떻게 하나요?"

## 5. 문서_생성
- **설명**: 체크리스트, 양식, 템플릿 등 실무 문서 작성 요청
- **특징**: "만들어", "작성", "양식", "체크리스트", "템플릿" 등 포함
- **예시**:
  * "비계 점검 체크리스트 만들어줘"
  * "안전 관리 계획서 초안 작성해줘"

## 6. 비교_분석
- **설명**: 여러 법령, 개념, 방법 등을 비교하는 질문
- **특징**: "차이", "비교", "vs", "어떤 것", "어느 것" 등 포함
- **예시**:
  * "산업안전보건법과 건축법의 차이는?"
  * "강관비계와 시스템비계 규정 비교"

# 출력 형식

반드시 다음 JSON 형식으로만 답변하세요:

{
  "query_type": "위 6가지 유형 중 하나 (정확한 이름)",
  "confidence": 0.0~1.0 (분류 확신도),
  "reasoning": "이 유형으로 분류한 구체적인 이유",
  "key_entities": ["질문에서 추출한 핵심 키워드들"],
  "extracted_article": "제○조 (법조문 조회인 경우, 없으면 null)"
}

# 주의사항
- 한 가지 유형으로만 분류하세요 (중복 불가)
- 확신도는 정직하게 평가하세요
- key_entities는 검색에 도움되는 명사/전문용어만 추출
- 애매한 경우 "일반_정보_검색"을 기본값으로 사용
"""
    
    def classify(self, query: str) -> Dict:
        """
        질문 유형 분류
        
        Args:
            query: 사용자 질문
        
        Returns:
            분류 결과 딕셔너리
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": self.classification_prompt},
                    {"role": "user", "content": f"질문: {query}"}
                ],
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # 유효성 검사
            valid_types = [
                "법조문_조회", "일반_정보_검색", "상황별_컨설팅",
                "절차_안내", "문서_생성", "비교_분석"
            ]
            
            if result.get("query_type") not in valid_types:
                print(f"⚠ 잘못된 유형: {result.get('query_type')}, 기본값 사용")
                result["query_type"] = "일반_정보_검색"
                result["confidence"] = 0.5
            
            return result
            
        except Exception as e:
            print(f"✗ 분류 실패: {e}")
            return {
                "query_type": "일반_정보_검색",
                "confidence": 0.5,
                "reasoning": f"분류 실패: {str(e)}",
                "key_entities": [],
                "extracted_article": None
            }
    
    def get_search_strategy(self, query_type: str) -> Dict:
        """유형별 검색 전략 반환"""
        strategies = {
            "법조문_조회": {
                "search_method": "hybrid",
                "top_k": 5
            },
            "일반_정보_검색": {
                "search_method": "hybrid",
                "top_k": 10
            },
            "상황별_컨설팅": {
                "search_method": "hybrid",
                "top_k": 15
            },
            "절차_안내": {
                "search_method": "hybrid", 
                "top_k": 10
            },
            "문서_생성": {
                "search_method": "hybrid",
                "top_k": 20
            },
            "비교_분석": {
                "search_method": "hybrid",
                "top_k": 12
            }
        }
        
        return strategies.get(query_type, strategies["일반_정보_검색"])