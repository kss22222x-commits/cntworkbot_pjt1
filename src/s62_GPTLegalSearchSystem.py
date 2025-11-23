"""
enhanced_legal_qa_system.py
GPT 기반 분류를 사용하는 법령 QA 시스템
"""

from openai import OpenAI
import json
from typing import Dict, List
from s61_QueryClassifier import QueryClassifier


class EnhancedLegalQASystem:
    """유형별 답변을 제공하는 고급 QA 시스템"""
    
    def __init__(self, search_engine, openai_api_key: str):
        self.search_engine = search_engine
        self.client = OpenAI(api_key=openai_api_key)
        self.classifier = QueryClassifier(openai_api_key)
        self.response_templates = self._load_response_templates()
    
    def _execute_search(self, query: str, strategy: Dict) -> List[Dict]:
        """검색 전략에 따라 검색 실행"""
        method = strategy['search_method']
        top_k = strategy['top_k']
        
        if method == 'hybrid':
            return self.search_engine.hybrid_search(query, top_k=top_k)
    
    def generate_answer(self, query: str, verbose: bool = True, 
                    format_for_user: bool = True) -> Dict:
      """
      질문에 대한 답변 생성
      
      Args:
          query: 사용자 질문
          verbose: 진행 상황 출력 여부
          format_for_user: 사용자 친화적 답변 추가 여부
      
      Returns:
          답변 딕셔너리 (user_friendly_answer 포함)
      """
      if verbose:
          print(f"\n{'='*80}")
          print(f"💬 질문: {query}")
          print("="*80)
      
      # 1단계: GPT로 질문 유형 분류
      if verbose:
          print("\n[1단계] GPT 질문 유형 분류 중...")
      
      classification = self.classifier.classify(query)
      query_type = classification["query_type"]
      
      if verbose:
          print(f"  ✓ 유형: {query_type}")
          print(f"  ✓ 확신도: {classification['confidence']:.2f}")
          print(f"  ✓ 이유: {classification['reasoning']}")
          if classification['key_entities']:
              print(f"  ✓ 핵심 키워드: {', '.join(classification['key_entities'])}")
      
      # 2단계: 검색 전략 결정
      if verbose:
          print("\n[2단계] 검색 전략 결정...")
      
      search_strategy = self.classifier.get_search_strategy(query_type)
      
      if verbose:
          print(f"  ✓ 검색 방법: {search_strategy['search_method']}")
          print(f"  ✓ 결과 수: {search_strategy['top_k']}")
      
      # 3단계: 문서 검색
      if verbose:
          print("\n[3단계] 문서 검색 중...")
      
      search_results = self._execute_search(query, search_strategy)
      
      if verbose:
          print(f"  ✓ {len(search_results)}개 문서 검색 완료")
      
      # 4단계: GPT 답변 생성 (JSON)
      if verbose:
          print("\n[4단계] GPT 구조화 답변 생성 중...")
      
      answer = self._generate_answer(query, query_type, search_results, classification)
      
      # 메타 정보 추가
      answer["_meta"] = {
          "query": query,
          "query_type": query_type,
          "classification": classification,
          "search_strategy": search_strategy,
          "sources": [
              {
                  "doc_name": r['metadata']['doc_name'],
                  "page": r['metadata']['page'],
                  "relevance_score": r.get('rrf_score', r.get('score', 0))
              }
              for r in search_results
          ]
      }
      
      if verbose:
          print("  ✓ 구조화 답변 생성 완료")
      
      # 5단계: 사용자 친화적 답변 생성 (추가!)
      if format_for_user:
          if verbose:
              print("\n[5단계] 사용자 친화적 답변 변환 중...")
          
          answer["user_friendly_answer"] = self._format_for_user(answer)
          
          if verbose:
              print("  ✓ 사용자 답변 생성 완료")
      
      if verbose:
          print("="*80 + "\n")
      
      return answer


    def _generate_answer(self, query: str, query_type: str, 
                        search_results: List[Dict], classification: Dict) -> Dict:
        """GPT로 답변 생성"""
        
        # 컨텍스트 구성
        context = self._build_context(search_results, query_type)
        
        # 시스템 프롬프트
        system_prompt = self.response_templates[query_type]
        
        # 사용자 메시지
        key_entities_str = ""
        if classification.get('key_entities'):
            key_entities_str = f"\n핵심 키워드: {', '.join(classification['key_entities'])}"
        
        user_message = f"""사용자 질문: {query}{key_entities_str}

    관련 법령 정보:
    {context}

    위 정보를 바탕으로 JSON 형식으로 답변해주세요. 확실하지 않거나, 모르는 항목의 경우 빈 값으로 남겨두세요."""
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            answer = json.loads(response.choices[0].message.content)
            return answer
            
        except Exception as e:
            print(f"  ✗ 답변 생성 실패: {e}")
            return {
                "error": str(e),
                "query": query,
                "query_type": query_type
            }


    def _format_for_user(self, json_response: dict) -> str:
        """JSON 답변을 자연스러운 대화체로 변환"""
        
        # _meta 제거한 답변만 사용
        clean_response = {k: v for k, v in json_response.items() if k != "_meta"}
        
        prompt = f"""
    다음 구조화된 법률 답변을 일반 사용자가 이해하기 쉬운 자연스러운 대화체로 변환해주세요.

    구조화된 답변:
    {json.dumps(clean_response, ensure_ascii=False, indent=2)}

    변환 규칙:
    1. **존댓말 사용** (~입니다, ~해주세요)
    2. **법조문 인용**: "제XX조에 따르면..." 형식으로 자연스럽게
    3. **문단 구성**: 불릿 포인트 대신 자연스러운 문단으로
    4. **전문 용어 설명**: 어려운 용어는 쉽게 풀어서
    5. **출처 표시**: 답변 끝에 "참고 법령: ..." 형태로 간단히

    예시:
    - ❌ "법조문: {{...}}" 
    - ✅ "산업안전보건법 제38조에 따르면 비계 작업 시..."

    - ❌ "주요 요구사항: ['항목1', '항목2']"
    - ✅ "주요 요구사항은 다음과 같습니다. 첫째, 항목1입니다. 둘째, 항목2입니다."

    자연스러운 대화체로 변환:
    """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content
            
        except Exception as e:
            print(f"  ✗ 사용자 답변 변환 실패: {e}")
            # 실패 시 JSON의 주요 내용만 간단히 반환
            if "법조문" in clean_response:
                return clean_response["법조문"].get("조문_내용", "답변 생성 실패")
            elif "주제" in clean_response:
                return clean_response.get("주제", "답변 생성 실패")
            else:
                return "답변 생성 중 오류가 발생했습니다."
              
    def _build_context(self, search_results: List[Dict], query_type: str) -> str:
        """검색 결과를 컨텍스트로 구성"""
        
        if not search_results:
            return "관련 문서를 찾을 수 없습니다."
        
        context_parts = []
        
        for i, result in enumerate(search_results, 1):
            meta = result['metadata']
            context_parts.append(
                f"[{i}] {meta['doc_name']} (p.{meta['page']})\n"
                f"{result['content']}\n"
            )
        
        return "\n".join(context_parts)
    
    def _load_response_templates(self) -> Dict:
        """유형별 시스템 프롬프트"""
        return {
            "법조문_조회": """당신은 법령 조회 전문가입니다.
요청된 조문을 정확하게 제시하고, 간단한 해설을 추가하세요.

출력 형식:
{
  "법조문": {
    "법령명": "문서명",
    "조항": "제○조",
    "조문_내용": "원문 그대로",
    "간단_해설": "핵심 내용만 2-3문장으로",
    "관련_조항": ["제○조", "제○조"]
  }
}""",
            
            "일반_정보_검색": """당신은 건설/안전 법규 전문가입니다.
관련 법령들을 종합하여 실무적으로 유용한 정보를 제공하세요.

출력 형식:
{
  "주제": "질문의 주제",
  "법적_근거": {
    "관련_법령": ["법령명 조항", "..."],
    "핵심_요구사항": "핵심 내용",
    "준수_방법": ["방법1", "방법2", "..."]
  },
  "실무_가이드라인": {
    "현장_점검_항목": ["항목1", "항목2"],
    "권장_조치": "추가 권장사항"
  }
}""",
            
            "상황별_컨설팅": """당신은 건설 현장 법률 컨설턴트입니다.
구체적 상황을 분석하고 법적 판단을 제시하세요.

출력 형식:
{
  "상황_분석": {
    "주요_변수": ["변수1", "변수2"],
    "적용_법령": "관련 법령"
  },
  "법적_판단": {
    "결론": "적법/부적법/조건부적법",
    "근거": "법적 근거",
    "필요_조치": ["조치1", "조치2"],
    "위반_시_리스크": "리스크 설명"
  },
  "권장_사항": "추가 권장사항",
  "면책_조항": "이는 일반적인 법적 정보이며, 구체적 사안은 전문가 상담이 필요합니다."
}""",
            
            "절차_안내": """당신은 행정 절차 안내 전문가입니다.
단계별로 명확하게 프로세스를 설명하세요.

출력 형식:
{
  "절차명": "절차 이름",
  "절차": [
    {
      "단계": 1,
      "내용": "단계 설명",
      "근거_법령": "관련 법령",
      "필요_서류": ["서류1", "서류2"],
      "담당_기관": "기관명",
      "소요_기간": "기간"
    }
  ],
  "주의사항": ["주의1", "주의2"]
}""",
            
            "문서_생성": """당신은 실무 문서 작성 전문가입니다.
법령에 근거한 실용적인 문서를 생성하세요.

출력 형식:
{
  "문서_유형": "체크리스트/양식/계획서",
  "제목": "문서 제목",
  "근거_법령": ["법령1", "법령2"],
  "내용": [
    {
      "번호": 1,
      "항목": "항목명",
      "기준": "기준 내용",
      "법적_근거": "관련 조항"
    }
  ],
  "사용_방법": "문서 활용 방법"
}""",
            
            "비교_분석": """당신은 법령 비교 분석 전문가입니다.
차이점을 명확히 구분하여 표로 제시하세요.

출력 형식:
{
  "비교_대상": ["대상1", "대상2"],
  "비교_항목": [
    {
      "항목": "비교 항목",
      "대상1": "설명",
      "대상2": "설명"
    }
  ],
  "핵심_차이점": "주요 차이점 요약",
  "실무_가이드": "실무적 선택 가이드"
}"""
        }