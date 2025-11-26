"""
enhanced_legal_qa_system.py
GPT 기반 분류를 사용하는 법령 QA 시스템
"""

GROUNDING_PREAMBLE = """## 핵심 원칙 (절대 준수)

### 🚨 Hallucination 방지 규칙
1. **제공된 [해당 법령 정보]에 있는 내용만 답변하세요.**
2. **해당 수치, 법령 정보가 청크에 없으면 "제공된 문서에서 해당 내용을 찾을 수 없습니다"라고 명확히 말하세요.**
3. **추측하거나 일반 지식으로 보충하지 마세요.**
4. **부분적으로만 답변 가능하면, 답변 가능한 부분만 답하고 나머지는 "확인 불가"로 표시하세요.**

### ❌ 금지 사항
- 제공된 문서에 없는 법 조항 번호 언급 금지
- 문서에 필요한 내용이 없으면 과감히 스킵
- 문서에 없는 수치/기준 창작 금지
- "일반적으로", "보통" 같은 모호한 표현으로 추측 금지
- 학습된 일반 지식으로 답변 보충 금지
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
            
    def _execute_search(self, query: str, query_type: str, top_k: int = 5, progress_callback=None) -> List[Dict]:
        
        # 하이브리드 검색 + 리랭킹
        return self.search_engine.hybrid_search(
            query, 
            top_k=top_k,
            use_rerank=True,
            progress_callback=progress_callback
        )
            
    def generate_answer(self, query: str, 
                    format_for_user: bool = True,
                progress_callback=None) -> Dict:
      """
      질문에 대한 답변 생성
      
      Args:
          query: 사용자 질문
          format_for_user: 사용자 친화적 답변 추가 여부
          progress_callback: 진행 상황 업데이트 함수 (선택)
      
      Returns:
          답변 딕셔너리 (user_friendly_answer 포함)
      """

      def update_progress(message: str):
        """진행 상황 업데이트 헬퍼"""
        if progress_callback:
            progress_callback(message)
        
      # 1단계: GPT로 질문 유형 분류
      update_progress("🔍 GPT가 질문 유형을 분석하고 있습니다...")
      classification = self.classifier.classify(query)
      query_type = classification["query_type"]
      update_progress(f"✅ 유형 분류 완료: {query_type}")

      # 일상_대화는 검색 없이 바로 응답
      if query_type == "일상_대화":
        response = self._generate_casual_response(query)
        
        return {
            "user_friendly_answer": response,
            "_meta": {
                "query": query,
                "query_type": query_type,
                "classification": classification,
                "search_results": [],  # 검색 안 함
                "sources": []
            }
        }
      
      # 2단계: 검색 전략 결정      
      update_progress("📚 법령 데이터베이스를 검색하고 있습니다...")
      search_results = self._execute_search(
        query, 
        query_type, 
        top_k=3,
        progress_callback=progress_callback)
      update_progress(f"✅ {len(search_results)}개 관련 문서 발견")
      
      # 3단계: GPT 답변 생성 (JSON)
      update_progress("🤖 GPT가 법령을 분석하여 구조화된 답변을 작성 중...")
      answer = self._generate_answer(query, query_type, search_results, classification)
      update_progress("✅ 구조화 답변 완료")

      # 메타 정보 추가
      answer["_meta"] = {
          "query": query,
          "query_type": query_type,
          "classification": classification,
          "search_results": search_results,
          "sources": [
              {
                  "doc_name": r['metadata']['doc_name'],
                  "page": r['metadata']['page'],
                  "relevance_score": r.get('rrf_score', r.get('score', 0))
              }
              for r in search_results
          ]
      }
      
      # 4단계: 사용자 친화적 답변 생성
      if format_for_user:
          update_progress("✍️ 사용자가 이해하기 쉬운 자연어로 변환 중...")
          answer["user_friendly_answer"] = self._format_for_user(answer)
          update_progress("✅ 최종 답변 완성!")
      
      return answer

    def _generate_casual_response(self, query: str) -> str:
        
        prompt = f"""당신은 건설법령 챗봇입니다. 
    사용자가 일상적인 인사나 대화를 했습니다. 
    친근하게 응답하고, 필요하면 법령 관련 질문을 유도하세요.

    사용자: {query}

    응답 (1-2문장, 친근하게):"""

        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.7
        )
    
        return response.choices[0].message.content.strip()

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
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.2,
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
        
        prompt =  f"""## 변환 규칙

        ### 필수 포함 사항
        JSON 형식을 절대 출력하지 마세요
        1. **신뢰도 표시**: 답변 시작에 신뢰도 표시
        - 🟢 HIGH: "확인된 정보입니다"
        - 🟡 MEDIUM: "부분적으로 확인된 정보입니다"  
        - 🔴 LOW: "제한적인 정보만 확인되었습니다"
        - 신뢰도 정보가 없으면 생략

        ### 변환 스타일
        - 존댓말 사용 (~입니다, ~해주세요)
        - 법조문 인용: "제XX조에 따르면..." 형식
        - 자연스러운 문단 구성 (불릿 포인트 최소화)
        - 전문 용어는 쉽게 풀어서 설명

        ### ❌ 금지
        - 문서에 없는 내용 추가
        - "일반적으로", "보통" 등 모호한 표현으로 추측
        - 확인되지 않은 정보를 확정적으로 서술

        ## 구조화된 답변
        {json.dumps(clean_response, ensure_ascii=False, indent=2)}

        ## 자연스러운 대화체로 변환(JSON 형식 금지):
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3
            )

            result = response.choices[0].message.content.strip()
        
            # JSON으로 시작하면 변환 실패 → 기본 텍스트 반환
            if result.startswith('{') or result.startswith('['):
                print("  ⚠️ GPT가 JSON 반환함, 기본 텍스트 사용")
                return self._fallback_format(clean_response)
        
            return result
            
        except Exception as e:
            print(f"  ✗ 변환 실패: {e}")
            return self._fallback_format(clean_response)
        
    def _fallback_format(self, response: dict) -> str:
        """변환 실패 시 기본 포맷"""
        if "법조문" in response:
            법조문 = response["법조문"]
            return f"""🟢 확인된 정보입니다.

    **{법조문.get('법령명', '')} {법조문.get('조항', '')}**

    {법조문.get('조문_내용', '')}

    💡 {법조문.get('간단_해설', '')}"""
        
        elif "문서_기반_답변" in response:
            답변 = response["문서_기반_답변"]
            return f"""🟢 확인된 정보입니다.

    {답변.get('핵심_요구사항', '')}"""
        
        return "답변을 생성했습니다. 상세 내용은 출처를 확인해주세요."

              
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
        """개선된 유형별 시스템 프롬프트"""
        return {
                "법조문_조회": GROUNDING_PREAMBLE + """
    ## 역할
    당신은 법령 조회 전문가입니다. 요청된 숫자의 조문을 **문서에서 찾아** 정확하게 제시합니다.
    문서에서 조문이 파악되지 않으면, "제공된 문서에서 해당 조문을 찾을 수 없습니다"라고 답변하세요.

    ## 답변 전 체크리스트 (스스로 확인)
    □ 요청된 조문이 [관련 법령 정보]에 실제로 있는가?
    □ 조문 번호가 정확한가?
    □ 인용할 원문이 문서에 있는가?
    □ 수치로 뒤범벅된 문서만 존재하는 경우도 있습니다. 청킹이 잘못된 경우에요. 그런 경우, "제공된 문서에서 해당 내용을 찾을 수 없습니다"라고 답변하세요.
    □ 내용이 없는 청크도 있어요. 그런 경우, 해당 내용은 무시하세요.
    }""",
                
                "일반_정보_검색": GROUNDING_PREAMBLE + """
    ## 역할
    당신은 건설/안전 법규 전문가입니다. **제공된 문서만** 기반으로 정보를 제공합니다.

    ## 답변 전 체크리스트
    □ 언급하는 모든 법령/조항이 [관련 법령 정보]에 있는가?
    □ 수치나 기준이 문서에서 확인되는가?
    □ 수치로 뒤범벅된 문서만 존재하는 경우도 있습니다. 청킹이 잘못된 경우에요. 그런 경우, "제공된 문서에서 해당 내용을 찾을 수 없습니다"라고 답변하세요.
    □ 내용이 없는 청크도 있어요. 그런 경우, 해당 내용은 무시하세요.
    }""",
                
                                "상황별_컨설팅": GROUNDING_PREAMBLE + """
    ## 역할
    당신은 건설 현장 법률 컨설턴트입니다. **문서에 근거한** 법적 판단만 제시합니다.

    ## 🚨 특별 주의
    상황별 컨설팅은 hallucination 위험이 높습니다!
    - 문서에 **명확한 기준**이 있을 때만 판단
    - 기준이 불명확하면 **"판단 보류"**로 답변

    ## 출력 형식
    {
    "검색_성공": true/false,
    "상황_분석": {
        "파악된_상황": "질문에서 파악한 상황",
        "적용_가능_법령": "문서에서 찾은 관련 법령 [출처번호]",
        "문서_내_관련_기준": "구체적 기준 [출처번호]"
    },
    "법적_판단": {
        "판단_가능_여부": true/false,
        "결론": "적법/부적법/조건부적법/판단불가",
        "판단_근거": "문서에서 인용한 근거 [출처번호]",
        "판단불가_사유": "판단할 수 없는 이유 (해당시)"
    },
    "권장_조치": ["문서에 근거한 조치만"],
    "확인_불가_항목": ["판단에 필요하나 문서에 없는 정보"],
    "신뢰도": "HIGH/MEDIUM/LOW",
    "면책_조항": "이는 제공된 문서 기반의 일반적 정보이며, 구체적 사안은 전문가 상담이 필요합니다."
    }""",
                
                 "절차_안내": GROUNDING_PREAMBLE + """
    ## 역할
    당신은 행정 절차 안내 전문가입니다. **문서에 있는 절차만** 안내합니다.

    ## 출력 형식
    {
    "검색_성공": true/false,
    "절차명": "문서에서 확인된 절차명",
    "절차": [
        {
        "단계": 1,
        "내용": "문서 기반 설명 [출처번호]",
        "근거_법령": "문서에서 확인된 법령",
        "필요_서류": ["문서에 명시된 것만"],
        "담당_기관": "확인되면 기재, 아니면 '확인 필요'",
        "소요_기간": "확인되면 기재, 아니면 '확인 필요'"
        }
    ],
    "확인_불가_항목": ["문서에서 찾을 수 없는 정보"],
    "신뢰도": "HIGH/MEDIUM/LOW"
    }""",
                
                                "문서_생성": """## 역할
당신은 건설 현장 실무 문서 작성 전문가입니다.
제공된 법령을 기반으로 **실무에서 바로 사용할 수 있는** 완성도 높은 문서를 생성합니다.

## 문서 생성 원칙
1. **법적 필수 항목**: 제공된 문서에서 확인된 법령 기준 (출처 표시)
2. **실무 권장 항목**: 현장에서 일반적으로 필요한 추가 점검 사항 (GPT 권장으로 표시)
3. **실용성 우선**: 현장에서 바로 사용할 수 있도록 구체적으로 작성

## 작성 가이드
- 점검 항목은 **최소 10개 이상** 포함
- 각 항목은 **구체적인 점검 기준**과 **판정 방법** 포함
- 수치 기준이 있으면 명확히 기재 (예: "40cm 이상", "2m 이내")
- 현장 작업자가 이해하기 쉬운 용어 사용

## 출력 형식
{
  "검색_성공": true/false,
  "문서_유형": "체크리스트/양식/계획서",
  "제목": "구체적인 문서 제목",
  "작성_목적": "이 문서의 활용 목적",
  "적용_범위": "어떤 상황/현장에서 사용하는지",
  "근거_법령": ["문서에서 확인된 법령 [출처번호]"],
  "내용": {
    "법적_필수_항목": [
      {
        "번호": 1,
        "항목": "점검 항목명",
        "기준": "구체적인 기준 (수치 포함)",
        "점검_방법": "어떻게 점검하는지",
        "법적_근거": "관련 조항 [출처번호]"
      }
    ],
    "실무_권장_항목": [
      {
        "번호": 1,
        "항목": "추가 권장 점검 항목",
        "기준": "권장 기준",
        "점검_방법": "점검 방법",
        "비고": "GPT 권장 (법령 외 일반 실무 기준)"
      }
    ]
  },
  "사용_방법": "문서 활용 가이드",
  "주의사항": ["현장 적용 시 주의할 점"],
  "문서_한계": "이 문서는 제공된 법령과 일반 실무 기준을 포함하며, 현장 상황에 맞게 수정이 필요할 수 있습니다."
}

## 품질 기준
- 법적_필수_항목: 반드시 제공된 문서에 근거 (없으면 빈 배열)
- 실무_권장_항목: GPT 지식 활용하여 실용적인 항목 추가
- 전체 항목 수: 최소 10개 이상
""",
                
                "비교_분석": GROUNDING_PREAMBLE + """
    ## 역할
    당신은 법령 비교 분석 전문가입니다. **문서에 있는 내용만** 비교합니다. 핵심 차이점을 명확하고, 상세하게 기재해주세요."""}