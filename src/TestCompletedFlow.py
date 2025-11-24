"""
test_complete_flow.py
전체 파이프라인 테스트: 분류 → 검색 → GPT 답변
"""

import os
from dotenv import load_dotenv
from s4_EmbeddingManager import EmbeddingManager
from s5_LegalSearchEngine import LegalSearchEngine
from s61_QueryClassifier import QueryClassifier
from s62_GPTLegalSearchSystem import EnhancedLegalQASystem
import json


def print_section(title: str):
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def test_classification(classifier: QueryClassifier):
    """1단계: 질문 분류"""
    print_section("1️⃣ 질문 분류 테스트")
    
    queries = [
        "제36조가 뭐야?",
        "비계 안전 기준은?",
        "우리 현장 3m 비계 괜찮나?",
        "용도변경 절차 알려줘",
        "비계 점검 체크리스트 만들어줘",
        "산업안전보건법과 건축법 차이는?"
    ]
    
    for query in queries:
        print(f"🔍 질문: '{query}'")
        result = classifier.classify(query)
        
        print(f"   유형: {result['query_type']}")
        print(f"   확신도: {result['confidence']:.0%}")
        print(f"   키워드: {result['key_entities']}")
        
        strategy = classifier.get_search_strategy(result['query_type'])
        print(f"   검색: {strategy['search_method']} (top_k={strategy['top_k']})\n")


def test_search(engine: LegalSearchEngine, classifier: QueryClassifier):
    """2단계: 검색"""
    print_section("2️⃣ 검색 테스트")
    
    queries = [
        "제36조가 뭐야?",
        "비계 안전 기준은?",
        "우리 현장 3m 비계 괜찮나?",
        "용도변경 절차 알려줘",
        "비계 점검 체크리스트 만들어줘",
        "산업안전보건법과 건축법 차이는?"
    ]
    
    for query in queries:
        
        classification = classifier.classify(query)
        strategy = classifier.get_search_strategy(classification['query_type'])
        
        print(f"질문: {query}")
        print(f"유형: {classification['query_type']}")
        print(f"검색: {strategy['search_method']}\n")
        
        if strategy['search_method'] == 'hybrid':
            results = engine.hybrid_search(query, top_k=strategy['top_k'])
        
        print(f"결과: {len(results)}건\n")
        for i, r in enumerate(results[:3], 1):
            print(f"[{i}] {r['metadata']['doc_name']} (p.{r['metadata']['page']})")
            print(f"    {r['content'][:150]}...\n")


def test_full_qa(qa_system: EnhancedLegalQASystem):
    """3단계: 전체 QA"""
    print_section("3️⃣ 전체 QA 시스템 테스트")
    
    test_cases = [
        ("제36조가 뭐야?", "법조문_조회"),
        ("비계 안전 기준은?", "일반_정보_검색"),
        ("우리 현장 3m 비계 괜찮나?", "상황별_컨설팅")
    ]
    
    for query, expected_type in test_cases:
        print(f"\n{'─'*80}")
        print(f"질문: {query}")
        print(f"예상 유형: {expected_type}")
        print(f"{'─'*80}\n")
        
        # format_for_user=True로 호출
        answer = qa_system.generate_answer(query, verbose=True, format_for_user=True)
        
        meta = answer.get("_meta", {})
        actual_type = meta.get("query_type", "")
        
        print(f"\n실제 유형: {actual_type}")
        print(f"일치: {'✅' if actual_type == expected_type else '❌'}")
        
        # 🆕 사용자 친화적 답변 먼저 출력
        print(f"\n{'='*80}")
        print("💬 [사용자용 답변]")
        print("="*80)
        if "user_friendly_answer" in answer:
            print(answer["user_friendly_answer"])
        else:
            print("(사용자 답변 없음)")
        
        # 🆕 구조화된 JSON 답변 출력 (개발자용)
        print(f"\n{'='*80}")
        print("📊 [개발자용 JSON 답변]")
        print("="*80)
        # _meta와 user_friendly_answer 제외한 핵심 답변만
        json_answer = {k: v for k, v in answer.items() 
                      if k not in ["_meta", "user_friendly_answer"]}
        print(json.dumps(json_answer, ensure_ascii=False, indent=2))
        
        # 참조 문서
        print(f"\n{'='*80}")
        print("📚 [참조 문서]")
        print("="*80)
        for i, s in enumerate(meta.get("sources", [])[:3], 1):
            score = s.get('relevance_score', 0)
            print(f"  [{i}] {s['doc_name']} (p.{s['page']}) - 관련도: {score:.3f}")


def main():
    print("="*80)
    print("🧪 전체 파이프라인 테스트")
    print("="*80)
    
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        print("❌ OPENAI_API_KEY 필요")
        return
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    vector_store_dir = os.path.join(project_root, "data", "vector_store", "construction_law")
    cache_dir = os.path.join(project_root, "data", "cache")
    
    print("\n🔧 초기화...")
    
    em = EmbeddingManager(OPENAI_API_KEY, "construction_law", cache_dir=cache_dir)
    index = em.load_index(os.path.join(vector_store_dir, "faiss_index.bin"))
    metadata = em.load_metadata(os.path.join(vector_store_dir, "metadata.json"))
    
    if not index or not metadata:
        print("❌ 인덱스 없음")
        return
    
    engine = LegalSearchEngine(index, metadata, em)
    classifier = QueryClassifier(OPENAI_API_KEY)
    qa_system = EnhancedLegalQASystem(engine, OPENAI_API_KEY)
    
    print("✅ 완료\n")
    
    try:
        test_classification(classifier)
        input("\n⏸️  Enter → 다음 단계...")
        
        test_search(engine, classifier)
        input("\n⏸️  Enter → 다음 단계...")
        
        test_full_qa(qa_system)
        
        print("\n" + "="*80)
        print("✅ 테스트 완료!")
        print("="*80)
        
    except Exception as e:
        print(f"\n❌ 오류: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()