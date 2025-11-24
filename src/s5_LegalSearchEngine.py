"""
s5_LegalSearchEngine.py
하이브리드 검색(벡터 + BM25) + 법령 특화 기능
"""

import numpy as np
import faiss
from typing import List, Dict, Optional
from rank_bm25 import BM25Okapi
import re
import json


"""
BM25Okapi
BM25Okapi는 문서 검색을 위한 키워드 기반 랭킹 알고리즘입니다.
검색어(query)와 문서들 간의 관련성 점수를 계산합니다
TF-IDF의 개선된 버전으로, 문서 길이 정규화와 단어 빈도 포화를 고려합니다
(BM25는 현실적인 검색을 위해 "문서 길이"와 "과도한 반복"의 영향을 조절해서 더 정확한 검색 결과를 제공합니다)

# 1. BM25 모델 생성
bm25 = BM25Okapi(tokenized_corpus)

# 2. 검색어로 문서 점수 계산
query = "안전 규정"
tokenized_query = query.split()

scores = bm25.get_scores(tokenized_query)
# 출력: [1.52, 0.93, 0.81]  # 각 문서의 관련성 점수

# 3. 가장 관련성 높은 문서 찾기
top_doc = bm25.get_top_n(tokenized_query, corpus, n=1)
# 출력: ['건설 안전 관리 규정을 준수해야 합니다']
"""

"""
bm25 = BM25Okapi(corpus)
```

이 한 줄이 하는 일:
```
1️⃣ 문서 빈도 (DF) 계산
   - "비계"가 몇 개 문서에 등장? → 127개
   - "안전"이 몇 개 문서에 등장? → 892개
   - "제57조"가 몇 개 문서에 등장? → 3개

2️⃣ 역문서 빈도 (IDF) 계산
   - IDF("비계") = log(1500/127) = 2.47
   - IDF("안전") = log(1500/892) = 0.52  ← 흔한 단어라 낮음
   - IDF("제57조") = log(1500/3) = 6.21  ← 희귀해서 높음

3️⃣ 평균 문서 길이 계산
   - avgdl = 전체 토큰 수 / 문서 수
   
4️⃣ 각 문서의 길이 저장
   - doc_lens = [350, 420, 380, ...]
"""

class LegalSearchEngine:
    """법령 특화 하이브리드 검색 엔진"""
    
    def __init__(self, 
                 faiss_index: faiss.Index,
                 metadata: List[Dict],  
                 embedding_manager=None):
        """
        Args:
            faiss_index: FAISS 인덱스
            metadata: 메타데이터 (chunks 정보 포함)
            embedding_manager: EmbeddingManager 인스턴스
        """
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.embedding_manager = embedding_manager
        
        # BM25 인덱스 생성
        print("\n🔧 BM25 인덱스 생성 중...")
        self.build_bm25_index()
        
        print("\n✓ LegalSearchEngine 초기화 완료")
        print(f"  - FAISS 벡터 수: {faiss_index.ntotal}")
        print(f"  - BM25 문서 수: {len(self.bm25_corpus)}")
    
    def tokenize_korean(self, text: str) -> List[str]:
        """
        한글 텍스트 토큰화
        이 함수는 한글 텍스트를 단어 단위로 분리하는 토큰화 함수입니다.
        정규표현식 \w+를 사용해서 텍스트에서 단어를 추출합니다
        모든 텍스트를 소문자로 변환합니다 (.lower())
        단어들을 리스트로 반환합니다
        """
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def build_bm25_index(self):
        """BM25 인덱스 구축 (metadata에서 직접)"""
        self.bm25_corpus = []
        
        for item in self.metadata:
            content = item.get('content', '')
            tokens = self.tokenize_korean(content)
            self.bm25_corpus.append(tokens)
        
        self.bm25 = BM25Okapi(self.bm25_corpus)
        print(f"  ✓ BM25 인덱스: {len(self.bm25_corpus)}개 문서")
    
    def contains_article(self, content: str, article: str) -> bool:
        """
        청크 내용에 특정 조(條)가 포함되어 있는지 확인
        
        Args:
            content: 청크 내용
            article: 조 번호 (예: "제36조")
        
        Returns:
            포함 여부
        """
        # 정규표현식으로 정확히 매칭
        pattern = re.escape(article) + r'(?:\s|[^\w가-힣]|$)'
        return bool(re.search(pattern, content))
    
    def contains_chapter(self, content: str, chapter: str) -> bool:
        """청크 내용에 특정 장(章)이 포함되어 있는지 확인"""
        pattern = re.escape(chapter) + r'(?:\s|[^\w가-힣]|$)'
        return bool(re.search(pattern, content))
    
    def vector_search(self, 
                     query: str,
                     top_k: int = 10,
                     filter_article: Optional[str] = None,
                     filter_chapter: Optional[str] = None) -> List[Dict]:
        """
        벡터 검색 + 법령 필터링 (텍스트 기반)
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            filter_article: 조 필터 (예: "제36조") - 청크 내용에서 검색
            filter_chapter: 장 필터 (예: "제2장") - 청크 내용에서 검색
        """
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager가 필요합니다.")
        
        # 쿼리 임베딩
        query_embedding = self.embedding_manager.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        # FAISS 검색 (필터링을 위해 더 많이 가져옴)
        search_k = top_k * 10 if (filter_article or filter_chapter) else top_k
        distances, indices = self.faiss_index.search(query_embedding, search_k)
        
        # 결과 구성 및 필터링
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= len(self.metadata):
                continue
            
            item = self.metadata[idx]
            content = item["content"]
            
            # 텍스트 기반 필터링
            if filter_article and not self.contains_article(content, filter_article):
                continue
            
            if filter_chapter and not self.contains_chapter(content, filter_chapter):
                continue
            
            result = {
                "rank": len(results) + 1,
                "chunk_id": item["chunk_id"],
                "content": content,
                "metadata": item["metadata"],
                "score": float(1 / (1 + distance)),
                "search_type": "vector"
            }
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def keyword_search(self,
                      query: str,
                      top_k: int = 10,
                      filter_article: Optional[str] = None,
                      filter_chapter: Optional[str] = None) -> List[Dict]:
        """
        키워드 검색 + 법령 필터링 (텍스트 기반)
        """
        query_tokens = self.tokenize_korean(query)
        scores = self.bm25.get_scores(query_tokens)
        
        # 스코어 정렬
        ranked_indices = np.argsort(scores)[::-1]
        
        # 결과 구성 및 필터링
        results = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue
            
            item = self.metadata[idx]
            content = item["content"]
            
            # 텍스트 기반 필터링
            if filter_article and not self.contains_article(content, filter_article):
                continue
            
            if filter_chapter and not self.contains_chapter(content, filter_chapter):
                continue
            
            result = {
                "rank": len(results) + 1,
                "chunk_id": item["chunk_id"],
                "content": content,
                "metadata": item["metadata"],
                "score": float(scores[idx]),
                "search_type": "keyword"
            }
            results.append(result)
            
            if len(results) >= top_k:
                break
        
        return results
    
    def reciprocal_rank_fusion(self,
                               vector_results: List[Dict],
                               keyword_results: List[Dict],
                               k: int = 60) -> List[Dict]:
        """RRF 알고리즘으로 결과 융합"""
        chunk_scores = {}
        chunk_data = {}
        
        for result in vector_results:
            chunk_id = result["chunk_id"]
            rank = result["rank"]
            rrf_score = 1 / (k + rank)
            
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score
            chunk_data[chunk_id] = result
        
        for result in keyword_results:
            chunk_id = result["chunk_id"]
            rank = result["rank"]
            rrf_score = 1 / (k + rank)
            
            chunk_scores[chunk_id] = chunk_scores.get(chunk_id, 0) + rrf_score
            if chunk_id not in chunk_data:
                chunk_data[chunk_id] = result
        
        sorted_chunks = sorted(chunk_scores.items(), key=lambda x: x[1], reverse=True)
        
        results = []
        for i, (chunk_id, score) in enumerate(sorted_chunks):
            result = chunk_data[chunk_id].copy()
            result["rank"] = i + 1
            result["rrf_score"] = float(score)
            result["search_type"] = "hybrid"
            results.append(result)
        
        return results
    
    def hybrid_search(self,
                     query: str,
                     top_k: int = 10,
                     filter_article: Optional[str] = None,
                     filter_chapter: Optional[str] = None) -> List[Dict]:
        """
        하이브리드 검색 + 법령 필터링
        
        Args:
            query: 검색 쿼리
            top_k: 반환할 결과 수
            filter_article: 조 필터 (예: "제36조")
            filter_chapter: 장 필터 (예: "제2장")
        """
        # 벡터 검색
        vector_results = self.vector_search(
            query, 
            top_k=top_k*2,
            filter_article=filter_article,
            filter_chapter=filter_chapter
        )
        
        # 키워드 검색
        keyword_results = self.keyword_search(
            query,
            top_k=top_k*2,
            filter_article=filter_article,
            filter_chapter=filter_chapter
        )
        
        # RRF 융합
        hybrid_results = self.reciprocal_rank_fusion(vector_results, keyword_results)
        
        return hybrid_results[:top_k]
    
def main():
    """테스트 코드"""
    import os
    from s4_EmbeddingManager import EmbeddingManager
    from dotenv import load_dotenv
    
    print("="*80)
    print("🔍 법령 특화 검색엔진 테스트")
    print("="*80)
    
    # 환경 변수 로드
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        print("\n✗ 오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return
    
    # 프로젝트 루트 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/
    project_root = os.path.dirname(current_dir)  # CNTWORKBOT_PJT1/
    
    # 경로 설정 (절대 경로)
    vector_store_dir = os.path.join(project_root, "data", "vector_store", "construction_law")
    cache_dir = os.path.join(project_root, "data", "cache")
    
    index_path = os.path.join(vector_store_dir, "faiss_index.bin")
    metadata_path = os.path.join(vector_store_dir, "metadata.json")
    
    print(f"\n프로젝트 루트: {project_root}")
    print(f"벡터 저장소: {vector_store_dir}")
    print(f"캐시 디렉토리: {cache_dir}")
    
    # EmbeddingManager 초기화
    em = EmbeddingManager(
        openai_api_key=OPENAI_API_KEY,
        institution="construction_law",
        cache_dir=cache_dir 
    )
    
    # 인덱스 로드
    index = em.load_index(index_path)
    metadata = em.load_metadata(metadata_path)
    
    if index is None or metadata is None:
        print("\n✗ 인덱스 또는 메타데이터를 찾을 수 없습니다.")
        print("먼저 s4_EmbeddingManager.py를 실행해주세요.")
        return
    
    # SearchEngine 초기화
    search_engine = LegalSearchEngine(
        faiss_index=index,
        metadata=metadata,
        embedding_manager=em
    )
    
    # 테스트 쿼리
    print("\n" + "="*80)
    print("📝 테스트 쿼리")
    print("="*80)
    
    query = "건폐율은 어떻게 계산하나요?"
    print(f"\n쿼리: {query}")
    
    # 하이브리드 검색
    results = search_engine.hybrid_search(query, top_k=5)
    
    print(f"\n검색 결과: {len(results)}건\n")
    for result in results:
        print(f"\n[{result['rank']}] {result['chunk_id']}")
        print(f"메타데이터:")
        print(json.dumps(result['metadata'], indent=2, ensure_ascii=False))
        print(f"\n내용 미리보기: {result['content'][:2000]}...")
        print("-" * 80)

if __name__ == "__main__":
    main()