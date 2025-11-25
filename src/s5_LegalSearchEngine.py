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
import pickle
import os

class LegalSearchEngine:
    """법령 특화 하이브리드 검색 엔진"""
    
    def __init__(self, 
                 faiss_index: faiss.Index,
                 metadata: List[Dict],  
                 embedding_manager=None,
                 bm25_index_path: Optional[str] = None):
        """
        Args:
            faiss_index: FAISS 인덱스
            metadata: 메타데이터 (chunks 정보 포함)
            embedding_manager: EmbeddingManager 인스턴스
            bm25_index_path: BM25 인덱스 저장 경로 (예: 'data/vector_store/bm25_index.pkl')
        """
        self.faiss_index = faiss_index
        self.metadata = metadata
        self.embedding_manager = embedding_manager
        self.bm25_index_path = bm25_index_path
        
        # BM25 인덱스 로드 또는 생성
        print("\n🔧 BM25 인덱스 초기화 중...")
        if bm25_index_path and os.path.exists(bm25_index_path):
            self.load_bm25_index()
        else:
            self.build_bm25_index()
            if bm25_index_path:
                self.save_bm25_index()
        
        print("\n✓ LegalSearchEngine 초기화 완료")
        print(f"  - FAISS 벡터 수: {faiss_index.ntotal}")
        print(f"  - BM25 문서 수: {len(self.bm25_corpus)}")
    
    def tokenize_korean(self, text: str) -> List[str]:
        """한글 텍스트 토큰화"""
        tokens = re.findall(r'\w+', text.lower())
        return tokens
    
    def build_bm25_index(self):
        """BM25 인덱스 구축 (metadata에서 직접)"""
        print("  ⏳ BM25 인덱스 새로 생성 중...")
        self.bm25_corpus = []
        
        for item in self.metadata:
            content = item.get('content', '')
            tokens = self.tokenize_korean(content)
            self.bm25_corpus.append(tokens)
        
        self.bm25 = BM25Okapi(self.bm25_corpus)
        print(f"  ✓ BM25 인덱스 생성 완료: {len(self.bm25_corpus)}개 문서")
    
    def save_bm25_index(self):
        """BM25 인덱스를 pickle로 저장"""
        if not self.bm25_index_path:
            print("  ⚠️  BM25 저장 경로가 지정되지 않았습니다.")
            return
        
        # 디렉토리 생성
        os.makedirs(os.path.dirname(self.bm25_index_path), exist_ok=True)
        
        # BM25 객체와 corpus를 함께 저장
        bm25_data = {
            'bm25': self.bm25,
            'corpus': self.bm25_corpus
        }
        
        with open(self.bm25_index_path, 'wb') as f:
            pickle.dump(bm25_data, f)
        
        print(f"  💾 BM25 인덱스 저장 완료: {self.bm25_index_path}")
    
    def load_bm25_index(self):
        """pickle에서 BM25 인덱스 로딩"""
        print(f"  📂 BM25 인덱스 로딩 중: {self.bm25_index_path}")
        
        with open(self.bm25_index_path, 'rb') as f:
            bm25_data = pickle.load(f)
        
        self.bm25 = bm25_data['bm25']
        self.bm25_corpus = bm25_data['corpus']
        
        print(f"  ✓ BM25 인덱스 로딩 완료: {len(self.bm25_corpus)}개 문서")
    
    def contains_article(self, content: str, article: str) -> bool:
        """청크 내용에 특정 조(條)가 포함되어 있는지 확인"""
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
        """벡터 검색 + 법령 필터링 (텍스트 기반)"""
        if not self.embedding_manager:
            raise ValueError("EmbeddingManager가 필요합니다.")
        
        query_embedding = self.embedding_manager.embed_text(query)
        query_embedding = query_embedding.reshape(1, -1).astype('float32')
        
        search_k = top_k * 10 if (filter_article or filter_chapter) else top_k
        distances, indices = self.faiss_index.search(query_embedding, search_k)
        
        results = []
        for idx, distance in zip(indices[0], distances[0]):
            if idx >= len(self.metadata):
                continue
            
            item = self.metadata[idx]
            content = item["content"]
            
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
        """키워드 검색 + 법령 필터링 (텍스트 기반)"""
        query_tokens = self.tokenize_korean(query)
        scores = self.bm25.get_scores(query_tokens)
        
        ranked_indices = np.argsort(scores)[::-1]
        
        results = []
        for idx in ranked_indices:
            if scores[idx] <= 0:
                continue
            
            item = self.metadata[idx]
            content = item["content"]
            
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
        """하이브리드 검색 + 법령 필터링"""
        vector_results = self.vector_search(
            query, 
            top_k=top_k*2,
            filter_article=filter_article,
            filter_chapter=filter_chapter
        )
        
        keyword_results = self.keyword_search(
            query,
            top_k=top_k*2,
            filter_article=filter_article,
            filter_chapter=filter_chapter
        )
        
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
    
    load_dotenv()
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        print("\n✗ 오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        return
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    vector_store_dir = os.path.join(project_root, "data", "vector_store", "construction_law")
    cache_dir = os.path.join(project_root, "data", "cache")
    
    index_path = os.path.join(vector_store_dir, "faiss_index.bin")
    metadata_path = os.path.join(vector_store_dir, "metadata.json")
    bm25_index_path = os.path.join(vector_store_dir, "bm25_index.pkl")  # 추가!
    
    print(f"\n프로젝트 루트: {project_root}")
    print(f"벡터 저장소: {vector_store_dir}")
    print(f"캐시 디렉토리: {cache_dir}")
    
    em = EmbeddingManager(
        openai_api_key=OPENAI_API_KEY,
        institution="construction_law",
        cache_dir=cache_dir 
    )
    
    index = em.load_index(index_path)
    metadata = em.load_metadata(metadata_path)
    
    if index is None or metadata is None:
        print("\n✗ 인덱스 또는 메타데이터를 찾을 수 없습니다.")
        print("먼저 s4_EmbeddingManager.py를 실행해주세요.")
        return
    
    # SearchEngine 초기화 (bm25_index_path 추가!)
    search_engine = LegalSearchEngine(
        faiss_index=index,
        metadata=metadata,
        embedding_manager=em,
        bm25_index_path=bm25_index_path  # 추가!
    )
    
    print("\n" + "="*80)
    print("📝 테스트 쿼리")
    print("="*80)
    
    query = "건폐율은 어떻게 계산하나요?"
    print(f"\n쿼리: {query}")
    
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