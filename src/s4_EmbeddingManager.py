"""
s4_EmbeddingManager.py
임베딩 생성과 FAISS 인덱스 관리
"""

"""
✅ FAISS 인덱스는 "벡터 창고"
┌─────────────────────────────────────────────────────────┐
│ FAISS 인덱스 = 1500개 벡터를 그냥 저장해둔 창고        │
│                                                         │
│ 📦 벡터 0:    [0.023, -0.056, 0.089, ...]              │
│ 📦 벡터 1:    [0.045, 0.012, -0.034, ...]              │
│ 📦 벡터 2:    [-0.078, 0.091, 0.056, ...]              │
│ ...                                                     │
│ 📦 벡터 1499: [0.034, -0.067, 0.045, ...]              │
│                                                         │
│ → 미리 정렬되어 있지 않음!                             │
│ → 검색할 때 거리 계산해서 정렬함                       │
└─────────────────────────────────────────────────────────┘
"""

import os
import json
import pickle
import hashlib
from typing import List, Dict, Optional, Tuple
import numpy as np
from openai import OpenAI
import faiss
from dotenv import load_dotenv


class EmbeddingManager:
    """임베딩 생성 및 FAISS 인덱스 관리 클래스"""
    
    def __init__(self, 
                 openai_api_key: str,
                 institution: str = "construction_law",
                 model: str = "text-embedding-3-large",
                 cache_dir: str = None,
                 dimension: int = 3072):
        """
        Args:
            openai_api_key: OpenAI API 키
            institution: 기관/프로젝트 이름 (캐시 파일명용)
            model: 임베딩 모델명
            cache_dir: 캐시 디렉토리 경로 (절대 경로 권장, None이면 "data/cache")
            dimension: 임베딩 차원
        """
        self.client = OpenAI(api_key=openai_api_key)
        self.model = model
        self.institution = institution
        self.dimension = dimension
        
        # 캐시 경로 자동 생성
        if cache_dir is None:
            cache_dir = "data/cache"
        
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, f"embeddings_{institution}.pkl")
        
        self.cache_path = cache_path
        self.embedding_cache = self.load_embedding_cache()
        
        print(f"🎯 EmbeddingManager 초기화")
        print(f"  - 모델: {model}")
        print(f"  - 차원: {dimension}")
        print(f"  - 캐시: {len(self.embedding_cache)}개 임베딩")
        print(f"  - 캐시 경로: {cache_path}")
    
    def load_embedding_cache(self) -> Dict[str, np.ndarray]:
        """임베딩 캐시 로드"""
        if os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    cache = pickle.load(f)
                print(f"✓ 캐시 로드: {len(cache)}개 임베딩")
                return cache
            except Exception as e:
                print(f"⚠ 캐시 로드 실패 ({e}), 새로 시작")
                return {}
        return {}
    
    def save_embedding_cache(self):
        """임베딩 캐시 저장"""
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.embedding_cache, f)
            print(f"✓ 캐시 저장: {len(self.embedding_cache)}개 임베딩")
        except Exception as e:
            print(f"⚠ 캐시 저장 실패: {e}")
    
    def get_text_hash(self, text: str) -> str:
        """텍스트의 MD5 해시 계산 (캐시 키)"""
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def embed_text(self, text: str) -> np.ndarray:
        """텍스트를 임베딩 벡터로 변환"""
        # 캐시 확인
        text_hash = self.get_text_hash(text)
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # OpenAI API 호출
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.model
            )
            
            embedding = np.array(response.data[0].embedding, dtype='float32')
            
            # 캐시에 저장
            self.embedding_cache[text_hash] = embedding
            
            return embedding
            
        except Exception as e:
            print(f"⚠ 임베딩 생성 실패: {e}")
            return np.zeros(self.dimension, dtype='float32')
    
    def embed_chunks(self, chunks: List[Dict], batch_size: int = 100) -> Tuple[List[np.ndarray], List[str]]:
        """
        여러 청크를 배치로 임베딩
        
        Returns:
            (임베딩 벡터 리스트, 청크 ID 리스트)
        """
        embeddings = []
        chunk_ids = []
        
        print(f"\n🧮 임베딩 생성 시작...")
        print(f"  - 총 청크 수: {len(chunks)}")
        print(f"  - 배치 크기: {batch_size}")
        
        cache_hits = 0
        cache_misses = 0
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            batch_texts = [chunk['content'] for chunk in batch]
            batch_chunk_ids = [chunk['chunk_id'] for chunk in batch]
            
            print(f"\n  배치 {i//batch_size + 1}/{(len(chunks)-1)//batch_size + 1}")
            
            # 배치 내에서 캐시 확인
            batch_embeddings = []
            texts_to_embed = []
            text_indices = []
            
            for j, text in enumerate(batch_texts):
                text_hash = self.get_text_hash(text)
                if text_hash in self.embedding_cache:
                    batch_embeddings.append(self.embedding_cache[text_hash])
                    cache_hits += 1
                else:
                    batch_embeddings.append(None)
                    texts_to_embed.append(text)
                    text_indices.append(j)
                    cache_misses += 1
            
            # 캐시에 없는 것만 API 호출
            if texts_to_embed:
                try:
                    response = self.client.embeddings.create(
                        input=texts_to_embed,
                        model=self.model
                    )
                    
                    for idx, data in enumerate(response.data):
                        embedding = np.array(data.embedding, dtype='float32')
                        original_idx = text_indices[idx]
                        batch_embeddings[original_idx] = embedding
                        
                        # 캐시에 저장
                        text_hash = self.get_text_hash(texts_to_embed[idx])
                        self.embedding_cache[text_hash] = embedding
                    
                    print(f"    ✓ {len(texts_to_embed)}개 새로 생성")
                    
                except Exception as e:
                    print(f"    ✗ 배치 실패: {e}")
                    for idx in text_indices:
                        if batch_embeddings[idx] is None:
                            batch_embeddings[idx] = np.zeros(self.dimension, dtype='float32')
            
            embeddings.extend(batch_embeddings)
            chunk_ids.extend(batch_chunk_ids)
            
            progress = min((i + batch_size) / len(chunks) * 100, 100)
            print(f"    진행률: {progress:.1f}%")
        
        print(f"\n✅ 임베딩 생성 완료!")
        print(f"  - 캐시 히트: {cache_hits}개")
        print(f"  - 새로 생성: {cache_misses}개")
        
        if cache_misses > 0:
            self.save_embedding_cache()
        
        return embeddings, chunk_ids
    
    def create_faiss_index(self, embeddings: List[np.ndarray]) -> faiss.Index:
        """FAISS 인덱스 생성"""
        print(f"\n🔧 FAISS 인덱스 생성 중...")
        
        # Flat 인덱스 (정확한 최근접 이웃 검색)
        index = faiss.IndexFlatL2(self.dimension)
        
        # numpy 배열로 변환
        embeddings_array = np.array(embeddings).astype('float32')
        
        # 인덱스에 추가
        index.add(embeddings_array)
        
        print(f"✓ FAISS 인덱스 생성 완료")
        print(f"  - 타입: Flat (L2)")
        print(f"  - 벡터 수: {index.ntotal}")
        
        return index
    
    def save_index(self, index: faiss.Index, index_path: str):
        """FAISS 인덱스 저장"""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        
        try:
            faiss.write_index(index, index_path)
            print(f"✓ 인덱스 저장: {index_path}")
        except Exception as e:
            print(f"✗ 인덱스 저장 실패: {e}")
    
    def load_index(self, index_path: str) -> Optional[faiss.Index]:
        """FAISS 인덱스 로드"""
        if not os.path.exists(index_path):
            print(f"⚠ 인덱스 없음: {index_path}")
            return None
        
        try:
            index = faiss.read_index(index_path)
            print(f"✓ 인덱스 로드: {index_path}")
            print(f"  - 벡터 수: {index.ntotal}")
            return index
        except Exception as e:
            print(f"✗ 인덱스 로드 실패: {e}")
            return None
    
    def save_metadata(self, chunks: List[Dict], chunk_ids: List[str], metadata_path: str):
        """메타데이터 저장"""
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        
        chunk_dict = {chunk['chunk_id']: chunk for chunk in chunks}
        
        metadata = []
        for i, chunk_id in enumerate(chunk_ids):
            chunk = chunk_dict.get(chunk_id, {})
            metadata.append({
                "index": i,
                "chunk_id": chunk_id,
                "content": chunk.get("content", ""),
                "metadata": chunk.get("metadata", {})
            })
        
        try:
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"✓ 메타데이터 저장: {metadata_path}")
            print(f"  - 항목 수: {len(metadata)}")
        except Exception as e:
            print(f"✗ 메타데이터 저장 실패: {e}")
    
    def load_metadata(self, metadata_path: str) -> Optional[List[Dict]]:
        """메타데이터 로드"""
        if not os.path.exists(metadata_path):
            print(f"⚠ 메타데이터 없음: {metadata_path}")
            return None
        
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            print(f"✓ 메타데이터 로드: {metadata_path}")
            print(f"  - 항목 수: {len(metadata)}")
            return metadata
        except Exception as e:
            print(f"✗ 메타데이터 로드 실패: {e}")
            return None
    
    def build_index_from_chunks(self, chunks_path: str, 
                                output_dir: str = None) -> Tuple[faiss.Index, List[Dict]]:
        """
        청크 파일에서 인덱스 구축 (전체 파이프라인)
        
        Returns:
            (FAISS 인덱스, 메타데이터 리스트)
        """
        if output_dir is None:
            output_dir = f"data/vector_store/{self.institution}"
        
        print("\n" + "="*80)
        print("🚀 FAISS 인덱스 구축 시작")
        print("="*80)
        
        # 1. 청크 로드
        print("\n[1] 청크 로드")
        with open(chunks_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)
        print(f"✓ {len(chunks)}개 청크")
        
        # 2. 임베딩 생성
        print("\n[2] 임베딩 생성")
        embeddings, chunk_ids = self.embed_chunks(chunks, batch_size=100)
        
        # 3. FAISS 인덱스 생성
        print("\n[3] FAISS 인덱스 생성")
        index = self.create_faiss_index(embeddings)
        
        # 4. 저장
        print("\n[4] 저장")
        index_path = os.path.join(output_dir, "faiss_index.bin")
        self.save_index(index, index_path)
        
        metadata_path = os.path.join(output_dir, "metadata.json")
        self.save_metadata(chunks, chunk_ids, metadata_path)
        
        metadata = self.load_metadata(metadata_path)
        
        print("\n" + "="*80)
        print("✅ 인덱스 구축 완료!")
        print("="*80)
        print(f"📁 출력: {output_dir}")
        print("="*80 + "\n")
        
        return index, metadata


def main():
    """
    EmbeddingManager 실행 메인 함수
    
    입력: data/chunks/construction_law_chunks.json
    출력: data/vector_store/construction_law/
           - faiss_index.bin
           - metadata.json
    캐시: data/cache/embeddings_construction_law.pkl
    """
    print("="*80)
    print("🧮 임베딩 및 FAISS 인덱스 구축")
    print("="*80)
    
    # 환경 변수 로드
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("\n✗ 오류: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다.")
        print("  .env 파일에 다음과 같이 설정해주세요:")
        print("  OPENAI_API_KEY=sk-your-api-key-here")
        return
    
    # 현재 파일 위치 기준으로 프로젝트 루트 찾기
    current_dir = os.path.dirname(os.path.abspath(__file__))  # src/
    project_root = os.path.dirname(current_dir)  # CNTCHATBOT_PJT2/
    
    # 경로 설정 (모두 절대 경로)
    CHUNKS_PATH = os.path.join(project_root, "data", "chunks", "construction_law_chunks.json")
    OUTPUT_DIR = os.path.join(project_root, "data", "vector_store", "construction_law")
    CACHE_DIR = os.path.join(project_root, "data", "cache")
 
    print(f"\n프로젝트 루트: {project_root}")
    print(f"입력 파일: {CHUNKS_PATH}")
    print(f"출력 디렉토리: {OUTPUT_DIR}")
    print(f"캐시 디렉토리: {CACHE_DIR}")
    
    # 입력 파일 존재 확인
    if not os.path.exists(CHUNKS_PATH):
        print(f"\n✗ 입력 파일이 없습니다: {CHUNKS_PATH}")
        print("먼저 s3_LegalChunking.py를 실행해주세요.")
        return
    
    # 이미 인덱스가 있으면 건너뛰기 옵션
    index_path = os.path.join(OUTPUT_DIR, "faiss_index.bin")
    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    
    if os.path.exists(index_path) and os.path.exists(metadata_path):
        response = input(f"\n⚠ 인덱스가 이미 존재합니다: {OUTPUT_DIR}\n덮어쓰시겠습니까? (y/n): ")
        if response.lower() != 'y':
            print("취소되었습니다.")
            return
    
    # EmbeddingManager 실행
    try:
        embedding_manager = EmbeddingManager(
            openai_api_key=openai_api_key,
            institution="construction_law",
            model="text-embedding-3-large",
            cache_dir=CACHE_DIR
        )
        
        index, metadata = embedding_manager.build_index_from_chunks(
            chunks_path=CHUNKS_PATH,
            output_dir=OUTPUT_DIR
        )
        
        print("="*80)
        print("✅ FAISS 인덱스 구축 완료!")
        print("="*80)
        print(f"\n생성된 파일:")
        print(f"  - {os.path.join(OUTPUT_DIR, 'faiss_index.bin')}")
        print(f"  - {os.path.join(OUTPUT_DIR, 'metadata.json')}")
        print(f"  - {os.path.join(CACHE_DIR, 'embeddings_construction_law.pkl')}")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n✗ 오류 발생: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()