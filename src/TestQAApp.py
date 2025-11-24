"""
app.py
건설법령 챗봇 Streamlit
- 진행 상황 실시간 표시
- 원본 JSON 데이터 보관
"""

import streamlit as st
import os
from dotenv import load_dotenv
from s4_EmbeddingManager import EmbeddingManager
from s5_LegalSearchEngine import LegalSearchEngine
from s62_GPTLegalSearchSystem import EnhancedLegalQASystem
from s61_QueryClassifier import QueryClassifier
import json
from io import BytesIO
from datetime import datetime

# PDF 생성용
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.units import mm

load_dotenv()

st.set_page_config(
    page_title="건설법령 챗봇",
    page_icon="🏗️",
    layout="wide"
)

# 스타일
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .query-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: bold;
        background-color: #4ecdc4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


def create_pdf(title: str, content: str) -> BytesIO:
    """텍스트를 PDF로 변환"""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    font_paths = [
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/System/Library/Fonts/AppleGothic.ttf",
        "C:/Windows/Fonts/malgun.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    ]
    
    font_registered = False
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                pdfmetrics.registerFont(TTFont('Korean', font_path))
                font_registered = True
                break
            except:
                continue
    
    font_name = 'Korean' if font_registered else 'Helvetica'
    
    c.setFont(font_name, 16)
    c.drawString(30*mm, height - 30*mm, title)
    
    c.setFont(font_name, 10)
    c.drawString(30*mm, height - 40*mm, f"생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    c.line(30*mm, height - 45*mm, width - 30*mm, height - 45*mm)
    
    c.setFont(font_name, 11)
    y_position = height - 55*mm
    line_height = 6*mm
    
    lines = content.split('\n')
    for line in lines:
        if y_position < 30*mm:
            c.showPage()
            c.setFont(font_name, 11)
            y_position = height - 30*mm
        
        while len(line) > 70:
            c.drawString(30*mm, y_position, line[:70])
            line = line[70:]
            y_position -= line_height
            if y_position < 30*mm:
                c.showPage()
                c.setFont(font_name, 11)
                y_position = height - 30*mm
        
        c.drawString(30*mm, y_position, line)
        y_position -= line_height
    
    c.save()
    buffer.seek(0)
    return buffer


def format_document_content(answer: dict) -> str:
    """문서_생성 응답을 편집 가능한 텍스트로 변환"""
    doc_type = answer.get("문서_유형", "문서")
    title = answer.get("제목", "제목 없음")
    
    content_lines = [
        f"{'='*60}",
        f"{title}",
        f"{'='*60}",
        "",
        f"문서 유형: {doc_type}",
        f"생성일: {datetime.now().strftime('%Y-%m-%d')}",
        "",
    ]
    
    if answer.get("근거_법령"):
        content_lines.append("[ 근거 법령 ]")
        for law in answer["근거_법령"]:
            content_lines.append(f"  • {law}")
        content_lines.append("")
    
    if answer.get("내용"):
        content_lines.append("[ 점검 항목 ]")
        content_lines.append("")
        for item in answer["내용"]:
            번호 = item.get("번호", "-")
            항목 = item.get("항목", "")
            기준 = item.get("기준", "")
            법적_근거 = item.get("법적_근거", "")
            
            content_lines.append(f"{번호}. {항목}")
            content_lines.append(f"   기준: {기준}")
            if 법적_근거:
                content_lines.append(f"   법적 근거: {법적_근거}")
            content_lines.append(f"   점검 결과: [ ] 적합  [ ] 부적합  [ ] 해당없음")
            content_lines.append("")
    
    if answer.get("사용_방법"):
        content_lines.append("[ 사용 방법 ]")
        content_lines.append(answer["사용_방법"])
        content_lines.append("")
    
    content_lines.extend([
        "",
        "─" * 60,
        "",
        "점검일: ______년 ____월 ____일",
        "",
        "점검자: _________________ (서명)",
        "",
        "관리감독자: _________________ (서명)",
        "",
        "─" * 60,
    ])
    
    return "\n".join(content_lines)


@st.cache_resource
def load_system():
    """시스템 로드"""
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    
    if not OPENAI_API_KEY:
        st.error("⚠️ OPENAI_API_KEY 필요")
        st.stop()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    vector_store_dir = os.path.join(project_root, "data", "vector_store", "construction_law")
    cache_dir = os.path.join(project_root, "data", "cache")
    
    with st.spinner("🔧 시스템 로딩..."):
        em = EmbeddingManager(OPENAI_API_KEY, "construction_law", cache_dir=cache_dir)
        
        index = em.load_index(os.path.join(vector_store_dir, "faiss_index.bin"))
        metadata = em.load_metadata(os.path.join(vector_store_dir, "metadata.json"))
        
        if not index or not metadata:
            st.error("⚠️ 인덱스 파일 없음")
            st.stop()
        
        engine = LegalSearchEngine(index, metadata, em)
        classifier = QueryClassifier(OPENAI_API_KEY)
        qa_system = EnhancedLegalQASystem(engine, OPENAI_API_KEY)
    
    return engine, classifier, qa_system


# 메인
st.markdown('<p class="main-title">🏗️ 건설법령 AI 챗봇</p>', unsafe_allow_html=True)

engine, classifier, qa_system = load_system()

# 사이드바
with st.sidebar:
    st.header("📖 사용 가이드")
    st.markdown("""
    **질문 유형:**
    - 🔴 법조문: "제36조 내용"
    - 🟢 정보: "비계 안전 기준"
    - 🔵 컨설팅: "3m 비계 괜찮아?"
    - 🟡 절차: "용도변경 절차"
    - 🟠 문서: "체크리스트 만들어"
    - 🟣 비교: "A법과 B법 차이"
    """)
    
    st.markdown("---")
    
    st.header("📝 문서 생성 가이드")
    st.markdown("""
    **요청 예시:**
    - "비계 점검 체크리스트 만들어줘"
    - "안전관리 계획서 초안 작성해줘"
    - "굴착작업 안전점검표 양식"
    
    **생성 후:**
    1. 📝 편집기에서 수정
    2. 💾 TXT / 📄 PDF 다운로드
    """)
    
    st.markdown("---")
    
    # 상세 정보 토글
    show_details = st.checkbox("🔍 상세 정보 표시", value=False)
    
    st.markdown("---")
    st.caption("💡 문서는 법령 기반이지만, 전문가 검토를 권장합니다.")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

if "current_document" not in st.session_state:
    st.session_state.current_document = None

if "document_title" not in st.session_state:
    st.session_state.document_title = ""

# 채팅 기록 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # 상세 정보 표시 (assistant 메시지 + 토글 ON + full_answer 있을 때)
        if show_details and msg["role"] == "assistant" and "full_answer" in msg:
            with st.expander("📊 상세 정보"):
                meta = msg["full_answer"].get("_meta", {})
                
                # 분류 정보
                st.write("**🏷️ 질문 유형:**", meta.get("query_type", "N/A"))
                st.write("**📈 확신도:**", f"{meta.get('classification', {}).get('confidence', 0):.0%}")
                
                # 참조 문서
                st.write("**📚 참조 문서:**")
                for i, s in enumerate(meta.get("sources", [])[:3], 1):
                    st.write(f"  [{i}] {s['doc_name']} (p.{s['page']})")

# 문서 편집기 표시
if st.session_state.current_document:
    st.markdown("---")
    st.markdown("### 📝 문서 편집기")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        edited_content = st.text_area(
            "문서 내용 (자유롭게 편집하세요)",
            value=st.session_state.current_document,
            height=400,
            key="document_editor"
        )
    
    with col2:
        st.markdown("**📥 다운로드**")
        
        st.download_button(
            label="💾 TXT 저장",
            data=edited_content.encode('utf-8'),
            file_name=f"{st.session_state.document_title}.txt",
            mime="text/plain",
            use_container_width=True
        )
        
        try:
            pdf_buffer = create_pdf(st.session_state.document_title, edited_content)
            st.download_button(
                label="📄 PDF 저장",
                data=pdf_buffer,
                file_name=f"{st.session_state.document_title}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
        except Exception as e:
            st.warning(f"PDF 생성 실패: {e}")
        
        st.markdown("---")
        
        if st.button("❌ 편집기 닫기", use_container_width=True):
            st.session_state.current_document = None
            st.session_state.document_title = ""
            st.rerun()

# 채팅 입력
if prompt := st.chat_input("질문을 입력하세요"):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        
        # ===== 진행 상황 표시 =====
        with st.status("🤔 답변 생성 중...", expanded=True) as status:
            
            # 1단계: 질문 분류
            st.write("🏷️ 질문 유형 분석 중...")
            classification = classifier.classify(prompt)
            query_type = classification["query_type"]
            confidence = classification["confidence"]
            st.write(f"   ✅ **{query_type}** (확신도: {confidence:.0%})")
            
            # 2단계: 검색 전략
            st.write("🔍 검색 전략 결정 중...")
            strategy = classifier.get_search_strategy(query_type)
            st.write(f"   ✅ {strategy['search_method']} 검색 (top_k={strategy['top_k']})")
            
            # 3단계: 문서 검색
            st.write("📚 관련 문서 검색 중...")
            search_results = engine.hybrid_search(prompt, top_k=strategy['top_k'])
            st.write(f"   ✅ {len(search_results)}개 문서 발견")
            
            # 4단계: 답변 생성
            st.write("✍️ GPT 답변 생성 중...")
            answer = qa_system.generate_answer(prompt, verbose=False, format_for_user=True)
            st.write("   ✅ 답변 생성 완료!")
            
            status.update(label="✅ 답변 완료!", state="complete", expanded=False)
        
        # ===== 답변 표시 =====
        meta = answer.get("_meta", {})
        query_type = meta.get("query_type", "일반_정보_검색")
        confidence = meta.get("classification", {}).get("confidence", 0)
        
        # 유형 배지
        st.markdown(f"""
        <span class="query-badge">{query_type}</span>
        <span style="color: gray;"> (확신도: {confidence:.0%})</span>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # ===== 문서_생성만 특별 처리 =====
        if query_type == "문서_생성":
            제목 = answer.get("제목", "생성된 문서")
            
            st.success(f"📄 **{제목}** 문서가 생성되었습니다!")
            
            document_content = format_document_content(answer)
            
            st.session_state.current_document = document_content
            st.session_state.document_title = 제목
            
            st.markdown("**📋 문서 미리보기:**")
            st.code(document_content[:500] + "..." if len(document_content) > 500 else document_content)
            
            st.info("👆 위 '문서 편집기'에서 내용을 수정하고 다운로드할 수 있습니다.")
            
            display_text = f"📄 {제목} 문서가 생성되었습니다. 위 편집기에서 수정 후 다운로드하세요."
            st.session_state.messages.append({
                "role": "assistant", 
                "content": display_text,
                "full_answer": answer  # 원본 보관
            })
            
            st.rerun()
        
        # ===== 나머지: user_friendly_answer 표시 =====
        else:
            display_text = answer.get("user_friendly_answer", "답변을 생성했습니다.")
            st.markdown(display_text)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": display_text,
                "full_answer": answer  # 원본 보관
            })