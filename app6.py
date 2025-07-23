import streamlit as st
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import wikipedia
import google.generativeai as genai

# ✅ API 키 설정
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
genai.configure(api_key="AIzaSyCwcvgtECYL08SvGIB6iqez-tqtboI4ITI")  # 사용자 제공 키

# ✅ 모델 및 디렉토리 설정
MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_DIR = 'wine_type_indexes'
model = SentenceTransformer(MODEL_NAME)

# ✅ 와인 종류 불러오기
def get_wine_types():
    types = [f.split('_')[0] for f in os.listdir(INDEX_DIR) if f.endswith('.index')]
    clean_types = ['rose' if 'ros' in t.lower() else t.lower() for t in types]
    return sorted(set(clean_types))

available_types = get_wine_types()

# ✅ LibreTranslate 기반 번역 함수
def libretranslate_translate(text, source='en', target='ko'):
    try:
        response = requests.post(
            "https://libretranslate.com/translate",
            data={
                'q': text,
                'source': source,
                'target': target,
                'format': 'text'
            },
            timeout=10
        )
        result = response.json()
        if 'translatedText' in result:
            return result['translatedText']
        else:
            print(f"[LibreTranslate Error]: {result.get('error', 'Unknown error')}")
            return None
    except Exception as e:
        print(f"[LibreTranslate Exception]: {e}")
        return None


# ✅ 한국어 설명 생성 (백업용)
def get_korean_description(wine_name, fallback_en_desc=None):
    try:
        # 나무위키 시도
        url = f"https://namu.wiki/w/{wine_name.replace(' ', '%20')}"
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                if len(text) > 30:
                    return text
    except:
        pass

    try:
        # 한국어 위키백과 시도
        wikipedia.set_lang("ko")
        return wikipedia.summary(wine_name, sentences=2)
    except:
        pass

    # 영어 위키백과 후 번역 or fallback desc 번역
    try:
        wikipedia.set_lang("en")
        summary_en = wikipedia.summary(wine_name, sentences=2)
        return libretranslate_translate(summary_en, source='en', target='ko')
    except:
        if fallback_en_desc:
            return libretranslate_translate(fallback_en_desc, source='en', target='ko')
    return None


# ✅ Gemini 감성 리뷰 생성
def get_chatbot_style_description(wine_name, description_en):
    try:
        prompt = (
            f"와인 이름: {wine_name}\n"
            f"영문 설명: {description_en}\n\n"
            f"이 와인은 어떤 사람에게 추천하고 싶은지, 어떤 음식과 어울리는지, "
            f"마셨을 때 어떤 감정이 드는지까지 감성적으로 표현해주세요. "
            f"소믈리에가 손님에게 추천하듯 따뜻하고 부드러운 어조로, 자연스럽고 매력적인 한국어로 작성해주세요."
        )
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Error]: {e}")
        return None

# ✅ Streamlit UI 구성
st.title("🍷 와인 추천 시스템")

wine_type = st.selectbox("와인 종류", available_types)
user_query = st.text_input("원하는 와인 키워드 입력 (예: 향긋한 복숭아, 스파이시, 꽃향)", "")

if st.button("와인 추천 받기") and user_query:
    # ✅ 인덱스 및 메타데이터 불러오기
    index_path = os.path.join(INDEX_DIR, f"{wine_type}_faiss.index")
    meta_path = os.path.join(INDEX_DIR, f"{wine_type}_metadata.pkl")
    index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)

    # ✅ 검색 수행
    query_vector = model.encode([user_query]).astype("float32")
    D, I = index.search(query_vector, k=30)

    seen = set()
    results = []
    for i in I[0]:
        wine = metadata[i]
        name = wine.get('wine_name', '')
        if name not in seen:
            results.append(wine)
            seen.add(name)

    # ✅ 향 키워드 기반 정렬
    aroma_keywords = ['fruity', 'floral', 'aromatic', 'perfumed', 'spice', 'scent', 'bouquet', 'nose']
    def aroma_score(desc):
        if not desc:
            return 0
        desc_lower = desc.lower()
        return sum(kw in desc_lower for kw in aroma_keywords)

    results.sort(key=lambda x: aroma_score(x.get('description', '')), reverse=True)

    # ✅ 국가 다양성 반영
    diverse_countries = []
    final_results = []
    for wine in results:
        country = wine.get('country', '').lower()
        if len(final_results) >= 5:
            break
        if country not in diverse_countries:
            final_results.append(wine)
            diverse_countries.append(country)

    # ✅ 출력
    st.markdown("### 🏅 추천 와인 리스트")
    for result in final_results:
        name = result.get("wine_name", "이름 없음")
        country = result.get("country", "정보 없음")
        price = result.get("price", "정보 없음")
        desc_en = result.get("description", "설명 없음")

        st.write(f"**{name}**")
        st.write(f"국가: {country}, 가격: ₩{price}")
        st.write(f"_영문 설명:_ {desc_en}")

        # 🍷 감성 리뷰
        st.markdown("### 🍷 추천 와인 리뷰 (감성)")
        chatbot_desc = get_chatbot_style_description(name, desc_en)
        if chatbot_desc:
            st.write(chatbot_desc)
        else:
            st.write("❌ 감성 리뷰 생성에 실패했습니다.")

        # 📎 참고 설명
        st.markdown("📎 참고 설명 ")
        translated_desc = libretranslate_translate(desc_en, source='en', target='ko')
        if translated_desc:
            st.write(translated_desc)
        else:
            st.write("❌ 번역 실패. 영어 설명만 제공됩니다.")


        st.markdown("---")
