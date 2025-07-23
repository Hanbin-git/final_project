import streamlit as st
import pandas as pd
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import wikipedia
import google.generativeai as genai

# ✅ 환경 설정
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
genai.configure(api_key="AIzaSyCwcvgtECYL08SvGIB6iqez-tqtboI4ITI")

MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_DIR = 'wine_type_indexes'
model = SentenceTransformer(MODEL_NAME)

# ✅ 번역 함수
def libretranslate_translate(text, source='en', target='ko'):
    try:
        response = requests.post(
            "https://libretranslate.com/translate",
            data={'q': text, 'source': source, 'target': target, 'format': 'text'},
            timeout=10
        )
        result = response.json()
        return result.get('translatedText', None)
    except Exception as e:
        print(f"[LibreTranslate Error]: {e}")
        return None

# ✅ 감성 리뷰 생성
def get_chatbot_style_description(wine_name, description_en):
    try:
        prompt = (
            f"와인 이름: {wine_name}\n"
            f"영문 설명: {description_en}\n\n"
            f"이 와인을 추천하고 싶은 사람, 어울리는 음식, 감정 등을 따뜻한 톤으로 한국어로 설명해주세요."
        )
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"[Gemini Error]: {e}")
        return None

# ✅ 와인 라벨 이미지 검색 (Google 이미지)
def get_wine_image_url(wine_name):
    try:
        query = wine_name + " wine label"
        url = f"https://www.google.com/search?tbm=isch&q={query}"
        headers = {"User-Agent": "Mozilla/5.0"}
        res = requests.get(url, headers=headers)
        soup = BeautifulSoup(res.text, "html.parser")
        img_tags = soup.select("img")
        for img in img_tags:
            src = img.get("src")
            if src and src.startswith("http"):
                return src
        return None
    except Exception as e:
        print(f"[Image Fetch Error]: {e}")
        return None

# ✅ 와인 타입 가져오기
def get_wine_types():
    types = [f.split('_')[0] for f in os.listdir(INDEX_DIR) if f.endswith('.index')]
    clean_types = ['rose' if 'ros' in t.lower() else t.lower() for t in types]
    return sorted(set(clean_types))

available_types = get_wine_types()

# ✅ UI 시작
st.set_page_config(page_title="와인 추천 시스템", layout="wide")
st.title("🍷 와인 추천 시스템")

wine_type = st.selectbox("와인 종류를 선택하세요", available_types)
user_keywords = st.text_input("와인 키워드를 입력하세요 (예: 향긋한, 스파이시, 복숭아)", "")

if st.button("추천 와인 보기") and user_keywords:
    index_path = os.path.join(INDEX_DIR, f"{wine_type}_faiss.index")
    meta_path = os.path.join(INDEX_DIR, f"{wine_type}_metadata.pkl")

    index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)

    query_vector = model.encode([user_keywords]).astype("float32")
    D, I = index.search(query_vector, k=30)

    seen = set()
    recommended_wines = []
    for idx in I[0]:
        wine = metadata[idx]
        name = wine.get('wine_name', '')
        if name not in seen:
            recommended_wines.append(wine)
            seen.add(name)

    aroma_keywords = ['fruity', 'floral', 'aromatic', 'perfumed', 'spice', 'scent']
    def aroma_score(desc):
        if not desc:
            return 0
        desc_lower = desc.lower()
        return sum(kw in desc_lower for kw in aroma_keywords)

    recommended_wines.sort(key=lambda x: aroma_score(x.get('description', '')), reverse=True)

    diverse = []
    final_wines = []
    for wine in recommended_wines:
        country = wine.get('country', '').lower()
        if country not in diverse:
            final_wines.append(wine)
            diverse.append(country)
        if len(final_wines) == 5:
            break

    # ✅ 출력
    st.markdown("### 🏅 추천 와인 리스트")
    for wine in final_wines:
        name = wine.get("wine_name", "이름 없음")
        country = wine.get("country", "정보 없음")
        price = wine.get("price", "정보 없음")
        desc_en = wine.get("description", "설명 없음")

        image_url = get_wine_image_url(name)

        with st.container():
            st.markdown(f"""
            <div style="background-color:#fff6f6;padding:15px;border-radius:12px;border:1px solid #ddd;">
                <h4 style="color:#800000;">🍇 {name.title()}</h4>
                <p><b>국가:</b> {country.title()} | <b>가격:</b> ₩{price}</p>
            """, unsafe_allow_html=True)

            # ✅ 이미지 출력
            if image_url:
                st.image(image_url, width=160, caption=f"{name.title()} Label")
            else:
                st.markdown("_이미지를 불러올 수 없습니다._")

            # ✅ 영문 설명
            st.markdown(f"**🌐 영문 설명:**\n\n{desc_en}")

            # ✅ 감성 리뷰
            chatbot_desc = get_chatbot_style_description(name, desc_en)
            if chatbot_desc:
                st.markdown(f"**🍷 감성 리뷰:**\n\n{chatbot_desc}")
            else:
                st.markdown("❌ 감성 리뷰 생성 실패")

            # ✅ 번역
            desc_ko = libretranslate_translate(desc_en)
            if desc_ko:
                st.markdown(f"**📎 참고 설명 (번역):**\n\n{desc_ko}")
            else:
                st.markdown("❌ 설명 번역 실패")

            st.markdown("</div>", unsafe_allow_html=True)
