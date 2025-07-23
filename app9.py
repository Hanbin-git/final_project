import streamlit as st
import pandas as pd
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import requests
from bs4 import BeautifulSoup
import google.generativeai as genai
from sklearn.metrics.pairwise import cosine_similarity

# ✅ 환경 설정
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"
genai.configure(api_key="AIzaSyCwcvgtECYL08SvGIB6iqez-tqtboI4ITI")

MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_DIR = 'wine_type_indexes'
model = SentenceTransformer(MODEL_NAME)

# ✅ 번역
def libretranslate_translate(text, source='en', target='ko'):
    try:
        response = requests.post(
            "https://libretranslate.com/translate",
            data={'q': text, 'source': source, 'target': target, 'format': 'text'},
            timeout=10
        )
        return response.json().get('translatedText', None)
    except:
        return None

# ✅ 감성 설명
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
    except:
        return None

# ✅ 이미지 검색
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
    except:
        return None

# ✅ 와인 타입
def get_wine_types():
    types = [f.split('_')[0] for f in os.listdir(INDEX_DIR) if f.endswith('.index')]
    return sorted(set(['rose' if 'ros' in t.lower() else t.lower() for t in types]))

# ✅ 선호 벡터
def get_preference_vector(saved_wines):
    descriptions = [w.get('description', '') for w in saved_wines if w.get('description')]
    if not descriptions:
        return None
    embeddings = model.encode(descriptions)
    return np.mean(embeddings, axis=0).reshape(1, -1)

# ✅ 초기 상태
if "saved_wines" not in st.session_state:
    st.session_state.saved_wines = []
if "show_all_saved" not in st.session_state:
    st.session_state.show_all_saved = False

# ✅ 페이지 구성
st.set_page_config(page_title="와인 추천 시스템", layout="wide")
st.title("🍷 와인 추천 시스템")

# ✅ 사이드바
with st.sidebar:
    st.markdown("## 📌 저장한 와인")
    st.markdown(f"현재 저장: **{len(st.session_state.saved_wines)}개**")

    if st.session_state.saved_wines:
        show_list = st.session_state.saved_wines if st.session_state.show_all_saved else st.session_state.saved_wines[-3:]

        for idx, wine in enumerate(show_list):
            name = wine.get("wine_name", "이름 없음")
            country = wine.get("country", "정보 없음")
            st.write(f"- {name} ({country})")
            if st.button(f"❌ 삭제하기", key=f"delete_{idx}"):
                st.session_state.saved_wines.remove(wine)
                st.rerun()

        toggle_label = "전체 보기 🔽" if not st.session_state.show_all_saved else "최근 3개만 🔼"
        if st.button(toggle_label):
            st.session_state.show_all_saved = not st.session_state.show_all_saved
    else:
        st.info("저장된 와인이 없습니다.")

# ✅ 메인 추천 영역
available_types = get_wine_types()
wine_type = st.selectbox("와인 종류를 선택하세요", available_types)
user_keywords = st.text_input("와인 키워드를 입력하세요 (예: 향긋한, 스파이시, 복숭아)", "")

if st.button("추천 와인 보기") and user_keywords:
    index_path = os.path.join(INDEX_DIR, f"{wine_type}_faiss.index")
    meta_path = os.path.join(INDEX_DIR, f"{wine_type}_metadata.pkl")

    index = faiss.read_index(index_path)
    with open(meta_path, 'rb') as f:
        metadata = pickle.load(f)

    query_vector = model.encode([user_keywords]).astype("float32")
    pref_vector = get_preference_vector(st.session_state.saved_wines)
    if pref_vector is not None:
        query_vector = (0.5 * query_vector + 0.5 * pref_vector).astype("float32")

    D, I = index.search(query_vector, k=30)
    seen, recommended_wines = set(), []

    for idx in I[0]:
        wine = metadata[idx]
        name = wine.get('wine_name', '')
        if name not in seen:
            recommended_wines.append(wine)
            seen.add(name)

    aroma_keywords = ['fruity', 'floral', 'aromatic', 'perfumed', 'spice', 'scent']
    def aroma_score(desc): return sum(kw in desc.lower() for kw in aroma_keywords) if desc else 0
    recommended_wines.sort(key=lambda x: aroma_score(x.get('description', '')), reverse=True)

    final_wines, diverse = [], []
    for wine in recommended_wines:
        country = wine.get('country', '').lower()
        if country not in diverse:
            final_wines.append(wine)
            diverse.append(country)
        if len(final_wines) == 5:
            break

    if not final_wines:
        st.warning("추천 가능한 와인이 없습니다. 키워드를 바꿔보세요.")

    st.markdown("### 🏅 추천 와인 리스트")
    for idx, wine in enumerate(final_wines):
        name = wine.get("wine_name", "이름 없음")
        country = wine.get("country", "정보 없음")
        price = wine.get("price", "정보 없음")
        desc_en = wine.get("description", "설명 없음")
        image_url = get_wine_image_url(name)

        with st.container():
            st.markdown(f"""
            <div style=\"background-color:#fff6f6;padding:15px;border-radius:12px;border:1px solid #ddd;\">
                <h4 style=\"color:#800000;\">🍇 {name.title()}</h4>
                <p><b>국가:</b> {country.title()} | <b>가격:</b> ₩{price}</p>
            """, unsafe_allow_html=True)

            if image_url:
                st.image(image_url, width=160, caption=f"{name.title()} Label")
            else:
                st.markdown("_이미지를 불러올 수 없습니다._")

            st.markdown(f"**🌐 영문 설명:**\n\n{desc_en}")
            chatbot_desc = get_chatbot_style_description(name, desc_en)
            if chatbot_desc:
                st.markdown(f"**🍷 감성 리뷰:**\n\n{chatbot_desc}")
            desc_ko = libretranslate_translate(desc_en)
            if desc_ko:
                st.markdown(f"**📎 참고 설명 (번역):**\n\n{desc_ko}")

            # ✅ 저장 버튼 (이름 기준 중복 방지 + 저장 후 rerun)
            if st.button(f"⭐ 저장하기", key=f"save_{idx}"):
                saved_names = [w['wine_name'] for w in st.session_state.saved_wines]
                if name not in saved_names:
                    st.session_state.saved_wines.append(wine)
                    st.success(f"{name} 저장 완료!")
                    st.rerun()
                else:
                    st.warning("이미 저장된 와인입니다.")

            st.markdown("</div>", unsafe_allow_html=True)
