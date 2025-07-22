# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
import faiss
import requests
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import matplotlib.pyplot as plt
from collections import Counter
import re

# ✅ 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

# ✅ OpenAI API 설정
client = OpenAI(api_key="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")
MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_DIR = 'wine_type_indexes'
model = SentenceTransformer(MODEL_NAME)

@st.cache_data
def libretranslate_translate(text):
    try:
        res = requests.post(
            "https://libretranslate.com/translate",
            data={'q': text, 'source': 'en', 'target': 'ko', 'format': 'text'}, timeout=10
        )
        return res.json().get('translatedText', None)
    except:
        return None

def postprocess_wine_description(text, max_sentences=3):
    if not text or not isinstance(text, str):
        return "(감성 리뷰를 불러오지 못했습니다.)"
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(r'(?<=[.!?])\s+', text)
    summary = " ".join(sentences[:max_sentences]).strip()
    return summary if summary[-1] in ".!?" else summary + "."

@st.cache_data
def get_chatbot_style_description(name, desc_en):
    try:
        prompt = f"""
        와인 이름: {name}
        영문 설명: {desc_en}
        이 와인을 추천하고 싶은 사람, 어울리는 음식, 감정 등을 감성적으로 설명해줘.
        """
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "너는 감성적인 와인 추천 전문가야."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.8,
            max_tokens=500,
        )
        return postprocess_wine_description(response.choices[0].message.content.strip())
    except Exception as e:
        return f"(설명 실패: {e})"

@st.cache_data
def get_wine_image_url(name):
    try:
        url = f"https://www.google.com/search?tbm=isch&q={name}+wine+label"
        headers = {"User-Agent": "Mozilla/5.0"}
        soup = BeautifulSoup(requests.get(url, headers=headers).text, "html.parser")
        for img in soup.select("img"):
            src = img.get("src")
            if src and src.startswith("http"):
                return src
    except:
        return None

def get_preference_vector(wines):
    descs = [w['description'] for w in wines if w.get('description')]
    if not descs:
        return None
    emb = model.encode(descs)
    return np.mean(emb, axis=0).reshape(1, -1)

if "saved_wines" not in st.session_state:
    st.session_state.saved_wines = []
if "show_all_saved" not in st.session_state:
    st.session_state.show_all_saved = False

st.set_page_config(page_title="와인 추천 시스템", layout="wide")
st.title("\U0001F377 와인 추천 시스템")

with st.sidebar:
    st.markdown("## 📌 저장한 와인")
    if st.session_state.saved_wines:
        wines = st.session_state.saved_wines[-5:] if not st.session_state.show_all_saved else st.session_state.saved_wines
        for i, wine in enumerate(wines):
            st.write(f"- {wine.get('wine_name')} ({wine.get('country')})")
            if st.button(f"❌ 삭제 {i}", key=f"delete_{i}"):
                st.session_state.saved_wines.remove(wine)
                st.rerun()
        st.button("전체 보기 🔽" if not st.session_state.show_all_saved else "최근 5개만 🔼",
                  on_click=lambda: st.session_state.update({"show_all_saved": not st.session_state.show_all_saved}))
        st.markdown("---")
        if st.button("🔁 저장한 와인 기반 추천"):
            vectors = []
            for wine in st.session_state.saved_wines:
                if "description" in wine and wine["description"]:
                    vectors.append(model.encode(wine["description"]))
            if vectors:
                user_vector = np.mean(vectors, axis=0).reshape(1, -1)
                index_path = os.path.join(INDEX_DIR, "red_faiss.index")
                meta_path = os.path.join(INDEX_DIR, "red_metadata.pkl")
                faiss_index = faiss.read_index(index_path)
                wine_data = pd.DataFrame(pickle.load(open(meta_path, 'rb')))
                D, I = faiss_index.search(user_vector, k=5)
                st.session_state.recommended_wines = wine_data.iloc[I[0]]
                st.session_state.show_recommendation = True
            else:
                st.warning("설명이 있는 와인을 저장해주세요.")
    else:
        st.info("저장된 와인이 없습니다.")

wine_types = ['red', 'white', 'sparkling', 'rose', 'fortified']
wine_type = st.selectbox("와인 종류 선택", wine_types)
user_keywords = st.text_input("와인 키워드 입력 (예: 산뜻한, 플로럴, 복숭아)")

if st.button("추천 와인 보기") and user_keywords:
    index_path = os.path.join(INDEX_DIR, f"{wine_type}_faiss.index")
    meta_path = os.path.join(INDEX_DIR, f"{wine_type}_metadata.pkl")
    index = faiss.read_index(index_path)
    metadata = pickle.load(open(meta_path, 'rb'))

    query_vector = model.encode([user_keywords]).astype("float32")
    pref_vector = get_preference_vector(st.session_state.saved_wines)
    if pref_vector is not None:
        query_vector = 0.5 * query_vector + 0.5 * pref_vector

    D, I = index.search(query_vector.astype("float32"), k=30)
    seen, recommended = set(), []
    for idx in I[0]:
        wine = metadata[idx]
        if wine['wine_name'] not in seen:
            recommended.append(wine)
            seen.add(wine['wine_name'])
        if len(recommended) == 5:
            break
    st.session_state.recommended = recommended

if "recommended" in st.session_state:
    st.subheader("📌 추천 와인")
    for idx, wine in enumerate(st.session_state.recommended):
        name, country = wine['wine_name'], wine['country']
        desc_en = wine.get("description", "설명 없음")
        price = wine.get("price", "정보 없음")

        st.markdown(f"**🍷 {name}**  ")
        st.markdown(f"국가: {country} / 가격: ₩{price}")
        image_url = get_wine_image_url(name)
        if image_url:
            st.image(image_url, width=150)
        st.markdown(f"🌐 **영문 설명**: {desc_en}")
        st.markdown(f"📎 **번역**: {libretranslate_translate(desc_en)}")
        st.markdown(f"💌 **감성 설명**: {get_chatbot_style_description(name, desc_en)}")
        if st.button("⭐ 저장", key=f"save_{idx}_{name}"):
            if name not in [w['wine_name'] for w in st.session_state.saved_wines]:
                st.session_state.saved_wines.append(wine)
                st.success(f"{name} 저장 완료!")
                st.rerun()
            else:
                st.warning("이미 저장된 와인입니다.")

if st.session_state.saved_wines:
    st.markdown("---")
    st.subheader("🔍 내가 좋아한 와인 키워드 분석")
    liked_texts = [w['description'].lower() for w in st.session_state.saved_wines if w.get('description')]
    tokens = re.findall(r'[a-zA-Z]{4,}', ' '.join(liked_texts))
    counter = Counter(tokens)
    top_k = counter.most_common(10)
    if top_k:
        labels, values = zip(*top_k)
        fig, ax = plt.subplots()
        ax.barh(labels[::-1], values[::-1])
        ax.set_title("좋아요한 와인 키워드")
        st.pyplot(fig)
