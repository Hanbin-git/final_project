from flask import Flask, jsonify
from flask_cors import CORS
import pandas as pd
from flask import request
import os, pickle, re
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()

client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    organization=os.getenv("OPENAI_ORG_ID"),
    project=os.getenv("OPENAI_PROJECT_ID")
)

app = Flask(__name__)
CORS(app)  # CORS 허용

CSV_PATH = r'D:\code_yoon\wineboard\backend\data\wine_data_merged_final.csv'

MODEL_NAME = 'all-MiniLM-L6-v2'
INDEX_DIR = './backend/data'
model = SentenceTransformer(MODEL_NAME)

# 감성 설명 생성 함수 (OpenAI 사용)

def postprocess(text, max_sentences=3):
    text = re.sub(r'\s+', ' ', text.strip())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(sentences[:max_sentences])

def get_emotional_description(name, desc_en):
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
            max_tokens=300,
        )
        return postprocess(response.choices[0].message.content)
    except Exception as e:
        print(f"감성 설명 생성 실패: {e}")
        return f"감성 설명 생성 실패"


@app.route('/api/recommend', methods=['POST'])
def recommend_wines():
    try:
        data = request.get_json()
        print("요청 데이터:", data)

        wine_type = data.get('type', 'red')
        keyword = data.get('keyword', '')
        saved_descriptions = data.get('saved_descriptions', [])

        index_path = os.path.abspath(os.path.join(INDEX_DIR, f"{wine_type}_faiss.index"))
        meta_path = os.path.abspath(os.path.join(INDEX_DIR, f"{wine_type}_metadata.pkl"))

        print("📁 index_path:", index_path)
        print("📁 meta_path:", meta_path)

        index = faiss.read_index(index_path)
        metadata = pickle.load(open(meta_path, 'rb'))

        keyword_vec = model.encode([keyword]).astype("float32")
        if saved_descriptions:
            saved_vecs = model.encode(saved_descriptions).astype("float32")
            pref_vec = np.mean(saved_vecs, axis=0).reshape(1, -1)
            query_vec = 0.5 * keyword_vec + 0.5 * pref_vec
        else:
            query_vec = keyword_vec

        D, I = index.search(query_vec, k=10)
        seen = set()
        result = []
        for idx in I[0]:
            wine = metadata[idx]
            name = wine.get('wine_name', '이름 없음')
            desc = wine.get('description', '')
            if name not in seen:
                wine['emotion'] = get_emotional_description(name, desc) if desc else "(설명 없음)"
                result.append(wine)
                seen.add(name)
            if len(result) == 5:
                break

        return jsonify(result)
    except Exception as e:
        print("❌ 에러 발생:", str(e))
        return jsonify({'error': str(e)}), 500


@app.route('/api/wines')
def get_wines():
    df = pd.read_csv(CSV_PATH)

    # 필수 컬럼 기준 결측치 제거
    df = df.dropna(subset=['name', 'country', 'vintage', 'price', 'points', 'variety_unique'])

    # 숫자형 컬럼 변환 (잘못된 값은 NaN으로)
    df['vintage'] = pd.to_numeric(df['vintage'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['points'] = pd.to_numeric(df['points'], errors='coerce')

    # 다시 한 번 NaN 제거 (위 변환 과정에서 생긴 NaN 제거)
    df = df.dropna(subset=['vintage', 'price', 'points'])

    # 타입 변환
    df['vintage'] = df['vintage'].astype(int)
    df['points'] = df['points'].astype(int)
    df['price'] = df['price'].astype(float)

    selected = df[['name', 'country', 'vintage', 'price', 'points', 'variety_unique']]
    wines = selected.to_dict(orient='records')
    return jsonify(wines)

# 빈티지별 평균 가격 및 포인트
@app.route('/api/wine-vintage-trend')
def get_wine_vintage_trend():
    df = pd.read_csv(CSV_PATH)

    # 필요한 컬럼만 선택하고 결측치 제거
    df = df.dropna(subset=['vintage', 'price', 'points'])

    # 숫자형 컬럼 변환 (잘못된 값은 NaN으로)
    df['vintage'] = pd.to_numeric(df['vintage'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['points'] = pd.to_numeric(df['points'], errors='coerce')

    # 변환 후 다시 NaN 제거
    df = df.dropna(subset=['vintage', 'price', 'points'])

    # 빈티지, 가격, 포인트를 정수/실수형으로 변환
    df['vintage'] = df['vintage'].astype(int)
    df['points'] = df['points'].astype(int)
    df['price'] = df['price'].astype(float)

    # 빈티지별로 그룹화하여 평균 계산
    # 'as_index=False'를 통해 'vintage' 컬럼을 데이터프레임의 일반 컬럼으로 유지
    aggregated_df = df.groupby('vintage', as_index=False)[['price', 'points']].mean()

    # 컬럼 이름 변경 (프론트엔드에서 사용하기 쉽게)
    aggregated_df = aggregated_df.rename(columns={'price': 'avg_price', 'points': 'avg_points'})

    # 빈티지 순서대로 정렬
    aggregated_df = aggregated_df.sort_values(by='vintage')

    # JSON 응답을 위해 딕셔너리 리스트로 변환
    trend_data = aggregated_df.to_dict(orient='records')

    return jsonify(trend_data)

@app.route('/api/wine-variety-stats')
def get_wine_variety_stats():
    df = pd.read_csv(CSV_PATH)

    # 필수 컬럼 필터 및 결측치 제거
    df = df.dropna(subset=['variety_unique', 'price'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])

    # 품종별 평균 가격 계산
    avg_price_df = df.groupby('variety_unique')['price'].mean().reset_index()
    avg_price_df.columns = ['variety', 'avgPrice']

    # 품종별 개수 계산
    count_df = df['variety_unique'].value_counts().reset_index()
    count_df.columns = ['variety', 'count']

    # 평균 가격과 개수를 병합
    merged_df = pd.merge(avg_price_df, count_df, on='variety')

    # 개수 기준 내림차순 정렬 후 상위 30개 추출
    top_30_df = merged_df.sort_values(by='count', ascending=False).head(30)

    return top_30_df.to_dict(orient='records')


@app.route('/api/wine-value-stats')
def get_wine_value_stats():
    df = pd.read_csv(CSV_PATH)

    # 유효한 데이터 필터링
    df = df.dropna(subset=['points', 'price'])
    df['points'] = pd.to_numeric(df['points'], errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['points', 'price'])

    df = df[(df['price'] < 1000) & (df['points'] > 0)]

    # 산점도용 데이터 일부만 추출 (예: 랜덤 5000개)
    scatter_df = df.sample(n=min(5000, len(df)), random_state=42)

    # 포인트 구간별 히스토그램
    bins = [80, 85, 90, 95, 100]
    labels = [f"{bins[i]}-{bins[i+1]}" for i in range(len(bins)-1)]
    df['range'] = pd.cut(df['points'], bins=bins, labels=labels, right=False)
    histogram = df['range'].value_counts().sort_index().reset_index()
    histogram.columns = ['range', 'count']

    return jsonify({
        'scatter': scatter_df[['points', 'price']].to_dict(orient='records'),
        'histogram': histogram.to_dict(orient='records')
    })

@app.route('/api/wine-vintage-stats')
def get_wine_vintage_stats():
    df = pd.read_csv(CSV_PATH)

    df = df.dropna(subset=['vintage', 'price', 'points'])
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['points'] = pd.to_numeric(df['points'], errors='coerce')
    df = df.dropna(subset=['price', 'points'])

    grouped = df.groupby('vintage').agg({
        'price': 'mean',
        'points': 'mean'
    }).reset_index()

    grouped.columns = ['vintage', 'avgPrice', 'avgPoints']
    grouped = grouped.sort_values('vintage')

    return jsonify(grouped.to_dict(orient='records'))

@app.route('/api/wine-country-stats')
def get_wine_country_stats():
    df = pd.read_csv(CSV_PATH)

    # 필수 컬럼 결측치 제거 및 변환
    df = df.dropna(subset=['country', 'points'])
    df['points'] = pd.to_numeric(df['points'], errors='coerce')
    df = df.dropna(subset=['points'])

    # 국가별 평균 평점과 개수 계산
    grouped = df.groupby('country').agg(
        avgPoints=('points', 'mean'),
        count=('country', 'count')
    ).reset_index()

    # 소수점 반올림
    grouped['avgPoints'] = grouped['avgPoints'].round(2)

    # 개수 기준 내림차순 정렬 후 상위 50개 추출
    top50 = grouped.sort_values('count', ascending=False).head(50)

    return top50.to_dict(orient='records')

@app.route('/api/price-buckets')
def get_price_buckets():
    df = pd.read_csv(CSV_PATH)
    df = df.dropna(subset=['price', 'points'])

    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['points'] = pd.to_numeric(df['points'], errors='coerce')
    df = df.dropna(subset=['price', 'points'])

    # 가격 구간을 10단위로 나누기
    df['price_bucket'] = (df['price'] // 10) * 10
    grouped = df.groupby('price_bucket')['points'].mean().reset_index()
    grouped.columns = ['priceBucket', 'avgPoints']
    grouped['label'] = grouped['priceBucket'].astype(int).astype(str) + '~' + (grouped['priceBucket'] + 9).astype(int).astype(str)
    grouped = grouped[['label', 'avgPoints']].round(2)

    return grouped.to_dict(orient='records')

# print("🔐 API KEY:", os.getenv("OPENAI_API_KEY"))
# print("🏢 ORG ID:", os.getenv("OPENAI_ORG_ID"))
# print("📦 PROJECT ID:", os.getenv("OPENAI_PROJECT_ID"))

if __name__ == "__main__":
    app.run(debug=True, port=5000)