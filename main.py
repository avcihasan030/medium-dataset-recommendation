# import os
#
# from fastapi import FastAPI
# from pydantic import BaseModel
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import pandas as pd
# from fastapi.staticfiles import StaticFiles
#
# app = FastAPI()
#
# # Statik dosya servisi için images klasörünü bağla
# app.mount("/images", StaticFiles(directory="images"), name="images")
#
#
# # Blog model sınıfı
# class Blog(BaseModel):
#     id: int
#     url: str
#     title: str
#     subtitle: str
#     image: str
#     claps: int
#     responses: int
#     reading_time: int
#     publication: str
#     date: str
#
#
# # CSV dosyasından verileri oku
# raw_data = pd.read_csv("data/preprocessed_data.csv")
#
# # TF-IDF vektörleme
# raw_data['content'] = raw_data['title'] + ' ' + raw_data['subtitle']
# tfidf_vectorizer = TfidfVectorizer(stop_words='english')
# tfidf_matrix = tfidf_vectorizer.fit_transform(raw_data['content'])
# cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
#
#
# class ArticleRecommendationRequest(BaseModel):
#     article_id: int
#
#
# class ArticleRecommendationResponse(BaseModel):
#     recommended_articles: list[Blog]
#
#
# @app.post("/get_article_recommendations", response_model=ArticleRecommendationResponse)
# async def get_article_recommendations(request: ArticleRecommendationRequest):
#     article_id = request.article_id
#     idx = raw_data[raw_data['id'] == article_id].index[0]
#     sim_scores = list(enumerate(cosine_sim[idx]))
#     sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
#     sim_scores = sim_scores[1:6]
#     article_indices = [i[0] for i in sim_scores]
#     recommended_articles = raw_data.iloc[article_indices].to_dict(orient='records')
#
#     # # Makale resim yollarını güncelle
#     # for article in recommended_articles:
#     #     article['image'] = f"http://127.0.0.1:8000/images/{article['image']}"
#     # Makale resim yollarını güncelle ve eşleşen resim dosyalarını bul
#     for article in recommended_articles:
#         article_title = article['title'].replace(' ',
#                                                  '').lower()  # Makale başlığını al ve boşlukları ve küçük harfe dönüştür
#         for file_name in os.listdir("images"):  # images klasöründe dolaş
#             if article_title in file_name.lower():  # Makale başlığı dosya adında varsa
#                 article['image'] = f"http://127.0.0.1:8000/images/{file_name}"  # Resim yolunu güncelle
#                 break  # Eşleşme bulunduğunda döngüden çık
#     return {"recommended_articles": recommended_articles}

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Statik dosya servisi için images klasörünü bağla
app.mount("/images", StaticFiles(directory="images"), name="images")


# Blog model sınıfı
class Blog(BaseModel):
    id: int
    url: str
    title: str
    subtitle: str
    image: str
    claps: int
    responses: int
    reading_time: int
    publication: str
    date: str


# CSV dosyasından verileri oku
try:
    raw_data = pd.read_csv("data/preprocessed_data.csv")
    raw_data['content'] = raw_data['title'] + ' ' + raw_data['subtitle']
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(raw_data['content'])
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
except Exception as e:
    print(f"Error loading data: {e}")


class ArticleRecommendationRequest(BaseModel):
    article_ids: list[int]


class ArticleRecommendationResponse(BaseModel):
    recommended_articles: list[Blog]


@app.post("/get_article_recommendations", response_model=ArticleRecommendationResponse)
async def get_article_recommendations(request: ArticleRecommendationRequest):
    try:
        article_ids = request.article_ids
        indices = [raw_data[raw_data['id'] == article_id].index[0] for article_id in article_ids]

        # Belirtilen makalelerin TF-IDF vektörlerinin ortalamasını al ve ndarray'e dönüştür
        combined_vector = np.asarray(tfidf_matrix[indices].mean(axis=0))
        sim_scores = cosine_similarity(combined_vector, tfidf_matrix)[0]

        # Similarlık skorlarına göre sıralayıp en benzer 5 makaleyi seç
        sim_scores = list(enumerate(sim_scores))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [score for score in sim_scores if score[0] not in indices][:5]
        article_indices = [i[0] for i in sim_scores]
        recommended_articles = raw_data.iloc[article_indices].to_dict(orient='records')

        # Makale resim yollarını güncelle ve eşleşen resim dosyalarını bul
        for article in recommended_articles:
            article_title = article['title'].replace(' ', '').lower()
            for file_name in os.listdir("images"):
                if article_title in file_name.lower():
                    article['image'] = f"http://127.0.0.1:8000/images/{file_name}"
                    break

        return {"recommended_articles": recommended_articles}
    except Exception as e:
        print(f"Error in recommendation function: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")
