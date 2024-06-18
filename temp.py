import requests

# API endpoint
url = "http://127.0.0.1:8000/get_article_recommendations"

# Test verisi: birden fazla makale ID'si içeren liste
data = {
    "article_ids": [22,33]  # Örnek makale ID'leri
}

# POST isteği gönder
response = requests.post(url, json=data)

# Yanıtı kontrol et ve yazdır
if response.status_code == 200:
    recommendations = response.json()
    print("Recommended Articles:")
    for article in recommendations['recommended_articles']:
        print(f"Title: {article['title']}")
        print(f"URL: {article['url']}")
        print(f"Image: {article['image']}")
        print(f"Claps: {article['claps']}")
        print(f"Responses: {article['responses']}")
        print(f"Reading Time: {article['reading_time']}")
        print(f"Publication: {article['publication']}")
        print(f"Date: {article['date']}")
        print("\n")
else:
    print(f"Request failed with status code {response.status_code}")
    print(response.text)
