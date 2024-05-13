# 저녁에 올라온 코드 보고 틀린 점 고치기

from flask import Flask, render_template, request
from konlpy.tag import Okt

import joblib 
import re

# flask run --debug
# python ./app.py

app = Flask(__name__)
app.debug = True

okt = Okt()

model_lr = None
tfidf_vector = None

model_nb = None
dtm_vector = None

def load_lr():
    global model_lr, tfidf_vector
    model_lr = joblib.load("model/movie_lr.pkl")
    tfidf_vector = joblib.load("model/movie_lr_dtm.pkl")

def load_nb():
    global model_nb, dtm_vector
    model_nb = joblib.load("model/movie_nb.pkl")
    dtm_vector = joblib.load("model/movie_nb_dtm.pkl")

# 작게작게 하라, 코드를 점진적으로 하나하나씩 하라, 안 그러면 나중에 시간을 너무너무 잡아먹는다
def tw_tokenizer(text): # 전처리 복원해야함, 노트에서 테스트다하고 여기서 테스트: 여기서 테스트하면 어디서 에러가 터졌는지 모른다!
    token_ko=  okt.morphs(text)
    return token_ko

def lt_t(text):
    review = re.sub(r"\d+", " ", text)
    text_vector = tfidf_vector.transform([review]) # callable 오류: 싸이킷런에서 메서드르 안 불렀거나 직렬화를 잘못 한거임!
    return text_vector

def lt_nb(text):
    stopwords = ["은","는","이","가"] # 불용어: 세종 코퍼스에서 다운로드 가능
    review = text.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣]", "")
    morphs = okt.morphs(review, stem=True) # 토큰 분리
    test = " ".join(morph for morph in morphs if not morph in stopwords)
    test_dtm = dtm_vector.transform(test)
    return test_dtm

@app.route("/")
def index():
    menu = { "home": True, "senti": False}
    return render_template("home.html", menu=menu)

@app.route("/senti", methods=['GET', 'POST'])
def senti():
    menu = { "home": False, "senti": True}
    if request.method == "GET": 
        return render_template("senti.html", menu=menu)
    else:
        review = request.form["review"]
        review_text = lt_t(review)
        lr_result = model_lr.predict(review_text)[0]
        review_text2 = lt_nb(review)
        nb_result = model_nb.predict(review_text2)[0]
        lr = "긍정" if lr_result else "부정"
        nb = "긍정" if nb_result else "부정"
        movie = {"review": review, "lr": lr , "nb": nb}
        return render_template("senti_result.html", menu=menu, movie=movie)

if __name__ == "__main__":
    load_lr()
    load_nb()
    app.run()

# 관례를 기억하라
# app.py
# static, templates 이름 변경하지마라

# pkl 파일 만들고
# 역직렬화 확인
# 실행 버튼 누르면 결과 표시
# 학습기와 전처리만 계속, pkl 파일 덮어쓰기
# 짧은 것 만들고, 시간 걸리는 것은 퇴근하면서 걸어두기

# 선형 모델 사용(천만건) : 45분 안짝
# -> 랜덤포레스트: 1시간 이상이면 -> 딥러닝(LSTM)

# LSTM 한번쯤은 만들어보라: 이미지와 텍스트는 딥러닝이 강세
# 숫자 데이터는 머신러닝이 좀더 속도가 빠르고 테스트도 여러번 할 수 있다.