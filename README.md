# 프로젝트 소개

본 프로젝트는 Fastcampus 강의 중 제공되는 프로젝트 입니다.

- 강의: [RAG 비법노트](https://fastcampus.co.kr/data_online_teddy)
- 프로젝트 링크: https://link.teddynote.com/PERPLEX


## 설치

다음의 명령어로 가상환경을 활성화 합니다.

```bash
poetry shell
```

패키지를 설치합니다.

```bash
poetry install
```

## 실행

```bash
poetry run streamlit run main.py
```

## 스트림릿에 배포

1. 다음의 [링크](https://share.streamlit.io/)로 접속합니다.
2. 계정을 생성 합니다.
3. 우측 상단의 "Create app" 버튼을 클릭합니다.
4. "Deploy a public app from GitHub" 버튼을 클릭합니다.
5. 본인의 "Repository" 를 입력합니다. 링크를 입력합니다.
6. Main file path 는 "main.py" 입니다.
7. "Advanced settings" 를 클릭합니다.
    - `Python version` 은 3.11 을 선택합니다.
    - `Secrets` 에 API KEY 를 입력합니다.
    - "Save" 버튼을 클릭합니다.
8. "Deploy" 버튼을 클릭합니다.


