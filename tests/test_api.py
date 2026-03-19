from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_predict():

    payload = {
        "sex": 1,
        "pclass": 3,
        "age": 25,
        "fare": 7.25,
        "embarked": 0
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data