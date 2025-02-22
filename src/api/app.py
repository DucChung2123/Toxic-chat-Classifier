from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List
from src.inference.predict import ToxicClassifierAPI

app = FastAPI(
    title="API Phát Hiện Nội Dung Độc Hại",
    description="API sử dụng mô hình BERT để phát hiện nội dung độc hại trong văn bản.",
    version="1.0.0",
    contact={
        "name": "Duc Chung",
        "email": "Chungcgcg@gmail.com",
    },
)

classifier = ToxicClassifierAPI()

# 🚀 **Định nghĩa Input Schema cho dự đoán một câu**
class TextInput(BaseModel):
    text: str = Field(..., example="Bạn ngu quá!", description="Văn bản đầu vào cần kiểm tra độ độc hại.")

# 🚀 **Định nghĩa Input Schema cho dự đoán nhiều câu**
class BatchTextInput(BaseModel):
    texts: List[str] = Field(
        ..., 
        example=["Mày bị điên à?", "Bạn thật tốt bụng!"], 
        description="Danh sách các câu cần kiểm tra."
    )

# 🚀 **Định nghĩa Output Schema**
class PredictionOutput(BaseModel):
    text: str = Field(..., description="Câu đầu vào đã được kiểm tra.")
    label: str = Field(..., example="toxic", description="Kết quả phân loại: 'toxic' (độc hại) hoặc 'normal' (bình thường).")
    confidence: float = Field(..., example=0.95, description="Độ tin cậy của mô hình với dự đoán này.")

@app.get("/", summary="Kiểm tra trạng thái API")
def home():
    """
    ✅ **Kiểm tra xem API có đang chạy không.**  
    Trả về một thông báo đơn giản xác nhận rằng API đang hoạt động.
    """
    return {"message": "API phát hiện nội dung độc hại đang chạy!"}

@app.post("/predict/", response_model=PredictionOutput, summary="Dự đoán độ độc hại của một câu")
def predict_text(input_data: TextInput):
    """
    🚀 **Dự đoán mức độ độc hại của một câu.**  
    - **Đầu vào:** Một chuỗi văn bản cần kiểm tra.
    - **Đầu ra:** Nhãn (`toxic` hoặc `normal`) cùng với độ tin cậy của mô hình.

    **Ví dụ yêu cầu:**
    ```json
    {
        "text": "Mày thật ngu ngốc!"
    }
    ```
    **Ví dụ phản hồi:**
    ```json
    {
        "text": "Mày thật ngu ngốc!",
        "label": "toxic",
        "confidence": 0.97
    }
    ```
    """
    result = classifier.predict(input_data.text)
    return result

@app.post("/batch_predict/", response_model=List[PredictionOutput], summary="Dự đoán độ độc hại của nhiều câu")
def batch_predict_text(input_data: BatchTextInput):
    """
    🚀 **Dự đoán mức độ độc hại của nhiều câu cùng lúc.**  
    - **Đầu vào:** Một danh sách các câu văn bản.
    - **Đầu ra:** Một danh sách kết quả phân loại.

    **Ví dụ yêu cầu:**
    ```json
    {
        "texts": [
            "Tao ghét mày!",
            "Bạn thật tốt bụng!"
        ]
    }
    ```
    **Ví dụ phản hồi:**
    ```json
    [
        {
            "text": "Tao ghét mày!",
            "label": "toxic",
            "confidence": 0.98
        },
        {
            "text": "Bạn thật tốt bụng!",
            "label": "normal",
            "confidence": 0.99
        }
    ]
    ```
    """
    results = classifier.batch_predict(texts=input_data.texts)
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
