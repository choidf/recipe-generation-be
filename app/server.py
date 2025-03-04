from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .predict import Predictor
from fastapi.staticfiles import StaticFiles

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Adjust this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add static files middleware
app.mount("/app/results", StaticFiles(directory="app/results"), name="static")

@app.get('/')
def read_root():
    return {'message': 'Cooking recipe generation model API'}

@app.post('/predict2')
async def predict2(data: dict):
    image_path = data['image_path']
    # Perform prediction
    result = Predictor().predict(image_path)
    return result

@app.post("/predict")
async def predict(file: UploadFile):
    try:
        if not file:
            raise HTTPException(status_code=422, detail="No file uploaded")
        
        file_path = f"app/images/{file.filename}"
        with open(file_path, "wb") as f:
            f.write(file.file.read())
            # Perform prediction
        result = Predictor().predict(file_path)
        return result
    except Exception as e:
        return {"message": e.args}
