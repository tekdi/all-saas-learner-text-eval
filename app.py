import os
from fastapi import FastAPI
from routes import router


app = FastAPI()

app.include_router(router)

if __name__ == "__main__":

    import uvicorn
    
    num_workers = os.cpu_count() or 1
    uvicorn.run("app:app", host="0.0.0.0", port=5001, debug=False, workers=num_workers)
     


