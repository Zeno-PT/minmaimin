from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from routers import router

app = FastAPI(
    title="MinMaiMin",
    description="Backend system for E.L.C.'s project => MinMaiMin.",
    version="0.1",
    docs_url="/docs/"
)

app.include_router(router)

@app.get("/")
async def root():
    return {"message": "MinMaiMin v0.1"}
