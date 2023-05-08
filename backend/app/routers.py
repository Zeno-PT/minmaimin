from fastapi import APIRouter, Depends, File, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import os
import pytesseract

from model import model_obj


class Payload(BaseModel):
    msg: str


router = APIRouter(
    prefix="/api",
    tags=["minmaimin"],
    responses={404: {"description": "Not found"}},
)


@router.post("/predict/")
def predict(payload: Payload):
    result = model_obj.predict(str(payload.msg))    
    return JSONResponse(content={'result': result})

@router.post('/ocr/')
def image_to_text(image: UploadFile = File(...)):
    try:
        os.mkdir("images")
    except Exception as e:
        return JSONResponse(content={'result': str(e)}) 
    file_name = os.getcwd()+"/images/"+image.filename.replace(" ", "-")
    with open(file_name,'wb+') as f:
        f.write(image.file.read())
        f.close()
    result = pytesseract.image_to_string(file_name, lang='tha').replace('\n', '')
    return JSONResponse(content={'result': result})