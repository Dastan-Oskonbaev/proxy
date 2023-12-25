import os

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import OAuth2, APIKeyHeader
from openai import AsyncOpenAI
from pydantic import BaseModel
from starlette import status

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

env_api_key = os.environ.get('API_KEY')

app = FastAPI()
api_key_header = APIKeyHeader(name="x-api-key", auto_error=False)
oauth2_scheme = OAuth2()


def get_api_key(api_key_header: str = Security(api_key_header),) -> str:
    if api_key_header in env_api_key:
        return api_key_header
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or missing API Key",
    )


openai_api_key = os.environ.get('OPENAI_API_KEY')

chat_gpt = AsyncOpenAI(
    api_key=openai_api_key
)


class ChatGpt(BaseModel):
    messages: list


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/chatgpt")
async def ask_chatgpt(item: ChatGpt, header_api_key: str = Security(get_api_key)):
    if header_api_key != env_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )
    gpt_response = await chat_gpt.chat.completions.create(
        model="gpt-4",
        messages=item.messages
    )
    return gpt_response
