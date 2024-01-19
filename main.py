import os
import logging
import traceback
from typing import Optional

from telegram import Bot
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Security, Request
from fastapi.security import OAuth2, APIKeyHeader
from openai import AsyncOpenAI
from pydantic import BaseModel
from starlette import status
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

logging.basicConfig(level=logging.INFO)

load_dotenv()

BOT_TOKEN = os.environ.get('BOT_TOKEN')
GROUP_CHAT_ID = os.environ.get('GROUP_CHAT_ID')

bot = Bot(token=BOT_TOKEN)

#
async def send_log_message(error_message, status_code, stack_trace):
    log_message = (
        f"Error occurred: {error_message}\n"
        f"Status Code: {status_code}\n"
        f"Stack Trace: {stack_trace}"
    )
    await bot.send_message(chat_id=GROUP_CHAT_ID, text=log_message)

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)

env_api_key = os.environ.get('API_KEY')

app = FastAPI()
app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
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
    tools: Optional[list[dict]] = None


class LogErrorsMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        try:
            response = await call_next(request)
            return response
        except HTTPException as e:
            error_message = str(e.detail)
            status_code = response.status_code
            stack_trace = traceback.format_exc()
            await send_log_message(error_message, status_code, stack_trace)
            return response
        except Exception as e:
            error_message = str(e)
            status_code = getattr(e, 'status_code', 500)
            stack_trace = traceback.format_exc()
            await send_log_message(error_message, status_code, stack_trace)
            raise


app.add_middleware(LogErrorsMiddleware)


@app.get("/")
async def read_root():
    return {"Hello": "World"}


@app.post("/chatgpt")
async def ask_chatgpt(item: ChatGpt,
                      header_api_key: str = Security(get_api_key),
                      model: str = "gpt-4-1106-preview", temperature: float = 0.7):
    if header_api_key != env_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Forbidden"
        )
    allowed_models = ['gpt-4-1106-preview', 'gpt-4', 'gpt-3.5-turbo-1106']
    if model not in allowed_models:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid model name"
        )
    kwargs = {
              "model": "gpt-4",
              "messages": item.messages,
              "temperature": temperature
              }
    if item.tools:
        kwargs["tools"] = item.tools

    gpt_response = await chat_gpt.chat.completions.create(
        **kwargs
    )
    response_content = gpt_response.choices[0].message.content if gpt_response.choices else None

    response_data = {
        "gpt_response": response_content,
        "model_used": model,
        "temperature_used": temperature
    }

    return response_data
