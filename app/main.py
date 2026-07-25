from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.config import Settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.settings = Settings()
    yield


def create_app() -> FastAPI:
    return FastAPI(title="AIC Video Retrieval", lifespan=lifespan)


app = create_app()
