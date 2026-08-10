import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import JSONResponse

from app.db import init_db
from app.queue_worker import start_workers, stop_workers
from app.routers.images import router as images_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("initializing database")
    init_db()
    logger.info("starting queue workers")
    await start_workers()
    yield
    logger.info("shutting down queue workers")
    await stop_workers()


app = FastAPI(
    title="Vehicle Image Processing Pipeline",
    description="Accepts vehicle images, processes them asynchronously, and reports data-quality issues.",
    version="1.0.0",
    lifespan=lifespan,
)

app.include_router(images_router)


@app.get("/health", tags=["health"])
def health():
    return {"status": "ok"}


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("unhandled exception on %s %s", request.method, request.url)
    return JSONResponse(status_code=500, content={"detail": "internal server error"})
