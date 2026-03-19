"""
1) Необходимо скачать видео на 10 минут с YouTube (можно использовать savefrom.net)

2) Загрузить его в облако (Яндекс.Диск Google Drive и т.п)

3) Скачивать у себя это видео в коде

4) Это видео разделить на кусочки по 20 секунд

5) Кусочки через batch распознать и склеить их воедино используя LCEL (LangChain Language Expression)

6) Скинуть в комментарии PR или ссылку на репозиторий в GitHub

"""
import base64
from dataclasses import dataclass
import os
import logging
from pathlib import Path

from moviepy.video.io.VideoFileClip import VideoFileClip
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openrouter import ChatOpenRouter
from langchain.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel


logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class VideoChunk:
    index: int
    path: str
    start: int
    end: int
    duration: int


class GoogleDriveDownloader:
    def __init__(self, output_path: str):
        self._output_path = output_path

        self._api_key = os.environ["GOOGLE_DRIVE_API_KEY"]
        self._service = build("drive", "v3", developerKey=self._api_key)

    def download_video(self, video_id: str) -> str:
        if os.path.exists(self._output_path ):
            logger.info("[x] Already downloaded")
            return self._output_path

        request = self._service.files().get_media(fileId=video_id)

        with open(self._output_path , "wb") as f:
            downloader = MediaIoBaseDownload(f, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        self._service.close()

        return self._output_path


class VideoSplitter:
    def __init__(self, output_dir: str, chunk_duration: int = 20):
        self._output_dir = output_dir
        self._chunk_duration = chunk_duration

    def split_video(self, video_path: str) -> list[VideoChunk]:
        Path(self._output_dir).mkdir(parents=True, exist_ok=True)
        clip = VideoFileClip(video_path)
        duration = clip.duration
        chunks = []
        
        logger.info(
            f"[x] Нарезка видео ({duration:.1f} сек) на чанки по " \
            f"{self._chunk_duration} сек...")
        
        for i, start in enumerate(range(0, int(duration), self._chunk_duration)):
            end = min(start + self._chunk_duration, duration)
            output_file = Path(self._output_dir) / f"chunk_{i:03d}.mp4"
            
            subclip = clip.subclipped(start, end)
            subclip.write_videofile(
                str(output_file),
                codec='libx264',
                audio_codec='aac',
                logger=None
            )
            chunks.append(
                VideoChunk(
                    index=i,
                    path=str(output_file),
                    start=start,
                    end=end,
                    duration=end - start
                )
            )
            logger.info(f"[x] Создан чанк {i}")
        
        clip.close()
        logger.info(f"[x] Создано {len(chunks)} чанков")

        return chunks


class VideoAnalyzer:
    def __init__(self, video_llm: BaseChatModel):
        self.video_llm = video_llm

    def describe_chunk(self, video_chunk: VideoChunk) -> str:
        with open(video_chunk.path, "rb") as f:
            video_data = base64.standard_b64encode(f.read()).decode("utf-8")
        
        message = HumanMessage(
            content=[
                {"type": "text", "text": "Describe content of this video."},
                {
                    "type": "video",
                    "base64": video_data,
                    "mime_type": "video/mp4"
                }
            ]
        )
        res = self.video_llm.invoke([message])
        return res.content[0]["text"]


class LlmPipeline:
    def __init__(
        self,
        video_downloader: GoogleDriveDownloader,
        video_splitter: VideoSplitter,
        video_analyzer: VideoAnalyzer,
        video_llm: BaseChatModel,
        text_llm: BaseChatModel

    ):
        self._video_downloader = video_downloader
        self._video_splitter = video_splitter
        self._video_analyzer = video_analyzer

        self._video_llm = video_llm
        self._text_llm = text_llm

        self._pipeline = (
            RunnableLambda(self._download)
            | RunnableLambda(self._split_video_by_chunks)
            | RunnableLambda(self._describe_chunks)
            | RunnableLambda(self._summarize_chunks_descriptions)
        )

    
    def _download(self, video_id: str):
        return self._video_downloader.download_video(video_id=video_id)
    
    def _split_video_by_chunks(self, video_path: str) -> list[VideoChunk]:
        return self._video_splitter.split_video(video_path=video_path)
    
    def _describe_chunks(self, chunks: list[VideoChunk]) -> list[str]:
        runnable = RunnableLambda(self._video_analyzer.describe_chunk)
        return runnable.batch(chunks)

    def _summarize_chunks_descriptions(self, chunks_descriptions: list[str]):
        messages = [
            SystemMessage(content="You're a text summarization assistant."),
            HumanMessage(
                content="The texts are a description of the video chunks." \
                        f"Give a general summary of it: {chunks_descriptions}")
        ]

        return self._text_llm.invoke(messages)
    
    def run(self, video_id: str):
        return self._pipeline.invoke(video_id)
    

if __name__ == "__main__":
    VIDEO_ID = "1caZ-vi51k1QuiQsmstRh4nQOKsuhSTeQ"
    VIDEO_OUTPUT_PATH = "./python_video.mp4"
    CHUNKS_OUTPUT_DIR = "./video_chunks"

    video_llm = ChatGoogleGenerativeAI(
        model="gemini-3-pro-preview",
        base_url='https://api.proxyapi.ru/google',
        api_key=os.environ["OPENAI_API_KEY"]
    )

    free_text_llm = ChatOpenRouter(
        model="stepfun/step-3.5-flash:free",
        openrouter_api_key=os.environ["OPENROUTER_API_KEY"],
        base_url="https://openrouter.ai/api/v1",
        temperature=0.2,
        max_retries=2,
    )

    video_downloader = GoogleDriveDownloader(output_path=VIDEO_OUTPUT_PATH)
    video_splitter = VideoSplitter(output_dir=CHUNKS_OUTPUT_DIR)
    video_analyzer = VideoAnalyzer(video_llm)


    llm_pipeline = LlmPipeline(
        video_downloader=video_downloader,
        video_splitter=video_splitter,
        video_analyzer=video_analyzer,
        video_llm=video_llm,
        text_llm=free_text_llm
    )

    result = llm_pipeline.run(VIDEO_ID)

    with open("video_summary.txt", "w") as f:
        f.write(result.content)
