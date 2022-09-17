import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Callable
from urllib.error import HTTPError

from pytube import Search, YouTube
from tinydb import Query, TinyDB
from tinydb.table import Document
from dataset.database import ConcurrentDatabase

categories = {
    "apparel & accessories": 0.2,
    "packaged goods": 0.2,
    "landmarks": 0.196,
    "furniture & home decor": 0.106,
    "storefronts": 0.102,
    "dishes": 0.068,
    "artwork": 0.054,
    "toys": 0.023,
    "memes": 0.017,
    "illustrations": 0.017,
    "cars": 0.017,
}


class VideoDownloadManager:
    def __init__(self, db: ConcurrentDatabase, download_dir: Path):
        self.db = db
        self.download_dir = download_dir

    def create_download_jobs(self):
        query = Query()

        with ThreadPoolExecutor() as executor:
            for category in categories.keys():
                undownloaded = self.db.search(
                    query.downloaded == False and query.category == category
                )
                executor.submit(self.download_job, category, undownloaded)

            executor.shutdown(True)

    def download_job(self, category: str, videos: List[Document]):
        downloader = _VideoDownloader(self.db, self.download_dir)

        for video in videos:
            downloader.download(category, video["video_id"])


class _VideoDownloader:

    YOUTUBE_LINK = "https://youtu.be/"

    def __init__(self, db: ConcurrentDatabase, download_dir: str):
        self.db = db
        self.download_dir = download_dir

    def download(self, category: str, video_id: str):
        query = Query()
        video = self.db.search(query.video_id == video_id)
        if not video:
            id = self.db.insert(
                {"video_id": video_id, "category": category, "downloaded": False}
            )
            video = self.db.get(doc_id=id)
        else:
            video = video[0]

        if not video["downloaded"]:
            try:
                YouTube(_VideoDownloader.YOUTUBE_LINK + video_id).streams.filter(
                    res="360p"
                ).first().download(
                    output_path=os.path.join(self.download_dir, category)
                )
            except HTTPError:
                return False
            self.db.update({"downloaded": True}, query.video_id == video_id)
            return True
        else:
            return True


class VideoURLFetcher:
    def __init__(self, database: ConcurrentDatabase, total_time_mins=10):
        self.db = database
        self.total_time = total_time_mins * 60

    def _default_video_predicate(video: YouTube) -> bool:
        return video.length <= 300

    def fetch_urls(
        self, video_predicate: Callable[[YouTube], bool] = _default_video_predicate
    ):
        for category, length_ratio in categories.items():
            category_time = self.total_time * length_ratio
            s = Search(category)

            query = Query()

            results = s.results

            while category_time > 0:
                for result in results:
                    try:
                        result.check_availability()
                        if not video_predicate(result):
                            raise ValueError
                    except:
                        continue
                    video_id = result.video_id
                    video = self.db.search(query.video_id == video_id)
                    if not video:
                        self.db.insert(
                            {
                                "video_id": video_id,
                                "category": category,
                                "downloaded": False,
                            }
                        )

                    category_time -= result.length

                    if category_time <= 0:
                        break

                s.get_next_results()

                if (results := s.results) is None:
                    break
