from threading import Thread
from threading import RLock
from tinydb import TinyDB, Query
from tinydb.table import Document
from pytube import YouTube, Search
import os
from urllib.error import HTTPError
from typing import List

keywords = {
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
    def __init__(self, path, download_dir):
        self.database = TinyDB(path)
        self.download_dir = download_dir
        self.db_lock = RLock()

    def fetch_urls(self):
        fetcher = VideoUrlFetcher(self.database)
        fetcher.fetch_urls()

    def create_download_jobs(self):
        threads = []
        video_query = Query()

        for keyword, _ in keywords.items():
            with self.db_lock:
                undownloaded = self.database.search(
                    video_query.downloaded == False and video_query.category == keyword
                )
                threads.append(
                    Thread(target=self.download_job, args=(keyword, undownloaded))
                )

        for thread in threads:
            thread.start()

        for thread in threads:
            thread.join()

    def download_job(self, category: str, videos: List[Document]):
        downloader = VideoDownloader(self.database, self.download_dir)
        for video in videos:
            downloader.download(category, video["video_id"])


class VideoDownloader:

    YOUTUBE_LINK = "https://youtu.be/"

    def __init__(self, database: TinyDB, download_dir: str):
        self.database = database
        self.download_dir = download_dir

    def download(self, category: str, video_id: str):
        video_query = Query()
        video = self.database.search(video_query.video_id == video_id)
        if not video:
            id = self.database.insert(
                {"video_id": video_id, "category": category, "downloaded": False}
            )
            video = self.database.get(doc_id=id)
        else:
            video = video[0]

        if not video["downloaded"]:
            try:
                YouTube(VideoDownloader.YOUTUBE_LINK + video_id).streams.filter(
                    res="360p", only_video=True
                ).first().download(
                    output_path=os.path.join(self.download_dir, category)
                )
            except HTTPError:
                return False
            self.database.update({"downloaded": True}, video_query.video_id == video_id)
            return True
        else:
            return True


class VideoUrlFetcher:
    def __init__(self, database: TinyDB, total_time_mins=10):
        self.database = database
        self.total_time = total_time_mins * 60

    def fetch_urls(self):
        for category, length_ratio in keywords.items():
            category_time = self.total_time * length_ratio
            s = Search(category)

            video_query = Query()

            results = s.results

            while category_time > 0:
                for result in results:
                    try:
                        result.check_availability()
                    except:
                        continue
                    video_id = result.video_id
                    video = self.database.search(video_query.video_id == video_id)
                    if not video:
                        self.database.insert(
                            {
                                "video_id": video_id,
                                "category": category,
                                "downloaded": False,
                            }
                        )

                    category_time -= result.length
                    if category_time <= 0:
                        break

                results = s.get_next_results()
                if results is None:
                    break


if __name__ == "__main__":
    manager = VideoDownloadManager("db.json", os.path.join(os.getcwd(), "videos"))
    manager.create_download_jobs()
