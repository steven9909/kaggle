from dataset.utils import VideoURLFetcher, VideoDownloadManager
import hydra
from omegaconf import DictConfig
from dataset.database import ConcurrentDatabase


@hydra.main(".", "config.yaml", None)
def main(config: DictConfig):
    db = ConcurrentDatabase(config.db_loc)
    if config.fetch_urls:
        fetcher = VideoURLFetcher(db, config.dataset_time)
        fetcher.fetch_urls()

    downloader = VideoDownloadManager(db, config.data_dir)
    downloader.create_download_jobs()


if __name__ == "__main__":
    main()
