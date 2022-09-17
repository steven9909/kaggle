from multiprocessing import RLock
from typing import Mapping, Union, Callable, Optional
from tinydb import TinyDB
from tinydb.queries import QueryLike


class ConcurrentDatabase:
    def __init__(self, db_dir):
        self.db = TinyDB(db_dir)
        self.lock = RLock()

    def search(self, cond: QueryLike):
        with self.lock:
            return self.db.search(cond)

    def insert(self, document: Mapping):
        with self.lock:
            return self.db.insert(document)

    def update(
        self,
        fields: Union[Mapping, Callable[[Mapping], None]],
        cond: Optional[QueryLike] = None,
    ):
        with self.lock:
            return self.db.update(fields, cond)

    def get(self, cond: Optional[QueryLike] = None, doc_id: Optional[int] = None):
        with self.lock:
            return self.db.get(cond, doc_id)
