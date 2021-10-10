from dataclasses import dataclass
from typing import List


@dataclass()
class QuerySample:
    query: str
    query_id: int
    positive_doc_id: str


@dataclass()
class ANNSearchRes:
    query_id: str
    search_results: List[str]


@dataclass()
class TrainSampleData:
    similar_doc_ids: List[str]
    positive_doc_id: str
    query: str
    query_id: int


@dataclass()
class IRTrainSample:
    query: str
    doc: str
    label: int


@dataclass()
class IRTrainSampleWithoutDoc:
    query: str
    doc_id: str
    label: int
