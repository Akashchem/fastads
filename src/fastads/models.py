from pydantic import BaseModel


class JobConfig(BaseModel):
    job_id: str
    competitor: str
    market: str
    input_path: str


class NormalizedAd(BaseModel):
    ad_id: str
    title: str | None = None
    body: str | None = None
