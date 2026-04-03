from pydantic import BaseModel


class JobConfig(BaseModel):
    job_id: str
    competitor: str
    market: str
    input_path: str


class NormalizedAd(BaseModel):
    ad_id: str
    page_name: str
    ad_copy: str
    video_url: str
