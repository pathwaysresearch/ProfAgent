# -*- coding: utf-8 -*-

import os
import re
import json
import time
import hashlib
from pathlib import Path
from typing import Optional, List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from youtube_transcript_api import YouTubeTranscriptApi

load_dotenv()

YOUTUBE_API_KEY = "[ENCRYPTED KEY]"
if not YOUTUBE_API_KEY:
    raise RuntimeError(
        "Missing YOUTUBE_API_KEY. Put it in your .env file or export it in your shell."
    )


class BaseScraper:
    """Base class for all scrapers. Handles rate limiting and caching."""

    def __init__(self, cache_dir: str = "./cache"):
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)

    def _cache_key(self, key: str) -> str:
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> str:
        return os.path.join(self.cache_dir, self._cache_key(key) + ".json")

    def _get_cached(self, key: str) -> Optional[dict]:
        path = self._cache_path(key)
        if os.path.exists(path):
            age_hours = (time.time() - os.path.getmtime(path)) / 3600
            if age_hours < 168:  # 7 day cache
                with open(path, "r", encoding="utf-8") as f:
                    return json.load(f)
        return None

    def _set_cached(self, key: str, data: dict):
        path = self._cache_path(key)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)

    def _rate_limit(self, seconds: float = 1.0):
        time.sleep(seconds)


class YouTubeChannelScraper(BaseScraper):
    """
    Uses YouTube Data API + youtube-transcript-api.

    Flow:
      1) channels.list -> uploads playlist
      2) playlistItems.list -> video IDs
      3) videos.list -> metadata
      4) youtube-transcript-api -> transcript
      5) save one JSON per video in videos/
    """

    def __init__(
        self,
        cache_dir: str = "./cache",
        videos_dir: str = "./videos",
        transcript_languages: Optional[List[str]] = None,
        rate_limit_seconds: float = 0.5,
        transcript_workers: int = 4,
    ):
        super().__init__(cache_dir=cache_dir)
        self.videos_dir = Path(videos_dir)
        self.videos_dir.mkdir(parents=True, exist_ok=True)

        self.transcript_languages = transcript_languages or ["en"]
        self.rate_limit_seconds = rate_limit_seconds
        self.transcript_workers = transcript_workers

        self.youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        self.transcript_api = YouTubeTranscriptApi()

    def _extract_channel_filter(
        self,
        channel_id: Optional[str] = None,
        handle: Optional[str] = None,
        username: Optional[str] = None,
    ) -> dict:
        filters = [channel_id is not None, handle is not None, username is not None]
        if sum(bool(x) for x in filters) != 1:
            raise ValueError("Specify exactly one of: channel_id, handle, username")

        if channel_id:
            return {"id": channel_id}

        if handle:
            handle = handle if handle.startswith("@") else f"@{handle}"
            return {"forHandle": handle}

        return {"forUsername": username}

    def get_channel_resource(
        self,
        channel_id: Optional[str] = None,
        handle: Optional[str] = None,
        username: Optional[str] = None,
    ) -> dict:
        cache_key = f"channel:{channel_id or handle or username}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        params = {
            "part": "snippet,contentDetails",
            "maxResults": 1,
            **self._extract_channel_filter(
                channel_id=channel_id,
                handle=handle,
                username=username,
            ),
        }

        try:
            response = self.youtube.channels().list(**params).execute()
        except HttpError as e:
            raise RuntimeError(f"YouTube channels.list failed: {e}") from e

        # Fallback if handle/username lookup returns nothing
        if not response.get("items") and (handle or username):
            query = handle or username
            search_response = self.youtube.search().list(
                part="snippet",
                q=query,
                type="channel",
                maxResults=5,
            ).execute()

            channel_ids = []
            for item in search_response.get("items", []):
                ch_id = item.get("id", {}).get("channelId")
                if ch_id:
                    channel_ids.append(ch_id)

            if channel_ids:
                response = self.youtube.channels().list(
                    part="snippet,contentDetails",
                    id=",".join(channel_ids[:1]),
                    maxResults=1,
                ).execute()

        self._set_cached(cache_key, response)
        return response

    def get_uploads_playlist_id(self, channel_resource: dict) -> str:
        items = channel_resource.get("items", [])
        if not items:
            raise ValueError("No channel found for the given filter")

        try:
            return items[0]["contentDetails"]["relatedPlaylists"]["uploads"]
        except KeyError as e:
            raise ValueError("Could not find uploads playlist for this channel") from e

    def iter_playlist_video_ids(
        self,
        playlist_id: str,
        max_videos: Optional[int] = None,
        max_pages: int = 50,
    ) -> List[str]:
        video_ids: List[str] = []
        page_token = None
        page_count = 0

        while True:
            page_count += 1
            if page_count > max_pages:
                break

            try:
                response = self.youtube.playlistItems().list(
                    part="snippet,contentDetails",
                    playlistId=playlist_id,
                    maxResults=50,
                    pageToken=page_token,
                ).execute()
            except HttpError as e:
                raise RuntimeError(f"YouTube playlistItems.list failed: {e}") from e

            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                resource_id = snippet.get("resourceId", {})
                video_id = resource_id.get("videoId")
                if video_id:
                    video_ids.append(video_id)
                    if max_videos and len(video_ids) >= max_videos:
                        return video_ids

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return video_ids

    def _chunked(self, items: List[str], size: int = 50):
        for i in range(0, len(items), size):
            yield items[i:i + size]

    def get_video_metadata(self, video_ids: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Returns:
          {
            video_id: {
              title, description, date, channel, video_id, video_url
            }
          }
        """
        meta: Dict[str, Dict[str, Any]] = {}

        for batch in self._chunked(video_ids, 50):
            try:
                response = self.youtube.videos().list(
                    part="snippet",
                    id=",".join(batch),
                ).execute()
            except HttpError as e:
                raise RuntimeError(f"YouTube videos.list failed: {e}") from e

            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                vid = item.get("id")
                if not vid:
                    continue

                published_at = snippet.get("publishedAt", "")
                date_only = published_at[:10] if published_at else None

                meta[vid] = {
                    "video_id": vid,
                    "video_url": f"https://www.youtube.com/watch?v={vid}",
                    "title": snippet.get("title", "") or "",
                    "description": snippet.get("description", "") or "",
                    "date": date_only,
                    "channel": snippet.get("channelTitle", "") or "",
                }

        return meta

    def _merge_transcript(self, transcript_snippets: List[Dict[str, Any]]) -> str:
        parts = []
        for snip in transcript_snippets:
            text = str(snip.get("text", "")).strip()
            if text:
                parts.append(text)
        return " ".join(parts).strip()

    def get_english_transcript(self, video_id: str) -> Optional[str]:
        """
        Returns a single merged transcript string in English.
        Tries English first; if unavailable, tries translation to English.
        """
        try:
            fetched = self.transcript_api.fetch(video_id, languages=self.transcript_languages)
            raw = fetched.to_raw_data()
            return self._merge_transcript(raw)
        except Exception:
            pass

        try:
            transcript_list = self.transcript_api.list(video_id)

            # Try explicit English transcript
            try:
                transcript = transcript_list.find_transcript(["en"])
                fetched = transcript.fetch()
                return self._merge_transcript(fetched.to_raw_data())
            except Exception:
                pass

            # Try any translatable transcript and translate to English
            for t in transcript_list:
                try:
                    if getattr(t, "is_translatable", False):
                        fetched = t.translate("en").fetch()
                        return self._merge_transcript(fetched.to_raw_data())
                except Exception:
                    continue
        except Exception:
            return None

        return None

    def _save_video_json(self, payload: Dict[str, Any]) -> str:
        path = self.videos_dir / f"{payload['video_id']}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return str(path)

    def scrape_channel(
        self,
        channel_id: Optional[str] = None,
        handle: Optional[str] = None,
        username: Optional[str] = None,
        max_videos: Optional[int] = None,
    ) -> List[dict]:
        """
        Scrape a channel's uploads and save one JSON per video in ./videos.
        """
        channel_key = channel_id or handle or username or "unknown"
        cached = self._get_cached(f"channel_scrape:{channel_key}:{max_videos}")
        if cached:
            return cached

        self._rate_limit(self.rate_limit_seconds)

        channel_resource = self.get_channel_resource(
            channel_id=channel_id,
            handle=handle,
            username=username,
        )
        uploads_playlist_id = self.get_uploads_playlist_id(channel_resource)

        video_ids = self.iter_playlist_video_ids(
            uploads_playlist_id,
            max_videos=max_videos,
        )
        metadata_map = self.get_video_metadata(video_ids)

        results = []

        # Sequential by default: simpler, safer, less likely to trip rate limits.
        for vid in video_ids:
            meta = metadata_map.get(
                vid,
                {
                    "video_id": vid,
                    "video_url": f"https://www.youtube.com/watch?v={vid}",
                    "title": "",
                    "description": "",
                    "date": None,
                    "channel": "",
                },
            )

            transcript = self.get_english_transcript(vid) or ""

            payload = {
                "video_id": meta["video_id"],
                "title": meta["title"],
                "description": meta["description"],
                "date": meta["date"],
                "channel": meta["channel"],
                "transcript": transcript,
                "video_url": meta["video_url"],
            }

            saved_to = self._save_video_json(payload)
            payload["saved_to"] = saved_to
            results.append(payload)

        self._set_cached(f"channel_scrape:{channel_key}:{max_videos}", results)
        return results
    
    def scrape_video(self, video_id: str) -> dict:
        """
        Scrape a single video using video_id and save JSON.
        """

        cache_key = f"video:{video_id}"
        cached = self._get_cached(cache_key)
        if cached:
            return cached

        self._rate_limit(self.rate_limit_seconds)

        # 1. Get metadata
        metadata_map = self.get_video_metadata([video_id])
        meta = metadata_map.get(
            video_id,
            {
                "video_id": video_id,
                "video_url": f"https://www.youtube.com/watch?v={video_id}",
                "title": "",
                "description": "",
                "date": None,
                "channel": "",
            },
        )

        # 2. Get transcript
        transcript = self.get_english_transcript(video_id) or ""

        # 3. Build payload
        payload = {
            "video_id": meta["video_id"],
            "title": meta["title"],
            "description": meta["description"],
            "date": meta["date"],
            "channel": meta["channel"],
            "transcript": transcript,
            "video_url": meta["video_url"],
        }

        # 4. Save
        saved_to = self._save_video_json(payload)
        payload["saved_to"] = saved_to

        # # 5. Cache
        # self._set_cached(cache_key, payload)

        return payload


if __name__ == "__main__":

    BASE_DIR = Path(__file__).parent
    scraper = YouTubeChannelScraper(
        videos_dir=BASE_DIR / "videos",
        cache_dir=BASE_DIR / "cache",
        transcript_languages=["en"],
        rate_limit_seconds=0.5,
    )
    result = scraper.scrape_video("msj1qZmJ4PY")

    print(result["title"])
    print(result["transcript"][:200])