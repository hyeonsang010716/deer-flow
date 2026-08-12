"""InfoQuest Search And Fetch API를 호출하는 유틸리티.

설정 방법은 다음을 참고한다:
https://docs.byteplus.com/en/docs/InfoQuest/What_is_Info_Quest
"""

import json
import logging
import os
from typing import Any

import requests

logger = logging.getLogger(__name__)


class InfoQuestClient:
    """InfoQuest web search 및 fetch API와 통신하는 client."""

    def __init__(self, fetch_time: int = -1, fetch_timeout: int = -1, fetch_navigation_timeout: int = -1, search_time_range: int = -1, image_search_time_range: int = -1, image_size: str = "i"):
        logger.info("\n============================================\n🚀 BytePlus InfoQuest Client Initialization 🚀\n============================================")

        self.fetch_time = fetch_time
        self.fetch_timeout = fetch_timeout
        self.fetch_navigation_timeout = fetch_navigation_timeout
        self.search_time_range = search_time_range
        self.image_search_time_range = image_search_time_range
        self.image_size = image_size
        self.api_key_set = bool(os.getenv("INFOQUEST_API_KEY"))
        if logger.isEnabledFor(logging.DEBUG):
            config_details = (
                f"\n📋 Configuration Details:\n"
                f"├── Fetch time: {fetch_time} {'(Default: No fetch time)' if fetch_time == -1 else '(Custom)'}\n"
                f"├── Fetch Timeout: {fetch_timeout} {'(Default: No fetch timeout)' if fetch_timeout == -1 else '(Custom)'}\n"
                f"├── Navigation Timeout: {fetch_navigation_timeout} {'(Default: No Navigation Timeout)' if fetch_navigation_timeout == -1 else '(Custom)'}\n"
                f"├── Search Time Range: {search_time_range} {'(Default: No Search Time Range)' if search_time_range == -1 else '(Custom)'}\n"
                f"├── Image Search Time Range: {image_search_time_range} {'(Default: No Image Search Time Range)' if image_search_time_range == -1 else '(Custom)'}\n"
                f"├── Image Size: {image_size} {'(Default: Medium)' if image_size == 'm' else '(Custom)'}\n"
                f"└── API Key: {'✅ Configured' if self.api_key_set else '❌ Not set'}"
            )

            logger.debug(config_details)
            logger.debug("\n" + "*" * 70 + "\n")

    def fetch(self, url: str, return_format: str = "html") -> str:
        if logger.isEnabledFor(logging.DEBUG):
            url_truncated = url[:50] + "..." if len(url) > 50 else url
            logger.debug(
                f"InfoQuest - Fetch API request initiated | "
                f"operation=crawl url | "
                f"url_truncated={url_truncated} | "
                f"has_timeout_filter={self.fetch_timeout > 0} | timeout_filter={self.fetch_timeout} | "
                f"has_fetch_time_filter={self.fetch_time > 0} | fetch_time_filter={self.fetch_time} | "
                f"has_navigation_timeout_filter={self.fetch_navigation_timeout > 0} | navi_timeout_filter={self.fetch_navigation_timeout} | "
                f"request_type=sync"
            )

        # header를 준비한다
        headers = self._prepare_headers()

        # request data를 준비한다
        data = self._prepare_crawl_request_data(url, return_format)

        logger.debug("Sending crawl request to InfoQuest API")
        try:
            response = requests.post("https://reader.infoquest.bytepluses.com", headers=headers, json=data)

            # status code가 200이 아닌지 확인한다
            if response.status_code != 200:
                error_message = f"fetch API returned status {response.status_code}: {response.text}"
                logger.debug("InfoQuest Crawler fetch API return status %d: %s for URL: %s", response.status_code, response.text, url)
                return f"Error: {error_message}"

            # 빈 response인지 확인한다
            if not response.text or not response.text.strip():
                error_message = "no result found"
                logger.debug("InfoQuest Crawler returned empty response for URL: %s", url)
                return f"Error: {error_message}"

            # response를 JSON으로 파싱해 reader_result를 추출한다
            try:
                response_data = json.loads(response.text)
                # reader_result가 있으면 추출한다
                if "reader_result" in response_data:
                    logger.debug("Successfully extracted reader_result from JSON response")
                    return response_data["reader_result"]
                elif "content" in response_data:
                    # reader_result가 없으면 content 필드로 fallback한다
                    logger.debug("reader_result missing in JSON response, falling back to content field: %s", response_data["content"])
                    return response_data["content"]
                else:
                    # 둘 다 없으면 원본 response를 반환한다
                    logger.warning("Neither reader_result nor content field found in JSON response")
            except json.JSONDecodeError:
                # response가 JSON이 아니면 원본 텍스트를 반환한다
                logger.debug("Response is not in JSON format, returning as-is")
                return response.text

            # 디버깅용으로 response 일부를 출력한다
            if logger.isEnabledFor(logging.DEBUG):
                response_sample = response.text[:200] + ("..." if len(response.text) > 200 else "")
                logger.debug("Successfully received response, content length: %d bytes, first 200 chars: %s", len(response.text), response_sample)
            return response.text
        except Exception as e:
            error_message = f"fetch API failed: {str(e)}"
            logger.error(error_message)
            return f"Error: {error_message}"

    @staticmethod
    def _prepare_headers() -> dict[str, str]:
        """request header를 준비한다."""
        headers = {
            "Content-Type": "application/json",
        }

        # API key가 있으면 추가한다
        if os.getenv("INFOQUEST_API_KEY"):
            headers["Authorization"] = f"Bearer {os.getenv('INFOQUEST_API_KEY')}"
            logger.debug("API key added to request headers")
        else:
            logger.warning("InfoQuest API key is not set. Provide your own key for authentication.")

        return headers

    def _prepare_crawl_request_data(self, url: str, return_format: str) -> dict[str, Any]:
        """포맷된 파라미터로 request data를 준비한다."""
        # return_format을 정규화한다
        if return_format and return_format.lower() == "html":
            normalized_format = "HTML"
        else:
            normalized_format = return_format

        data = {"url": url, "format": normalized_format}

        # 양수로 설정된 timeout 파라미터를 추가한다
        timeout_params = {}
        if self.fetch_time > 0:
            timeout_params["fetch_time"] = self.fetch_time
        if self.fetch_timeout > 0:
            timeout_params["timeout"] = self.fetch_timeout
        if self.fetch_navigation_timeout > 0:
            timeout_params["navi_timeout"] = self.fetch_navigation_timeout

        # 적용된 timeout 파라미터를 로깅한다
        if timeout_params:
            logger.debug("Applying timeout parameters: %s", timeout_params)
            data.update(timeout_params)

        return data

    def web_search_raw_results(
        self,
        query: str,
        site: str,
        output_format: str = "JSON",
    ) -> dict:
        """InfoQuest Web-Search API에서 결과를 동기적으로 가져온다."""
        headers = self._prepare_headers()

        params = {"format": output_format, "query": query}
        if self.search_time_range > 0:
            params["time_range"] = self.search_time_range

        if site != "":
            params["site"] = site

        response = requests.post("https://search.infoquest.bytepluses.com", headers=headers, json=params)
        response.raise_for_status()

        # 디버깅용으로 response 일부를 출력한다
        response_json = response.json()
        if logger.isEnabledFor(logging.DEBUG):
            response_sample = json.dumps(response_json)[:200] + ("..." if len(json.dumps(response_json)) > 200 else "")
            logger.debug(f"Search API request completed successfully | service=InfoQuest | status=success | response_sample={response_sample}")

        return response_json

    @staticmethod
    def clean_results(raw_results: list[dict[str, dict[str, dict[str, Any]]]]) -> list[dict]:
        """InfoQuest Web-Search API 결과를 정리한다."""
        logger.debug("Processing web-search results")

        seen_urls = set()
        clean_results = []
        counts = {"pages": 0, "news": 0}

        for content_list in raw_results:
            content = content_list["content"]
            results = content["results"]

            if results.get("organic"):
                organic_results = results["organic"]
                for result in organic_results:
                    clean_result = {
                        "type": "page",
                    }
                    if "title" in result:
                        clean_result["title"] = result["title"]
                    if "desc" in result:
                        clean_result["desc"] = result["desc"]
                        clean_result["snippet"] = result["desc"]
                    if "url" in result:
                        clean_result["url"] = result["url"]
                        url = clean_result["url"]
                        if isinstance(url, str) and url and url not in seen_urls:
                            seen_urls.add(url)
                            clean_results.append(clean_result)
                            counts["pages"] += 1

            if results.get("top_stories"):
                news = results["top_stories"]
                for obj in news["items"]:
                    clean_result = {
                        "type": "news",
                    }
                    if "time_frame" in obj:
                        clean_result["time_frame"] = obj["time_frame"]
                    if "source" in obj:
                        clean_result["source"] = obj["source"]
                    title = obj.get("title")
                    url = obj.get("url")
                    if title:
                        clean_result["title"] = title
                    if url:
                        clean_result["url"] = url
                    if title and isinstance(url, str) and url and url not in seen_urls:
                        seen_urls.add(url)
                        clean_results.append(clean_result)
                        counts["news"] += 1
        logger.debug(f"Results processing completed | total_results={len(clean_results)} | pages={counts['pages']} | news_items={counts['news']} | unique_urls={len(seen_urls)}")

        return clean_results

    def web_search(
        self,
        query: str,
        site: str = "",
        output_format: str = "JSON",
    ) -> str:
        if logger.isEnabledFor(logging.DEBUG):
            query_truncated = query[:50] + "..." if len(query) > 50 else query
            logger.debug(
                f"InfoQuest - Search API request initiated | "
                f"operation=search webs | "
                f"query_truncated={query_truncated} | "
                f"has_time_filter={self.search_time_range > 0} | time_filter={self.search_time_range} | "
                f"has_site_filter={bool(site)} | site={site} | "
                f"request_type=sync"
            )

        try:
            logger.debug("InfoQuest Web-Search - Executing search with parameters")
            raw_results = self.web_search_raw_results(
                query,
                site,
                output_format,
            )
            if "search_result" in raw_results:
                logger.debug("InfoQuest Web-Search - Successfully extracted search_result from JSON response")
                results = raw_results["search_result"]

                logger.debug("InfoQuest Web-Search - Processing raw search results")
                cleaned_results = self.clean_results(results["results"])

                result_json = json.dumps(cleaned_results, indent=2, ensure_ascii=False)

                logger.debug(f"InfoQuest Web-Search - Search tool execution completed | mode=synchronous | results_count={len(cleaned_results)}")
                return result_json

            elif "content" in raw_results:
                # search_result가 없으면 content 필드로 fallback한다
                error_message = "web search API return wrong format"
                logger.error("web search API return wrong format, no search_result nor content field found in JSON response, content: %s", raw_results["content"])
                return f"Error: {error_message}"
            else:
                # 둘 다 없으면 원본 response를 반환한다
                logger.warning("InfoQuest Web-Search - Neither search_result nor content field found in JSON response")
                return json.dumps(raw_results, indent=2, ensure_ascii=False)

        except Exception as e:
            error_message = f"InfoQuest Web-Search - Search tool execution failed | mode=synchronous | error={str(e)}"
            logger.error(error_message)
            return f"Error: {error_message}"

    @staticmethod
    def clean_results_with_image_search(raw_results: list[dict[str, dict[str, dict[str, Any]]]]) -> list[dict]:
        """InfoQuest Web-Search API 결과를 정리한다."""
        logger.debug("Processing web-search results")

        seen_urls = set()
        clean_results = []
        counts = {"images": 0}

        for content_list in raw_results:
            content = content_list["content"]
            results = content["results"]

            if results.get("images_results"):
                images_results = results["images_results"]
                for result in images_results:
                    clean_result = {}
                    if "original" in result:
                        clean_result["image_url"] = result["original"]
                        url = clean_result["image_url"]
                        if isinstance(url, str) and url and url not in seen_urls:
                            seen_urls.add(url)
                            clean_results.append(clean_result)
                            counts["images"] += 1
                    if "title" in result:
                        clean_result["title"] = result["title"]
        logger.debug(f"Results processing completed | total_results={len(clean_results)} | images={counts['images']} | unique_urls={len(seen_urls)}")

        return clean_results

    def image_search_raw_results(
        self,
        query: str,
        site: str = "",
        output_format: str = "JSON",
    ) -> dict:
        """InfoQuest Web-Search API에서 image search 결과를 동기적으로 가져온다."""
        headers = self._prepare_headers()

        params = {"format": output_format, "query": query, "search_type": "Images"}

        # 지정된 경우 time_range 필터를 추가한다(1-365)
        if 1 <= self.image_search_time_range <= 365:
            params["time_range"] = self.image_search_time_range
        elif self.image_search_time_range > 0:
            logger.warning(f"time_range {self.image_search_time_range} is out of valid range (1-365), ignoring")

        # 지정된 경우 site 필터를 추가한다
        if site:
            params["site"] = site

        # 지정된 경우 image_size 필터를 추가한다
        if self.image_size and self.image_size in ["l", "m", "i"]:
            params["image_size"] = self.image_size
        elif self.image_size:
            logger.warning(f"image_size {self.image_size} is not valid, must be 'l', 'm', or 'i'")

        response = requests.post("https://search.infoquest.bytepluses.com", headers=headers, json=params)
        response.raise_for_status()

        # 디버깅용으로 response 일부를 출력한다
        response_json = response.json()
        if logger.isEnabledFor(logging.DEBUG):
            response_sample = json.dumps(response_json)[:200] + ("..." if len(json.dumps(response_json)) > 200 else "")
            logger.debug(f"Image Search API request completed successfully | service=InfoQuest | status=success | response_sample={response_sample}")

        return response_json

    def image_search(
        self,
        query: str,
        site: str = "",
        output_format: str = "JSON",
    ) -> str:
        if logger.isEnabledFor(logging.DEBUG):
            query_truncated = query[:50] + "..." if len(query) > 50 else query
            logger.debug(
                f"InfoQuest - Image Search API request initiated | "
                f"operation=search images | "
                f"query_truncated={query_truncated} | "
                f"has_site_filter={bool(site)} | site={site} | "
                f"image_search_time_range={self.image_search_time_range if self.image_search_time_range >= 1 and self.image_search_time_range <= 365 else 'default'} | "
                f"image_size={self.image_size} |"
                f"request_type=sync"
            )

        try:
            logger.info("InfoQuest Image Search - Executing search with parameters")
            raw_results = self.image_search_raw_results(
                query,
                site,
                output_format,
            )

            if "search_result" in raw_results:
                logger.debug("InfoQuest Image Search - Successfully extracted search_result from JSON response")
                results = raw_results["search_result"]

                logger.debug(f"InfoQuest Image Search - Processing raw image search results: {results}")
                cleaned_results = self.clean_results_with_image_search(results["results"])

                result_json = json.dumps(cleaned_results, indent=2, ensure_ascii=False)

                logger.debug(f"InfoQuest Image Search - Image search tool execution completed | mode=synchronous | results_count={len(cleaned_results)}")
                return result_json

            elif "content" in raw_results:
                # search_result가 없으면 content 필드로 fallback한다
                error_message = "image search API return wrong format"
                logger.error("image search API return wrong format, no search_result nor content field found in JSON response, content: %s", raw_results["content"])
                return f"Error: {error_message}"
            else:
                # 둘 다 없으면 원본 response를 반환한다
                logger.warning("InfoQuest Image Search - Neither search_result nor content field found in JSON response")
                return json.dumps(raw_results, indent=2, ensure_ascii=False)

        except Exception as e:
            error_message = f"InfoQuest Image Search - Image search tool execution failed | mode=synchronous | error={str(e)}"
            logger.error(error_message)
            return f"Error: {error_message}"
