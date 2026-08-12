"""서버 측 웹 도구가 공유하는 URL 안전성 검사."""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from urllib.parse import urlparse

_BLOCKED_HOSTNAMES = {"localhost", "metadata.google.internal"}


def resolve_host_addresses(hostname: str) -> list[ipaddress._BaseAddress]:
    """SSRF 검사를 위해 hostname을 모든 IP 주소로 resolve한다."""
    addresses: list[ipaddress._BaseAddress] = []
    try:
        infos = socket.getaddrinfo(hostname, None)
    except (socket.gaierror, UnicodeError):
        return addresses
    for info in infos:
        sockaddr = info[4]
        try:
            addresses.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    return addresses


def is_blocked_address(address: ipaddress._BaseAddress) -> bool:
    """웹 도구가 기본적으로 접근하면 안 되는 주소면 True를 반환한다."""
    return address.is_private or address.is_loopback or address.is_link_local or address.is_reserved or address.is_multicast or address.is_unspecified


def validate_public_http_url(
    url: str,
    *,
    allow_private_addresses: bool = False,
    action: str = "fetch",
    resolver: Callable[[str], list[ipaddress._BaseAddress]] | None = None,
) -> str | None:
    """서버 측 웹 도구가 fetch하기 전에 http(s) URL을 검증한다.

    거부해야 할 URL이면 ``"Error: ..."`` 문자열을, 진행해도 되면 ``None``을 반환한다.
    self-hosted fetch/render 서비스에 대해 의도적으로 보수적으로 판단한다. 이런 서비스는
    배포 네트워크 안에서 돌기 때문에 그대로 두면 클라우드 metadata나 사설 호스트에 닿을 수 있다.
    """
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return "Error: Only http:// and https:// URLs are supported"

    if allow_private_addresses:
        return None

    hostname = parsed.hostname
    if not hostname:
        return "Error: URL host could not be parsed"

    normalized_host = hostname.strip().rstrip(".").lower()
    if normalized_host in _BLOCKED_HOSTNAMES:
        return f"Error: Refusing to {action} a private or loopback address"

    try:
        literal_ip = ipaddress.ip_address(normalized_host)
    except ValueError:
        literal_ip = None

    if literal_ip is not None:
        candidates = [literal_ip]
    else:
        resolve = resolver or resolve_host_addresses
        candidates = resolve(hostname)
        if not candidates:
            return "Error: URL host could not be resolved"

    if any(is_blocked_address(addr) for addr in candidates):
        return f"Error: Refusing to {action} a private, loopback, or metadata address"
    return None
