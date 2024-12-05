from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import httpx


def dummy_httpx_request(
    method: str = "GET", url: str = "http://localhost"
) -> "httpx.Request":
    import httpx

    return httpx.Request(method, url)


def dummy_httpx_response(
    status_code: int = 501,
    content: bytes | None = b"Not Implemented",
    *,
    request: Optional["httpx.Request"] = None,
) -> "httpx.Response":
    import httpx

    return httpx.Response(
        status_code=status_code,
        request=request or dummy_httpx_request(),
        content=content,
    )
