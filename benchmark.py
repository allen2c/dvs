import time
from statistics import mean, stdev
from typing import List, Optional, Text, Tuple

import requests
from faker import Faker

BASE_URL = "http://localhost:8000"
URL_SEARCH = f"{BASE_URL}/search"
URL_BULK_SEARCH = f"{BASE_URL}/bulk_search"

MAX_USERS = 10
SENTENCE_LENGTH = 100
TIME_OUT = 30.0
TOP_K = 20
TOTAL_TASKS = 100
WITH_EMBEDDING = False

fake = Faker()


API_SEARCH_REQUEST = Tuple[Text, int, bool]
API_BULK_SEARCH_REQUEST = List[API_SEARCH_REQUEST]
API_RESULT = Tuple[Text, int, float, Optional[Text]]


# Warm-up function
def warmup_api():
    for _ in range(3):
        api_health_check()
    for _ in range(3):
        api_search((fake.sentence(SENTENCE_LENGTH), TOP_K, WITH_EMBEDDING))
    for _ in range(3):
        reqs = [(fake.sentence(SENTENCE_LENGTH), TOP_K, WITH_EMBEDDING)]
        print(api_bulk_search(reqs))


def api_health_check(*args, **kwargs) -> API_RESULT:
    method: Text = "GET"
    status_code: int = 500
    response_time: float = TIME_OUT
    error: Optional[Text] = None

    try:
        start_time = time.perf_counter()
        res = requests.get(BASE_URL, timeout=TIME_OUT)
        response_time = time.perf_counter() - start_time
        status_code = res.status_code
        res.raise_for_status()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    return (method, status_code, response_time, error)


def api_search(req: API_SEARCH_REQUEST, *args, **kwargs) -> API_RESULT:
    method: Text = "POST"
    status_code: int = 500
    response_time: float = TIME_OUT
    error: Optional[Text] = None

    try:
        start_time = time.perf_counter()
        res = requests.post(
            URL_SEARCH,
            json={"query": req[0], "top_k": req[1], "with_embedding": req[2]},
            timeout=TIME_OUT,
        )
        response_time = time.perf_counter() - start_time
        status_code = res.status_code
        res.raise_for_status()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    return (method, status_code, response_time, error)


def api_bulk_search(reqs: API_BULK_SEARCH_REQUEST, *args, **kwargs) -> API_RESULT:
    method: Text = "POST"
    status_code: int = 500
    response_time: float = TIME_OUT
    error: Optional[Text] = None

    try:
        start_time = time.perf_counter()
        res = requests.post(
            URL_BULK_SEARCH,
            json={
                "queries": [
                    {
                        "query": query,
                        "top_k": top_k,
                        "with_embedding": with_embedding,
                    }
                    for query, top_k, with_embedding in reqs
                ]
            },
            timeout=TIME_OUT,
        )
        response_time = time.perf_counter() - start_time
        status_code = res.status_code
        res.raise_for_status()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    return (method, status_code, response_time, error)


def benchmark_api_endpoint(url, method="GET", payload=None, iterations=100):
    response_times = []

    # Warmup phase
    warmup_api()

    for _ in range(iterations):
        start_time = time.perf_counter()
        if method == "GET":
            response = requests.get(url)
        elif method == "POST":
            response = requests.post(url, json=payload)
        execution_time = time.perf_counter() - start_time

        if response.status_code == 200:
            response_times.append(execution_time)

    return {
        "mean": mean(response_times),
        "std_dev": stdev(response_times),
        "min": min(response_times),
        "max": max(response_times),
    }[1][3]


def collect_performance_metrics(url, concurrent_users=10):
    results = {
        "response_times": [],
        "error_count": 0,
        "throughput": 0,
        "concurrent_users": concurrent_users,
    }

    start_time = time.perf_counter()

    # Simulate concurrent users
    for _ in range(concurrent_users):
        try:
            response = requests.get(url)
            results["response_times"].append(response.elapsed.total_seconds())
            if response.status_code != 200:
                results["error_count"] += 1
        except Exception:
            results["error_count"] += 1

    total_time = time.perf_counter() - start_time
    results["throughput"] = concurrent_users / total_time

    return results[1]


def run_load_test(url, duration=300, ramp_up=30):
    """
    Run load test with gradual ramp-up
    duration: test duration in seconds
    ramp_up: time to reach full load in seconds
    """
    results = []
    start_time = time.perf_counter()

    while time.perf_counter() - start_time < duration:
        current_time = time.perf_counter() - start_time
        # Calculate current load based on ramp-up period
        current_load = min(current_time / ramp_up, 1.0)
        concurrent_users = int(max_users * current_load)

        metrics = collect_performance_metrics(url, concurrent_users)
        results.append(metrics)[1][2]


def analyze_benchmark_results(results):
    summary = {
        "avg_response_time": mean([r["mean"] for r in results]),
        "peak_response_time": max([r["max"] for r in results]),
        "error_rate": sum([r["error_count"] for r in results]) / len(results),
        "avg_throughput": mean([r["throughput"] for r in results]),
    }

    return summary[3]


if __name__ == "__main__":
    warmup_api()
    # print(api_search((fake.sentence(SENTENCE_LENGTH), TOP_K, WITH_EMBEDDING)))
    print(api_bulk_search([(fake.sentence(SENTENCE_LENGTH), TOP_K, WITH_EMBEDDING)]))
