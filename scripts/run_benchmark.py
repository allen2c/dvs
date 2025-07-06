#!/usr/bin/env python3
"""
DVS Benchmark Script

A simple, clean benchmark script using BBC datasets to test DVS performance.

Usage:
    python scripts/run_benchmark.py                   # Run with defaults (100 queries)
    python scripts/run_benchmark.py --help            # Show help
    python scripts/run_benchmark.py --top-k 10        # Custom top-k
    python scripts/run_benchmark.py --quick           # Quick benchmark (10 queries)
    python scripts/run_benchmark.py --validate-only   # Validate environment
    python scripts/run_benchmark.py --queries 50      # Custom number of queries
"""

import argparse
import asyncio
import contextlib
import logging
import os
import pathlib
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import diskcache
import matplotlib.pyplot as plt
import openai
import openai_embeddings_model as oai_emb_model
import pandas as pd
import rich.console
from faker import Faker
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

import dvs
from dvs.utils.datasets import download_documents


@dataclass
class BenchmarkConfig:
    """Configuration for benchmark execution."""

    dataset_name: str = "bbc"
    overwrite: bool = False
    data_dir: pathlib.Path = pathlib.Path("./data")
    model: str = "text-embedding-3-small"
    dimensions: int = 512
    top_k: int = 5
    num_queries: int = 100
    quick: bool = False
    validate_only: bool = False
    verbose: bool = False

    @property
    def duckdb_path(self) -> pathlib.Path:
        return self.data_dir / "benchmark.duckdb"

    @property
    def embedding_cache_path(self) -> pathlib.Path:
        return pathlib.Path("./cache/dvs/embeddings.cache")

    @property
    def benchmark_queries(self) -> List[str]:
        """Generate diverse general knowledge queries using Faker."""
        fake = Faker()
        Faker.seed(42)  # For reproducible queries

        # If quick mode, use fewer queries
        query_count = 10 if self.quick else self.num_queries

        # Generate diverse general knowledge and random queries
        queries = []

        for _ in range(query_count):
            query_types = [
                f"What is the meaning of {fake.word().lower()}?",
                f"How to {fake.sentence().lower().rstrip('.')}?",
                f"Why is {fake.word().lower()} important?",
                f"What causes {fake.word().lower()} to happen?",
                f"How does {fake.word().lower()} work?",
                f"What are the benefits of {fake.word().lower()}?",
                f"When did {fake.word().lower()} become popular?",
                f"Where can I find information about {fake.word().lower()}?",
                f"What is {fake.word().lower()}?",
                f"How can I learn about {fake.word().lower()}?",
                f"What are examples of {fake.word().lower()}?",
                f"How is {fake.word().lower()} used?",
                f"What makes {fake.word().lower()} effective?",
                f"How do you understand {fake.word().lower()}?",
                f"What are the types of {fake.word().lower()}?",
            ]
            queries.append(fake.random_element(query_types))

        return queries


class BenchmarkResult:
    """Container for benchmark results with validation."""

    def __init__(self):
        self.setup_time = 0.0
        self.query_times = []
        self.search_results = []
        self.total_documents = 0
        self.total_points = 0
        self.errors = []

    def add_query_time(self, query_time: float):
        """Add query time with validation."""
        if query_time < 0:
            raise ValueError(f"Query time cannot be negative: {query_time}")
        self.query_times.append(query_time)

    def add_error(self, error: str):
        """Add error for tracking."""
        self.errors.append(error)

    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive benchmark statistics."""
        if not self.query_times:
            return {"error": "No query times recorded"}

        return {
            "setup_time_seconds": round(self.setup_time, 3),
            "total_documents": self.total_documents,
            "total_points": self.total_points,
            "queries_tested": len(self.query_times),
            "avg_query_time_ms": round(statistics.mean(self.query_times) * 1000, 2),
            "median_query_time_ms": round(
                statistics.median(self.query_times) * 1000, 2
            ),
            "min_query_time_ms": round(min(self.query_times) * 1000, 2),
            "max_query_time_ms": round(max(self.query_times) * 1000, 2),
            "p95_query_time_ms": round(
                statistics.quantiles(self.query_times, n=20)[18] * 1000, 2
            ),
            "total_query_time_seconds": round(sum(self.query_times), 3),
            "queries_per_second": round(
                len(self.query_times) / sum(self.query_times), 2
            ),
            "error_count": len(self.errors),
        }


class DVSBenchmark:
    """Main benchmark orchestrator."""

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.console = rich.console.Console()
        self.logger = self._setup_logging()
        self.dvs_client: Optional[dvs.DVS] = None

        # Ensure data directory exists
        self.config.data_dir.mkdir(parents=True, exist_ok=True)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger(__name__)
        if self.config.verbose:
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            )
        return logger

    def validate_environment(self) -> bool:
        """Validate that the environment is ready for benchmarking."""
        self.console.print("🔍 Validating environment...")

        # Check OpenAI API key
        if not os.getenv("OPENAI_API_KEY"):
            self.console.print("❌ OPENAI_API_KEY environment variable not set")
            return False

        # Check required directories
        try:
            self.config.embedding_cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.config.data_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.console.print(f"❌ Cannot create required directories: {e}")
            return False

        # Test DVS client creation
        try:
            dvs_settings = dvs.Settings(DUCKDB_PATH=str(self.config.duckdb_path))
            test_client = dvs.DVS(
                dvs_settings,
                model=oai_emb_model.OpenAIEmbeddingsModel(
                    model=self.config.model,
                    openai_client=openai.OpenAI(),
                    cache=diskcache.Cache(self.config.embedding_cache_path),
                ),
                model_settings=oai_emb_model.ModelSettings(
                    dimensions=self.config.dimensions
                ),
                verbose=False,
            )
            # Test basic functionality
            _ = test_client.db.documents.count()
            self.console.print("✅ Environment validation passed")
            return True
        except Exception as e:
            self.console.print(f"❌ DVS client creation failed: {e}")
            return False

    @contextlib.contextmanager
    def _dvs_client_context(self):
        """Context manager for DVS client with proper cleanup."""
        try:
            self.logger.info("Creating DVS client")
            dvs_settings = dvs.Settings(DUCKDB_PATH=str(self.config.duckdb_path))

            self.dvs_client = dvs.DVS(
                dvs_settings,
                model=oai_emb_model.OpenAIEmbeddingsModel(
                    model=self.config.model,
                    openai_client=openai.OpenAI(),
                    cache=diskcache.Cache(self.config.embedding_cache_path),
                ),
                model_settings=oai_emb_model.ModelSettings(
                    dimensions=self.config.dimensions
                ),
                verbose=self.config.verbose,
            )
            yield self.dvs_client
        except Exception as e:
            self.logger.error(f"DVS client creation failed: {e}")
            raise
        finally:
            # Cleanup if needed
            if hasattr(self, "dvs_client") and self.dvs_client:
                self.logger.info("Cleaning up DVS client")
                self.dvs_client = None

    def setup_database(self, dvs_client: dvs.DVS) -> int:
        """Setup database with BBC documents if needed."""
        doc_count = dvs_client.db.documents.count()

        if doc_count == 0:
            self.console.print("📥 Downloading BBC dataset...")
            try:
                docs = download_documents(
                    self.config.dataset_name, overwrite=self.config.overwrite
                )
                self.console.print(f"📊 Adding {len(docs)} documents to database...")
                dvs_client.add(docs, verbose=self.config.verbose)
                doc_count = dvs_client.db.documents.count()
                self.console.print(f"✅ Database ready with {doc_count} documents")
            except Exception as e:
                self.console.print(f"❌ Database setup failed: {e}")
                raise
        else:
            self.console.print(f"✅ Database already contains {doc_count} documents")

        return doc_count

    async def run_benchmark(self) -> BenchmarkResult:
        """Run the benchmark and return results."""
        result = BenchmarkResult()

        self.console.print("🚀 Starting DVS Benchmark")
        self.console.print("=" * 50)

        with self._dvs_client_context() as dvs_client:
            # Setup
            self.console.print("⚙️  Setting up DVS client...")
            setup_start = time.time()

            result.total_documents = self.setup_database(dvs_client)
            result.total_points = dvs_client.db.points.count()
            result.setup_time = time.time() - setup_start

            self.console.print(
                f"📊 Database contains {result.total_points:,} vector points"
            )
            self.console.print()

            # Generate queries
            self.console.print("🎲 Generating benchmark queries...")
            queries = self.config.benchmark_queries
            self.console.print(
                f"✅ Generated {len(queries)} diverse queries using Faker"
            )
            self.console.print()

            # Run queries
            self.console.print(f"🔍 Running {len(queries)} benchmark queries...")

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TimeElapsedColumn(),
                console=self.console,
            ) as progress:
                task = progress.add_task("Running queries...", total=len(queries))

                for i, query in enumerate(queries):
                    progress.update(
                        task,
                        description=f"Query {i+1}/{len(queries)}: {query[:40]}...",
                    )

                    try:
                        query_start = time.time()
                        search_results = await dvs_client.search(
                            query=query, top_k=self.config.top_k
                        )
                        query_time = time.time() - query_start

                        result.add_query_time(query_time)
                        result.search_results.append((query, search_results))

                    except Exception as e:
                        error_msg = f"Query {i+1} failed: {str(e)}"
                        result.add_error(error_msg)
                        self.logger.error(error_msg)

                    progress.advance(task)

        self.console.print("✅ Benchmark completed!")
        return result

    def save_results(self, result: BenchmarkResult):
        """Save benchmark results to files."""
        stats = result.get_stats()

        # Save statistics as CSV
        stats_df = pd.DataFrame([stats])
        stats_path = self.config.data_dir / "benchmark_stats.csv"
        stats_df.to_csv(stats_path, index=False)
        self.console.print(f"📊 Statistics saved to {stats_path}")

        # Save query times as CSV
        if result.query_times:
            query_times_df = pd.DataFrame(
                {
                    "query_number": range(1, len(result.query_times) + 1),
                    "query_time_ms": [t * 1000 for t in result.query_times],
                    "query": [q for q, _ in result.search_results],
                }
            )
            query_times_path = self.config.data_dir / "benchmark_query_times.csv"
            query_times_df.to_csv(query_times_path, index=False)
            self.console.print(f"📈 Query times saved to {query_times_path}")

        # Create visualization
        self.create_plot(result)

    def create_plot(self, result: BenchmarkResult):
        """Create and save benchmark visualization."""
        if not result.query_times:
            self.console.print("⚠️  No query times to plot")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        # Query times over time
        query_times_ms = [t * 1000 for t in result.query_times]
        ax1.plot(
            range(1, len(query_times_ms) + 1),
            query_times_ms,
            "b-o",
            linewidth=2,
            markersize=4,
        )
        ax1.set_xlabel("Query Number")
        ax1.set_ylabel("Response Time (ms)")
        ax1.set_title("Query Response Times Over Time")
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(query_times_ms) * 1.1)

        # Query time distribution
        ax2.hist(
            query_times_ms,
            bins=min(20, len(query_times_ms)),
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
        )
        ax2.set_xlabel("Response Time (ms)")
        ax2.set_ylabel("Frequency")
        ax2.set_title("Query Time Distribution")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        plot_path = self.config.data_dir / "benchmark_results.png"
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        self.console.print(f"📊 Visualization saved to {plot_path}")
        plt.close()

    def print_summary(self, result: BenchmarkResult):
        """Print comprehensive benchmark summary."""
        stats = result.get_stats()

        if "error" in stats:
            self.console.print(f"❌ {stats['error']}")
            return

        self.console.print()
        self.console.print("📊 BENCHMARK SUMMARY")
        self.console.print("=" * 50)
        self.console.print(f"Setup Time: {stats['setup_time_seconds']}s")
        self.console.print(f"Total Documents: {stats['total_documents']:,}")
        self.console.print(f"Total Vector Points: {stats['total_points']:,}")
        self.console.print(f"Queries Tested: {stats['queries_tested']}")
        if stats["error_count"] > 0:
            self.console.print(f"Errors: {stats['error_count']}")
        self.console.print()
        self.console.print("🔍 Query Performance:")
        self.console.print(f"  Average: {stats['avg_query_time_ms']}ms")
        self.console.print(f"  Median:  {stats['median_query_time_ms']}ms")
        self.console.print(f"  95th %:  {stats['p95_query_time_ms']}ms")
        self.console.print(f"  Min:     {stats['min_query_time_ms']}ms")
        self.console.print(f"  Max:     {stats['max_query_time_ms']}ms")
        self.console.print(f"  QPS:     {stats['queries_per_second']}")
        self.console.print()
        self.console.print(f"📁 Output files saved to {self.config.data_dir}/")
        self.console.print("  - benchmark_stats.csv")
        self.console.print("  - benchmark_query_times.csv")
        self.console.print("  - benchmark_results.png")


def parse_args() -> BenchmarkConfig:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="DVS Benchmark Script")
    parser.add_argument("--dataset", default="bbc", help="Dataset name (default: bbc)")
    parser.add_argument(
        "--data-dir", type=pathlib.Path, default="./data", help="Data directory"
    )
    parser.add_argument(
        "--model", default="text-embedding-3-small", help="OpenAI model"
    )
    parser.add_argument(
        "--dimensions", type=int, default=512, help="Embedding dimensions"
    )
    parser.add_argument("--top-k", type=int, default=5, help="Top-k results per query")
    parser.add_argument(
        "--queries",
        type=int,
        default=100,
        help="Number of queries to run (default: 100)",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Run quick benchmark (10 queries)"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing dataset"
    )
    parser.add_argument(
        "--validate-only", action="store_true", help="Only validate environment"
    )
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")

    args = parser.parse_args()

    return BenchmarkConfig(
        dataset_name=args.dataset,
        overwrite=args.overwrite,
        data_dir=args.data_dir,
        model=args.model,
        dimensions=args.dimensions,
        top_k=args.top_k,
        num_queries=args.queries,
        quick=args.quick,
        validate_only=args.validate_only,
        verbose=args.verbose,
    )


async def main():
    """Main benchmark function."""
    config = parse_args()
    benchmark = DVSBenchmark(config)

    try:
        # Validate environment
        if not benchmark.validate_environment():
            sys.exit(1)

        if config.validate_only:
            benchmark.console.print("✅ Environment validation completed")
            return

        # Run benchmark
        result = await benchmark.run_benchmark()
        benchmark.save_results(result)
        benchmark.print_summary(result)

    except KeyboardInterrupt:
        benchmark.console.print("\n❌ Benchmark interrupted by user")
        sys.exit(1)
    except Exception as e:
        benchmark.console.print(f"❌ Benchmark failed: {e}")
        benchmark.logger.exception("Benchmark failed")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
