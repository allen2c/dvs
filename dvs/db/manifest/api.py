# dvs/db/manifest/api.py
import logging
import typing

import duckdb

import dvs
import dvs.utils.openapi as openapi_utils
from dvs.types.manifest import Manifest as ManifestType
from dvs.utils.debug_print import debug_print
from dvs.utils.display import DISPLAY_SQL_PARAMS
from dvs.utils.timer import Timer

logger = logging.getLogger(__name__)


class Manifest:
    def __init__(self, dvs: "dvs.DVS"):
        self.dvs = dvs

    def touch(self, *, verbose: bool | None = None) -> bool:
        """
        Create the manifest table if it does not exist.
        Returns True when table creation is completed successfully.
        """
        verbose = self.dvs.verbose if verbose is None else verbose

        with Timer() as timer:
            self._touch(verbose=verbose)

        if verbose:
            dur = timer.duration * 1000
            logger.debug(
                f"Created table: '{dvs.DVS_MANIFEST_TABLE_NAME}' in {dur:.3f} ms"
            )

        return True

    def receive(self, *, verbose: bool | None = None) -> ManifestType | None:
        """
        Retrieve the manifest from the DuckDB database.
        Returns None if no manifest record exists.
        """
        verbose = self.dvs.verbose if verbose is None else verbose

        with Timer() as timer:
            out = self._receive(verbose=verbose)

        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Retrieved manifest in {dur:.3f} ms")

        return out

    def create(
        self, manifest: ManifestType, *, verbose: bool | None = None
    ) -> ManifestType:
        """
        Insert a new manifest record into the DuckDB database.
        Returns the created manifest instance.
        """
        verbose = self.dvs.verbose if verbose is None else verbose

        with Timer() as timer:
            out = self._create(manifest, verbose=verbose)

        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Created manifest in {dur:.3f} ms")

        return out

    def drop(self, *, verbose: bool | None = None) -> bool:
        """
        Drop the manifest table and all its data.
        This operation is irreversible.
        """
        verbose = self.dvs.verbose if verbose is None else verbose

        with Timer() as timer:
            self._drop(verbose=verbose)

        if verbose:
            dur = timer.duration * 1000
            logger.debug(
                f"Dropped table: '{dvs.DVS_MANIFEST_TABLE_NAME}' in {dur:.3f} ms"
            )

        return True

    def _touch(self, *, verbose: bool | None = None) -> bool:
        """
        Internal method to create the manifest table if it does not exist.
        Handles table creation SQL generation and execution.
        """

        with Timer() as timer:
            create_table_sql = openapi_utils.openapi_to_create_table_sql(
                ManifestType.model_json_schema(), table_name=dvs.DVS_MANIFEST_TABLE_NAME
            ).strip()

            try:
                self.dvs.conn.sql(create_table_sql)
            except duckdb.CatalogException as e:
                if "already exists" in str(e).lower():
                    logger.debug(
                        f"Table '{dvs.DVS_MANIFEST_TABLE_NAME}' already exists"
                    )
                else:
                    raise e

        debug_print(
            f"{create_table_sql}",
            title=f"Creating table: '{dvs.DVS_MANIFEST_TABLE_NAME}' with SQL",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )

        return True

    def _receive(self, *, verbose: bool | None = None) -> ManifestType | None:
        """
        Internal method to retrieve the manifest from the DuckDB database.
        Executes SELECT query and validates the result as ManifestType.
        """
        columns = list(ManifestType.model_json_schema()["properties"].keys())
        columns_expr = ",".join(columns)

        query = f"SELECT {columns_expr} FROM {dvs.DVS_MANIFEST_TABLE_NAME}"

        with Timer() as timer:
            result = self.dvs.conn.execute(query).fetchone()

        debug_print(
            f"{query}",
            title="Retrieving manifest with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )

        if result is None:
            return None

        data = dict(zip(columns, result))
        manifest = ManifestType.model_validate(data)

        return manifest

    def _create(
        self, manifest: ManifestType, *, verbose: bool | None = None
    ) -> ManifestType:
        """
        Internal method to insert a manifest record into the DuckDB database.
        Executes INSERT query with manifest data as parameters.
        """
        columns = list(manifest.model_json_schema()["properties"].keys())
        columns_expr = ", ".join(columns)
        placeholders = ", ".join(["?" for _ in columns])
        parameters: typing.List[typing.Tuple[typing.Any, ...]] = [
            tuple(getattr(manifest, c) for c in columns)
        ]

        query = (
            f"INSERT INTO {dvs.DVS_MANIFEST_TABLE_NAME} ({columns_expr}) "
            + f"VALUES ({placeholders})"
        )

        with Timer() as timer:
            self.dvs.conn.executemany(query, parameters)

        debug_print(
            f"{query}\n{DISPLAY_SQL_PARAMS.format(params=parameters)}",
            title="Creating manifest with SQL:",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            verbose=verbose,
        )

        return manifest

    def _drop(self, *, verbose: bool | None = None) -> None:
        """
        Internal method to drop the manifest table.
        """
        query = f"DROP TABLE IF EXISTS {dvs.DVS_MANIFEST_TABLE_NAME}"

        with Timer() as timer:
            self.dvs.conn.execute(query)

        debug_print(
            f"{query}",
            footer=f"Duration: {timer.duration * 1000:.3f} ms",
            title=f"Dropping table: '{dvs.DVS_MANIFEST_TABLE_NAME}' with SQL",
            verbose=verbose,
        )
