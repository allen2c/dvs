import logging
import typing

import dvs
import dvs.utils.openapi as openapi_utils
from dvs.types.manifest import Manifest as ManifestType
from dvs.utils.display import DISPLAY_SQL_PARAMS, DISPLAY_SQL_QUERY
from dvs.utils.timer import Timer

logger = logging.getLogger(__name__)


class Manifest:
    def __init__(self, dvs: "dvs.DVS"):
        self.dvs = dvs

    def touch(self, *, verbose: bool | None = None) -> bool:
        """
        Create the manifest table if it does not exist.
        """
        verbose = self.dvs.verbose if verbose is None else verbose

        with Timer() as timer:
            self._touch(verbose=verbose)

        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Created table: '{dvs.MANIFEST_TABLE_NAME}' in {dur:.3f} ms")

        return True

    def receive(self, *, verbose: bool | None = None) -> ManifestType | None:
        """
        Retrieve the manifest from the DuckDB database.
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
        Create a manifest record in the DuckDB database.
        """
        verbose = self.dvs.verbose if verbose is None else verbose

        with Timer() as timer:
            out = self._create(manifest, verbose=verbose)

        if verbose:
            dur = timer.duration * 1000
            logger.debug(f"Created manifest in {dur:.3f} ms")

        return out

    def _touch(self, *, verbose: bool | None = None) -> bool:
        """
        Create the manifest table if it does not exist.
        """
        create_table_sql = openapi_utils.openapi_to_create_table_sql(
            ManifestType.model_json_schema(), table_name=dvs.MANIFEST_TABLE_NAME
        ).strip()

        if verbose:
            self.dvs.settings.console.print(
                f"\nCreating table: '{dvs.MANIFEST_TABLE_NAME}' with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=create_table_sql)}\n"
            )

        self.dvs.conn.sql(create_table_sql)

        return True

    def _receive(self, *, verbose: bool | None = None) -> ManifestType | None:
        """
        Retrieve the manifest from the DuckDB database.
        """
        columns = list(ManifestType.model_json_schema()["properties"].keys())
        columns_expr = ",".join(columns)

        query = f"SELECT {columns_expr} FROM {dvs.MANIFEST_TABLE_NAME}"

        if verbose:
            self.dvs.settings.console.print(
                "\nRetrieving manifest with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
            )

        result = self.dvs.conn.execute(query).fetchone()

        if result is None:
            return None

        data = dict(zip(columns, result))
        manifest = ManifestType.model_validate(data)

        return manifest

    def _create(
        self, manifest: ManifestType, *, verbose: bool | None = None
    ) -> ManifestType:
        """
        Create a manifest record in the DuckDB database.
        """
        columns = list(manifest.model_json_schema()["properties"].keys())
        columns_expr = ", ".join(columns)
        placeholders = ", ".join(["?" for _ in columns])
        parameters: typing.List[typing.Tuple[typing.Any, ...]] = [
            tuple(getattr(manifest, c) for c in columns)
        ]

        query = (
            f"INSERT INTO {dvs.MANIFEST_TABLE_NAME} ({columns_expr}) "
            + f"VALUES ({placeholders})"
        )

        if verbose:
            self.dvs.settings.console.print(
                "\nCreating manifest with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=query)}\n"
                + f"{DISPLAY_SQL_PARAMS.format(params=parameters)}\n"
            )

        self.dvs.conn.executemany(query, parameters)

        return manifest
