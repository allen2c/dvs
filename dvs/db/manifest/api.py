import typing

import dvs
import dvs.utils.openapi as openapi_utils
from dvs.types.manifest import Manifest as ManifestType
from dvs.utils.display import DISPLAY_SQL_QUERY


class Manifest:
    def __init__(self, dvs: "dvs.DVS"):
        self.dvs = dvs

    def touch(self, *, verbose: bool | None = None) -> bool:
        """
        Create the manifest table if it does not exist.
        """
        create_table_sql = openapi_utils.openapi_to_create_table_sql(
            ManifestType.model_json_schema(), table_name=dvs.MANIFEST_TABLE_NAME
        ).strip()

        if verbose:
            self.dvs.settings.console.print(
                "\nCreating manifest table with SQL:\n"
                + f"{DISPLAY_SQL_QUERY.format(sql=create_table_sql)}\n"
            )

        self.dvs.conn.sql(create_table_sql)

        return True

    def receive(self) -> ManifestType | None:
        columns = list(ManifestType.model_json_schema()["properties"].keys())
        columns_expr = ",".join(columns)

        query = f"SELECT {columns_expr} FROM {dvs.MANIFEST_TABLE_NAME}"

        result = self.dvs.conn.execute(query).fetchone()

        if result is None:
            return None

        data = dict(zip(columns, result))
        manifest = ManifestType.model_validate(data)

        return manifest

    def create(self, manifest: ManifestType) -> ManifestType:
        columns = list(manifest.model_json_schema()["properties"].keys())
        columns_expr = ", ".join(columns)
        placeholders = ", ".join(["?" for _ in columns])
        parameters: typing.List[typing.Tuple[typing.Any, ...]] = [
            tuple(getattr(doc, c) for c in columns) for doc in [manifest]
        ]

        query = (
            f"INSERT INTO {dvs.MANIFEST_TABLE_NAME} ({columns_expr}) "
            + f"VALUES ({placeholders})"
        )

        self.dvs.conn.executemany(query, parameters)

        return manifest
