import argparse
from dvs.utils.datasets import download_documents
import typing
import textwrap


def main(name: typing.Literal["bbc"], overwrite: bool = False):
    """Download datasets with specified parameters."""
    download_documents(name, overwrite=overwrite)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download datasets for DVS (Document Vector Store)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent(
            """
            Examples:
                %(prog)s bbc                    # Download BBC dataset
                %(prog)s bbc --overwrite        # Download BBC dataset, overwriting existing files
                %(prog)s bbc --no-overwrite     # Download BBC dataset, skip if exists (default)
            """  # noqa: E501
        ),
    )

    parser.add_argument(
        "name",
        choices=["bbc"],
        help="Name of the dataset to download",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing dataset files if they exist",
    )

    parser.add_argument(
        "--no-overwrite",
        dest="overwrite",
        action="store_false",
        help="Skip download if dataset files already exist (default)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(name=args.name, overwrite=args.overwrite)
