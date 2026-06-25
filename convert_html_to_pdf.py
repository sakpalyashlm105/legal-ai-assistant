#!/usr/bin/env python3
"""
HTML to PDF Converter using Playwright
Converts .htm and .html files to PDF while preserving folder structure.

Installation:
    python -m pip install playwright pypdf
    python -m playwright install chromium

Usage:
    python convert_html_to_pdf.py
    python convert_html_to_pdf.py --source "path/to/source" --output "path/to/output"
    python convert_html_to_pdf.py --overwrite
    python convert_html_to_pdf.py --dry-run
    python convert_html_to_pdf.py --remove-html-after-conversion
    python convert_html_to_pdf.py --remove-empty-folders
    python convert_html_to_pdf.py --remove-html-after-conversion --remove-empty-folders
"""

import argparse
import asyncio
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Tuple
import logging

try:
    from playwright.async_api import async_playwright
except ImportError:
    print("Error: Playwright is not installed.")
    print("Install it using: python -m pip install playwright")
    print("Then install Chromium: python -m playwright install chromium")
    sys.exit(1)

try:
    from pypdf import PdfReader
    from pypdf.errors import PdfReadError
except ImportError:
    print("Error: pypdf is not installed.")
    print("Install it using: python -m pip install pypdf")
    sys.exit(1)


class HTMLToPDFConverter:
    """Converts HTML files to PDF using Playwright."""

    def __init__(
        self,
        source_dir: Path,
        output_dir: Path,
        overwrite: bool = False,
        dry_run: bool = False,
        remove_html: bool = False,
        remove_empty_folders: bool = False,
    ):
        self.source_dir = source_dir.resolve()
        self.output_dir = output_dir.resolve()
        self.overwrite = overwrite
        self.dry_run = dry_run
        self.remove_html = remove_html
        self.remove_empty_folders = remove_empty_folders

        # Conversion stats
        self.total_found = 0
        self.successful = 0
        self.failed = 0
        self.skipped = 0
        self.failed_files: List[Tuple[str, str]] = []

        # HTML removal stats
        self.pairs_found = 0
        self.html_deleted = 0
        self.html_retained = 0
        self.invalid_pdfs = 0

        # Folder cleanup stats
        self.empty_folders_deleted = 0

        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        self.output_dir.mkdir(parents=True, exist_ok=True)

        log_file = self.output_dir / f"conversion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

        logger = logging.getLogger(__name__)
        logger.setLevel(logging.DEBUG)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)

        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        return logger

    def validate_directories(self) -> bool:
        if not self.source_dir.exists():
            self.logger.error(f"Source directory does not exist: {self.source_dir}")
            return False

        if not self.source_dir.is_dir():
            self.logger.error(f"Source path is not a directory: {self.source_dir}")
            return False

        self.logger.info(f"Source directory: {self.source_dir}")
        self.logger.info(f"Output directory: {self.output_dir}")

        if self.dry_run:
            self.logger.info("DRY-RUN MODE: No files will be created, modified, or deleted.")

        return True

    def find_html_files(self) -> List[Path]:
        html_files = []
        for pattern in ["**/*.htm", "**/*.html"]:
            html_files.extend(self.source_dir.glob(pattern))

        html_files = sorted(set(html_files))
        self.total_found = len(html_files)
        self.logger.info(f"Found {self.total_found} HTML file(s) to process")
        return html_files

    def get_output_pdf_path(self, html_file: Path) -> Path:
        relative_path = html_file.relative_to(self.source_dir)
        pdf_relative = relative_path.with_suffix('.pdf')
        return self.output_dir / pdf_relative

    async def convert_file(self, html_file: Path, index: int, total: int) -> bool:
        output_pdf = self.get_output_pdf_path(html_file)

        if output_pdf.exists() and not self.overwrite:
            self.logger.info(
                f"[{index}/{total}] Skipped: {html_file.name} -> {output_pdf.name} (already exists)"
            )
            self.skipped += 1
            return True

        if self.dry_run:
            self.logger.info(
                f"[{index}/{total}] [DRY-RUN] Would convert: {html_file.name} -> {output_pdf.name}"
            )
            self.successful += 1
            return True

        output_pdf.parent.mkdir(parents=True, exist_ok=True)

        try:
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context()
                page = await context.new_page()

                file_url = html_file.as_uri()
                await page.goto(file_url, wait_until="networkidle", timeout=60000)

                await page.pdf(
                    path=str(output_pdf),
                    format="A4",
                    print_background=True,
                    margin={
                        "top": "10mm",
                        "right": "10mm",
                        "bottom": "10mm",
                        "left": "10mm",
                    },
                )

                await browser.close()

            if not self._validate_pdf_basic(output_pdf):
                self.logger.warning(
                    f"[{index}/{total}] Invalid PDF generated: {html_file.name}"
                )
                self.failed += 1
                self.failed_files.append((html_file.name, "PDF validation failed"))
                return False

            self.logger.info(
                f"[{index}/{total}] Converted: {html_file.name} -> {output_pdf.name}"
            )
            self.successful += 1
            return True

        except Exception as e:
            error_msg = str(e)
            self.logger.error(
                f"[{index}/{total}] Failed to convert {html_file.name}: {error_msg}"
            )
            self.failed += 1
            self.failed_files.append((html_file.name, error_msg))
            return False

    def _validate_pdf_basic(self, pdf_path: Path) -> bool:
        """Basic validation: exists and non-empty. Used after conversion."""
        try:
            if not pdf_path.exists():
                return False
            if pdf_path.stat().st_size == 0:
                pdf_path.unlink()
                return False
            return True
        except Exception:
            return False

    def _validate_pdf_strict(self, pdf_path: Path) -> Tuple[bool, str]:
        """
        Strict validation using pypdf:
        - file exists
        - size > 0
        - can be opened by pypdf (not corrupted or encrypted)
        - contains at least one page
        Returns (is_valid, reason).
        """
        if not pdf_path.exists():
            return False, "file does not exist"

        if pdf_path.stat().st_size == 0:
            return False, "file is empty (0 bytes)"

        try:
            reader = PdfReader(str(pdf_path))
            if reader.is_encrypted:
                return False, "file is encrypted"
            if len(reader.pages) == 0:
                return False, "file has zero pages"
            return True, "ok"
        except PdfReadError as e:
            return False, f"corrupted or unreadable: {e}"
        except Exception as e:
            return False, f"error opening file: {e}"

    async def process_all(self) -> bool:
        html_files = self.find_html_files()

        if not html_files:
            self.logger.warning("No HTML files found to process")
            return True

        self.logger.info("=" * 80)
        self.logger.info("Starting HTML to PDF conversion...")
        self.logger.info("=" * 80)

        for index, html_file in enumerate(html_files, 1):
            await self.convert_file(html_file, index, len(html_files))

        return True

    def remove_html_after_conversion(self, html_files: List[Path]) -> None:
        """
        For each HTML file, validate its matching PDF then delete the HTML.
        Skipped PDFs are validated the same way as freshly converted ones.
        """
        self.logger.info("=" * 80)
        self.logger.info("HTML REMOVAL PHASE")
        self.logger.info("=" * 80)

        for html_file in html_files:
            pdf_path = self.get_output_pdf_path(html_file)
            self.pairs_found += 1

            if not pdf_path.exists():
                self.logger.warning(
                    f"RETAINED (no matching PDF): {html_file.name}"
                )
                self.html_retained += 1
                continue

            is_valid, reason = self._validate_pdf_strict(pdf_path)

            if not is_valid:
                self.logger.warning(
                    f"RETAINED (invalid PDF - {reason}): {html_file.name}"
                )
                self.invalid_pdfs += 1
                self.html_retained += 1
                continue

            if self.dry_run:
                self.logger.info(
                    f"[DRY-RUN] Would delete HTML: {html_file} (PDF validated OK)"
                )
                self.html_deleted += 1
            else:
                try:
                    html_file.unlink()
                    self.logger.info(
                        f"DELETED HTML: {html_file.name} (PDF validated OK)"
                    )
                    self.html_deleted += 1
                except Exception as e:
                    self.logger.error(
                        f"RETAINED (could not delete): {html_file.name}: {e}"
                    )
                    self.html_retained += 1

    def cleanup_empty_folders(self) -> None:
        """
        Recursively remove empty folders inside source_dir, deepest first.
        Never removes source_dir itself.
        """
        self.logger.info("=" * 80)
        self.logger.info("EMPTY FOLDER CLEANUP PHASE")
        self.logger.info("=" * 80)

        # Walk bottom-up so deepest directories are processed first
        all_dirs = sorted(
            [d for d in self.source_dir.rglob("*") if d.is_dir()],
            key=lambda d: len(d.parts),
            reverse=True,
        )

        for directory in all_dirs:
            if directory == self.source_dir:
                continue

            # A directory is empty if it has no files or subdirectories remaining
            try:
                contents = list(directory.iterdir())
            except PermissionError as e:
                self.logger.warning(f"Cannot read directory {directory}: {e}")
                continue

            if not contents:
                if self.dry_run:
                    self.logger.info(f"[DRY-RUN] Would delete empty folder: {directory}")
                    self.empty_folders_deleted += 1
                else:
                    try:
                        directory.rmdir()
                        self.logger.info(f"Deleted empty folder: {directory}")
                        self.empty_folders_deleted += 1
                    except Exception as e:
                        self.logger.warning(f"Could not delete folder {directory}: {e}")

    def print_summary(self) -> None:
        self.logger.info("=" * 80)
        self.logger.info("CONVERSION SUMMARY")
        self.logger.info("=" * 80)

        dry = " (DRY-RUN - no changes made)" if self.dry_run else ""

        self.logger.info(f"Dry-run mode:                 {'YES' + dry if self.dry_run else 'NO'}")
        self.logger.info(f"Total HTML files found:       {self.total_found}")
        self.logger.info(f"Successfully converted:       {self.successful}")
        self.logger.info(f"Failed conversions:           {self.failed}")
        self.logger.info(f"Skipped (already exist):      {self.skipped}")

        if self.remove_html:
            self.logger.info(f"HTML/PDF pairs found:         {self.pairs_found}")
            self.logger.info(f"HTML files deleted:           {self.html_deleted}")
            self.logger.info(f"HTML files retained:          {self.html_retained}")
            self.logger.info(f"Invalid PDFs (HTML kept):     {self.invalid_pdfs}")

        if self.remove_empty_folders:
            self.logger.info(f"Empty folders deleted:        {self.empty_folders_deleted}")

        self.logger.info(f"Output folder:                {self.output_dir}")

        if self.failed_files:
            self.logger.info("\nFailed files:")
            for filename, error in self.failed_files:
                self.logger.info(f"  - {filename}: {error}")

        self.logger.info("=" * 80)

    async def run(self) -> bool:
        if not self.validate_directories():
            return False

        html_files = self.find_html_files()

        self.logger.info("=" * 80)
        self.logger.info("Starting HTML to PDF conversion...")
        self.logger.info("=" * 80)

        for index, html_file in enumerate(html_files, 1):
            await self.convert_file(html_file, index, len(html_files))

        if self.remove_html and html_files:
            self.remove_html_after_conversion(html_files)

        if self.remove_empty_folders:
            self.cleanup_empty_folders()

        self.print_summary()

        return self.failed == 0


async def main():
    parser = argparse.ArgumentParser(
        description="Convert HTML files to PDF using Playwright",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python convert_html_to_pdf.py
  python convert_html_to_pdf.py --source "path/to/source" --output "path/to/output"
  python convert_html_to_pdf.py --overwrite
  python convert_html_to_pdf.py --dry-run
  python convert_html_to_pdf.py --remove-html-after-conversion
  python convert_html_to_pdf.py --remove-html-after-conversion --remove-empty-folders
  python convert_html_to_pdf.py --dry-run --remove-html-after-conversion --remove-empty-folders
        """
    )

    default_source = Path(__file__).parent / "data" / "raw"
    default_output = Path(__file__).parent / "data" / "pdf"

    parser.add_argument(
        "--source",
        type=Path,
        default=default_source,
        help=f"Source directory containing HTML files (default: {default_source})",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help=f"Output directory for PDF files (default: {default_output})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing PDF files (default: skip existing)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would happen without creating, overwriting, moving, or deleting any files or folders",
    )
    parser.add_argument(
        "--remove-html-after-conversion",
        action="store_true",
        help=(
            "Delete each source HTML file after its matching PDF is validated "
            "(exists, size > 0, readable by pypdf, has at least one page). "
            "Never deletes the HTML if the PDF is missing, empty, encrypted, corrupted, or has zero pages."
        ),
    )
    parser.add_argument(
        "--remove-empty-folders",
        action="store_true",
        help=(
            "Recursively delete empty folders inside the source directory after HTML removal. "
            "Processes deepest folders first. Never deletes the source directory itself."
        ),
    )

    args = parser.parse_args()

    converter = HTMLToPDFConverter(
        source_dir=args.source,
        output_dir=args.output,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        remove_html=args.remove_html_after_conversion,
        remove_empty_folders=args.remove_empty_folders,
    )

    success = await converter.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\nConversion cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error: {e}")
        sys.exit(1)
