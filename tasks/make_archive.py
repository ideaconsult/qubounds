from pathlib import Path
import zipfile

# + tags=["parameters"]
product = None
folders_to_scan = None
extensions_to_include = [".xlsx"]
enabled = False
recursive = False
# -


def zip_files_from_folders(
    folders,
    extensions,
    output_zip,
    recursive=True
):
    """
    Create a ZIP file containing files with specific extensions
    from multiple folders.

    Parameters
    ----------
    folders : list[str or Path]
        List of directories to scan
    extensions : set[str]
        Example: {".txt", ".csv", ".pdf"}
    output_zip : str or Path
        Output zip filename
    recursive : bool
        Search subdirectories if True
    """

    folders = [Path(f) for f in folders]
    extensions = {ext.lower() for ext in extensions}
    output_zip = Path(output_zip)

    with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:

        for folder in folders:
            if not folder.exists():
                print(f"Skipping missing folder: {folder}")
                continue

            if recursive:
                files = folder.rglob("*")
            else:
                files = folder.glob("*")

            for file in files:
                if file.is_file() and file.suffix.lower() in extensions:
                    # Keep relative path inside zip
                    arcname = file.relative_to(folder.parent)
                    zipf.write(file, arcname)

    print(f"ZIP created: {output_zip}")


if enabled:
    zip_files_from_folders(
        folders=folders_to_scan,
        extensions=set(extensions_to_include),
        output_zip=product["zip"],
        recursive=recursive
    )
