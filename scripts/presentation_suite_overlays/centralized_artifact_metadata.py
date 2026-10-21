"""Migrate portable SN producers to the shared internal metadata store."""

from pathlib import Path
import runpy

HERE = Path(__file__).resolve().parent
LEGACY_INTERMEDIATE = runpy.run_path(str(HERE / "artifact_publication.py"))["_HELPER"]
LEGACY_DELIVERY = runpy.run_path(str(HERE / "delivery_scope.py"))["HELPERS"]

INTERMEDIATE = '''def _mark_intermediate_artifact(filename):
    """Keep this exact QA output private during automatic artifact discovery."""
    from pathlib import Path
    import runpy
    helper = runpy.run_path(str(Path(__file__).with_name("artifact_metadata.py")))
    helper["write_metadata"](filename, {"type": "intermediate_asset"})


'''

DELIVERY = '''def _declare_delivery_scope(root):
    from pathlib import Path
    import runpy
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    helper = runpy.run_path(str(Path(__file__).with_name("artifact_metadata.py")))
    helper["write_scope"](root, {"schema_version": 1, "default": "intermediate"})


def _publish_delivery_file(filename):
    from pathlib import Path
    import runpy
    target = Path(filename)
    if not target.is_file():
        raise ValueError(f"Delivery file does not exist: {target}")
    helper = runpy.run_path(str(Path(__file__).with_name("artifact_metadata.py")))
    helper["write_metadata"](target, {"type": "artifact"})


'''


def apply(relative, data):
    intermediate = {
        "skills/sn-ppt-standard/scripts/render.py",
        "skills/sn-ppt-standard/scripts/deck.py",
        "skills/sn-ppt-dazzle/scripts/render_deck.py",
    }
    delivery = {
        "skills/sn-ppt-standard/scripts/deck.py",
        "skills/sn-ppt-entry/scripts/parse_user_docs.py",
        "skills/sn-ppt-dazzle/scripts/render_deck.py",
    }
    exporter = "skills/sn-ppt-standard/scripts/export_pptx/html_to_pptx.mjs"
    if relative not in intermediate | delivery | {exporter}:
        return data
    text = data.decode("utf-8")
    for targets, old, new in ((intermediate, LEGACY_INTERMEDIATE, INTERMEDIATE),
                              (delivery, LEGACY_DELIVERY, DELIVERY)):
        if relative in targets:
            if text.count(old) != 1:
                raise ValueError(f"Central artifact metadata overlay needs review: {relative}")
            text = text.replace(old, new, 1)
    if relative == exporter:
        old = "  if (result.failCount === 0) writeFileSync(resolve(dirname(outputPath), `.${basename(outputPath)}.artifact.json`), '{\"type\":\"artifact\"}\\n');"
        if text.count(old) != 1:
            raise ValueError("Central artifact metadata exporter overlay needs review")
        # Insert after any hashbang; the exporter uses ESM and the helper is CJS.
        import_line = "import artifactDelivery from '../artifact_delivery.js';\n"
        position = text.index("\n") + 1 if text.startswith("#!") else 0
        text = text[:position] + import_line + text[position:]
        text = text.replace(old, "  if (result.failCount === 0) artifactDelivery.publishArtifact(outputPath);", 1)
    return text.encode("utf-8")
