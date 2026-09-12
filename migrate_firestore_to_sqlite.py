"""
One-time migration: copies every document from the old Firestore metadata
store into the new local SQLite one (see app/metadata_store.py's docstring
for why this switch happened).

Reads via app/metadata_store_firestore.py (the retired Firestore module,
kept specifically so this script has something to read from) and writes
via app/metadata_store.py (the new SQLite module), preserving each
document's original created_at timestamp rather than resetting it to
"now" for every migrated row.

Safe to re-run — writes are upserts keyed by doc_id, so running this
twice just re-copies the same data, it doesn't duplicate anything.

Usage:
    python migrate_firestore_to_sqlite.py
"""
from app import metadata_store, metadata_store_firestore

print("Reading all documents from Firestore...")
rows = metadata_store_firestore.get_all_raw()
print(f"Found {len(rows)} document(s) to migrate.\n")

if not rows:
    print("Nothing to migrate.")
else:
    ok = 0
    failed = 0
    for i, (doc_id, data) in enumerate(rows, 1):
        try:
            created_at = data.get("created_at")
            created_at_iso = created_at.isoformat() if created_at is not None else None

            extra = {
                k: v for k, v in data.items()
                if k not in ("gcs_uri", "image_url", "ocr_text", "created_at")
            }

            metadata_store.save_metadata(
                doc_id,
                data.get("gcs_uri", ""),
                data.get("image_url", ""),
                data.get("ocr_text", ""),
                extra=extra,
                created_at=created_at_iso,
            )
            ok += 1
            if i % 25 == 0 or i == len(rows):
                print(f"  [{i}/{len(rows)}] migrated...")
        except Exception as e:
            failed += 1
            print(f"  [{i}/{len(rows)}] {doc_id}: FAILED - {e}")

    print(f"\nDone. {ok} migrated, {failed} failed.")
    print(f"SQLite file: {metadata_store.DB_PATH}")
