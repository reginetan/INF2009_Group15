"""
enroll_from_image.py
--------------------
Development helper -- enroll students from local image files.
Usage:
    python enroll_from_image.py --id T0034426A --name "Rayner Tan" --image images/ray1.jpeg
    python enroll_from_image.py --folder images/
    python enroll_from_image.py --list
    python enroll_from_image.py --remove T0034426A
"""
import argparse
import logging
from pathlib import Path
import cv2
from face_recogniser import FaceRecogniser

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

SUPPORTED_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
DEFAULT_IMAGE_DIR = Path(__file__).parent / "images"

def enroll_single(recogniser, student_id, name, image_path):
    img = cv2.imread(image_path)
    if img is None:
        log.error("Cannot read image: %s", image_path)
        return
    ok = recogniser.enroll(student_id, name, img)
    print(f"{'OK Enrolled' if ok else 'FAILED':12s}  ID={student_id}  Name={name}")

def enroll_folder(recogniser, folder):
    folder_path = Path(folder)
    images = [p for p in folder_path.iterdir() if p.suffix.lower() in SUPPORTED_EXTS]
    if not images:
        log.warning("No images found in %s", folder_path)
        return
    for img_path in sorted(images):
        stem = img_path.stem
        if "_" in stem:
            sid, name = stem.split("_", 1)
        else:
            sid = name = stem
        enroll_single(recogniser, sid, name, str(img_path))

def main():
    parser = argparse.ArgumentParser(description="Enroll students from local image files")
    parser.add_argument("--id",     type=str)
    parser.add_argument("--name",   type=str)
    parser.add_argument("--image",  type=str)
    parser.add_argument("--folder", type=str)
    parser.add_argument("--list",   action="store_true")
    parser.add_argument("--remove", type=str, metavar="STUDENT_ID")
    args = parser.parse_args()

    recogniser = FaceRecogniser()

    if args.list:
        students = recogniser.db.list_all()
        if not students:
            print("No students enrolled yet.")
        else:
            print(f"\n{'ID':<15} {'Name':<25} {'Enrolled At'}")
            print("-" * 65)
            for s in students:
                print(f"{s['student_id']:<15} {s['name']:<25} {s['enrolled_at']}  embeddings={s['num_embeddings']}")
            print(f"\nTotal: {len(students)} student(s)")
        return

    if args.remove:
        ok = recogniser.db.remove(args.remove)
        print(f"{'Removed' if ok else 'Not found'}: {args.remove}")
        return

    if args.folder:
        enroll_folder(recogniser, args.folder)
        return

    if args.image:
        if not args.id or not args.name:
            parser.error("--image requires --id and --name")
        enroll_single(recogniser, args.id, args.name, args.image)
        return

    log.info("No arguments given -- enrolling from default folder: %s", DEFAULT_IMAGE_DIR)
    enroll_folder(recogniser, str(DEFAULT_IMAGE_DIR))

if __name__ == "__main__":
    main()
