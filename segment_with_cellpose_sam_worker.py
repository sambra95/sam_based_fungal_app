# segment_with_cellpose_sam_worker.py
import sys
import traceback
import numpy as np

# adjust this import to match where your function lives
from helpers.cellpose_functions import segment_with_cellpose_sam


def main():
    if len(sys.argv) != 3:
        print(
            f"Usage: {sys.argv[0]} input.npz output.npz",
            file=sys.stderr,
        )
        sys.exit(2)

    in_path, out_path = sys.argv[1], sys.argv[2]

    try:
        data = np.load(in_path, allow_pickle=True)
        rec = data["rec"].item()
        kwargs = data["kwargs"].item()

        rec_out = segment_with_cellpose_sam(rec, **kwargs)

        np.savez_compressed(out_path, rec=rec_out)

    except Exception:
        # print full traceback so we see it from the parent process
        traceback.print_exc()
        sys.exit(2)


if __name__ == "__main__":
    main()
