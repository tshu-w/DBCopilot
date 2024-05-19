import sys
from pathlib import Path

from src.models import SchemaEncoder

if __name__ == "__main__":
    # Covert the model to HuggingFace format
    for arg in sys.argv[1:]:
        ckpt_path = Path(arg)
        model = SchemaEncoder.load_from_checkpoint(ckpt_path)
        tgt_path = ckpt_path.parent / "model"
        model.model.save_pretrained(tgt_path)
        model.collate_fn.tokenizer.save_pretrained(tgt_path)
