"""
CLI entry point.

Examples
--------
# Train with sample data
python main.py train --data sample_data.json --epochs 30

# Score a single pair
python main.py score --purpose "일본어 실력 향상" --action "자막 없이 애니메이션 시청"

# Interactive mode
python main.py interactive
"""

import argparse
import json
import sys


def cmd_train(args):
    from train import train

    kwargs = {}
    if args.embedding_model:
        kwargs["embedding_kwargs"] = {"model_name": args.embedding_model}

    result = train(
        data_path=args.data,
        save_path=args.save,
        embedding_backend=args.backend,
        embedding_kwargs=kwargs.get("embedding_kwargs"),
        val_split=args.val_split,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        cache_dir=args.cache_dir,
    )
    print(f"\n✅ Training complete. Best val loss: {result['best_val_loss']:.4f}")
    print(f"   Model saved to: {args.save}")


def cmd_score(args):
    from inference import RelevanceAI

    ai = RelevanceAI.load(args.checkpoint)
    print(ai.describe(args.purpose, args.action))


def cmd_interactive(args):
    from inference import RelevanceAI

    print(f"Loading model from {args.checkpoint} …")
    ai = RelevanceAI.load(args.checkpoint)
    print("✅ 모델 로드 완료. 'q' 입력 시 종료.\n")

    while True:
        purpose = input("목적 (purpose): ").strip()
        if purpose.lower() == "q":
            break
        action = input("행위 (action) : ").strip()
        if action.lower() == "q":
            break
        print()
        print(ai.describe(purpose, action))
        print("-" * 40)


def main():
    parser = argparse.ArgumentParser(description="Purpose-Action Relevance AI")
    sub = parser.add_subparsers(dest="command", required=True)

    # ── train ──
    t = sub.add_parser("train", help="Train the model")
    t.add_argument("--data",           default="sample_data.json")
    t.add_argument("--save",           default="relevance_model.pt")
    t.add_argument("--backend",        default="sentence-transformers",
                   choices=["sentence-transformers", "openai"])
    t.add_argument("--embedding-model", default=None,
                   help="Override the default embedding model name")
    t.add_argument("--epochs",         type=int,   default=30)
    t.add_argument("--batch-size",     type=int,   default=16)
    t.add_argument("--lr",             type=float, default=3e-4)
    t.add_argument("--val-split",      type=float, default=0.15)
    t.add_argument("--cache-dir",      default="emb_cache")

    # ── score ──
    s = sub.add_parser("score", help="Score a single (purpose, action) pair")
    s.add_argument("--checkpoint", default="relevance_model.pt")
    s.add_argument("--purpose",    required=True)
    s.add_argument("--action",     required=True)

    # ── interactive ──
    i = sub.add_parser("interactive", help="Interactive scoring loop")
    i.add_argument("--checkpoint", default="relevance_model.pt")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "score":
        cmd_score(args)
    elif args.command == "interactive":
        cmd_interactive(args)


if __name__ == "__main__":
    main()
