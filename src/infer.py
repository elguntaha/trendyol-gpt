from dataclasses import asdict
from src.config import load_config, build_argparser
from src.utils import setup_logging

def main():
    ap = build_argparser()
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    
    if args.print:
        import json
        print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
        return
    
    log = setup_logging(cfg.logging.level)
    log.info("Loaded config; running inference stub...")
    # TODO: load models, build features for test, write cfg.inference.output_csv

if __name__ == "__main__":
    main()
