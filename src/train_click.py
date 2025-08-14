from dataclasses import asdict
from src.config import load_config, build_argparser
from src.utils import set_global_seed, setup_logging, save_config_snapshot

def main():
    ap = build_argparser()
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.set)
    
    if args.print:
        import json
        print(json.dumps(asdict(cfg), indent=2, ensure_ascii=False))
        return
    
    log = setup_logging(cfg.logging.level)
    set_global_seed(cfg.seed)
    if cfg.logging.save_config_copy:
        save_config_snapshot(asdict(cfg), cfg.paths.runs_dir, cfg.logging.run_name_prefix + "_click")
    log.info("Loaded config; training CLICK model stub...")
    # TODO: implement training; use cfg.models['click']

if __name__ == "__main__":
    main()
