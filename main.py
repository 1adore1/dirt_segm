from src.cli import CLIInterface

cli = CLIInterface()
cfg = cli.parse_args()
cli.run(cfg)
