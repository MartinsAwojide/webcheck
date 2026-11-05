# Thin layer over the demo launcher (webcheck.launcher) for Linux and macOS; on Windows run the uv lines directly.
# demo, stop and status run on the Jetson with its system python3 (it has no uv) in DEMO_DIR; the board targets run
# on a PC with uv. ARGS passes options through, e.g. make demo ARGS="--clear-db max". BOARD=user@host names the Jetson.
DEMO_DIR ?= $(HOME)/webcheck_demo

.PHONY: demo stop status board board-stop board-status board-log

demo:
	python3 $(DEMO_DIR)/webcheck_demo.py start $(ARGS)

stop:
	python3 $(DEMO_DIR)/webcheck_demo.py stop

status:
	python3 $(DEMO_DIR)/webcheck_demo.py status

board:
	uv run webcheck-demo board start $(ARGS)

board-stop:
	uv run webcheck-demo board stop

board-status:
	uv run webcheck-demo board status

board-log:
	uv run webcheck-demo board log
