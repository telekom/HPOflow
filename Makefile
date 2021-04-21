.PHONY: style

src := hpoflow tests setup.py

# format the code
format:
	black $(src)
	isort $(src)

# check the code
check:
	black $(src) --check --diff
	flake8 $(src)
