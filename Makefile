.PHONY: style

src := hpoflow tests setup.py

# format the code
format:
	black $(src)
	isort $(src)
	mdformat *.md

# check the code
check:
	black $(src) --check --diff
	flake8 $(src)
	mdformat --check *.md
