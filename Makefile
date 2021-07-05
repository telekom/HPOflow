.PHONY: style

src := hpoflow tests setup.py

# format the code
format:
	style-doc --max_len 99 --py_only $(src)
	black $(src)
	isort $(src)
	mdformat *.md

# check the code
check:
	style-doc --max_len 99 --check_only --py_only $(src)
	black $(src) --check --diff
	flake8 $(src)
	isort $(src) --check --diff	
	mdformat --check *.md
