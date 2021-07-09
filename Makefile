src := hpoflow tests docs setup.py

# check the code
check:
	pydocstyle --count $(src)
	black $(src) --check --diff
	flake8 $(src)
	isort $(src) --check --diff
	mdformat --check *.md

# format the code
format:
	black $(src)
	isort $(src)
	mdformat *.md

sphinx:
	cd docs && $(MAKE) clean html && cd ..
