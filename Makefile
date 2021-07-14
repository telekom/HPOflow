src := hpoflow
other-src := tests docs setup.py

# check the code
check:
	pydocstyle --count $(src) $(other-src)
	black $(src) $(other-src) --check --diff
	flake8 $(src) $(other-src)
	isort $(src) $(other-src) --check --diff
	mdformat --check *.md
	mypy $(src) $(other-src)
	pylint $(src)

# format the code
format:
	black $(src) $(other-src)
	isort $(src) $(other-src)
	mdformat *.md

sphinx:
	cd docs && $(MAKE) clean html && cd ..
