src := hpoflow
test-src := tests
other-src := docs setup.py

# check the code
check:
	pydocstyle --count $(src) $(test-src) $(other-src)
	black $(src) $(test-src) $(other-src) --check --diff
	flake8 $(src) $(test-src) $(other-src)
	isort $(src) $(test-src) $(other-src) --check --diff
	mdformat --check *.md
	mypy --install-types --non-interactive $(src) $(test-src) $(other-src)
	pylint $(src)

# format the code
format:
	black $(src) $(test-src) $(other-src)
	isort $(src) $(test-src) $(other-src)
	mdformat *.md

test:
	pytest $(test-src)

sphinx:
	cd docs && $(MAKE) clean html && cd ..
