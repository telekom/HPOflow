src := hpoflow
test-src := tests
other-src := docs setup.py

check:
	pydocstyle --count $(src) $(test-src) $(other-src)
	black $(src) $(test-src) $(other-src) --check --diff
	flake8 $(src) $(test-src) $(other-src)
	isort $(src) $(test-src) $(other-src) --check --diff
	mdformat --check *.md
	mypy --install-types --non-interactive $(src) $(test-src) $(other-src)
	pylint $(src)

format:
	black $(src) $(test-src) $(other-src)
	isort $(src) $(test-src) $(other-src)
	mdformat *.md

test:
	pytest $(test-src)

sphinx:
	cd docs && $(MAKE) clean html && cd ..

page:
	xdg-open docs/build/html/index.html
