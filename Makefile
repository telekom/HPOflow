src := hpoflow tests docs setup.py

# check the code
check:
	style-doc --max_len 99 --check_only $(src)
	black $(src) --check --diff
	flake8 $(src)
	isort $(src) --check --diff
	mdformat --check *.md

# format the code
format:
	style-doc --max_len 99 $(src)
	black $(src)
	isort $(src)
	mdformat *.md

sphinx:
	cd docs && $(MAKE) clean html && cd ..
