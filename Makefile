ifeq ($(OS), Windows_NT)
# Styling
.PHONY: style
style:
	black . --line-length 150
	isort . -rc
	flake8 . --exit-zero
else
# Styling
.PHONY: style
style:
	python3 -m black . --line-length 150
	python3 -m isort . -rc
	python3 -m flake8 . --exit-zero
endif