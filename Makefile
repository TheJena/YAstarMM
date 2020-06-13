.PHONY: black clean docs init mypy upgrade

# Lets not argue about code style :D
# https://github.com/psf/black#the-uncompromising-code-formatter
black:
	$(eval PY_FILES := $$(shell					\
		find . -iname '*py'					\
		| grep -v '#'						\
		| tr -s '\n' ' '))
	black --diff --line-length 79 --target-version py36		\
		$(PY_FILES)						\

# Remove emacs backup files and python compiled bytecode
clean:
	find . -type f -iname "*~"          -exec rm  -fv {} \;
	find . -type d -iname "__pycache__" -exec rm -rfv {} \;

# Renderize markdown documentation into pdf files in docs/
docs:
	find . -type f -iname "*.md" -print				\
	| sed 's/\.md//'						\
	| xargs -n1 -I@ pandoc @.md					\
		--from=gfm						\
		--output docs/@.pdf					\
		--pdf-engine=xelatex					\
		--variable=geometry:margin=1in				\
		--variable=papersize:a4

# Install required packages with pip
init:
	pip install -r requirements.txt

# Loop forever and show mypy hints at each modification of .py files
mypy:
	while true; do							\
		$(eval PY_FILES := $$(shell				\
			find . -iname '*py'				\
			| grep -v '#'					\
			| tr -s '\n' ' '))				\
		inotifywait --quiet --event modify			\
			$(PY_FILES)					\
		&& clear						\
		&& mypy $(PY_FILES)					\
		; echo -e "\n\treveal_type( expr )\t may help you :D"	\
	; done

# Upgrade all outdated packages with pip
# ( many thanks to https://stackoverflow.com/a/3452888 )
upgrade:
	pip list --outdated --format=freeze				\
	| grep -v '^\-e'						\
	| cut -d = -f 1							\
	| xargs -n1 pip install -U
