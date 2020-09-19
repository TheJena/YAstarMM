# Please define your own dataset path
DATASETS ?= $(DIR_datasets)

.PHONY: black clean docs init mypy update upgrade

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
	clear
	find . -type f -iname "*~"          -exec rm  -fv {} \;
	find . -type d -iname "__pycache__" -exec rm -rfv {} \;

# Renderize markdown documentation into pdf files in docs/
docs:
	clear
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
	clear
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

# Update symbolic link to the most recent available dataset
update:
	clear
	mkdir -p provided_data
	ln --symbolic --force $(shell					\
		find ${DATASETS} -type f -name '00_dataset_*' -exec	\
			stat -c '%Y %n' {} \;				\
		| sort -n						\
		| sed 's/^[0-9]*\s*//'					\
		| tail -n 1)						\
		provided_data/00_dataset_latest.xlsx

# Upgrade all outdated packages with pip
# ( many thanks to https://stackoverflow.com/a/3452888 )
upgrade:
	pip list --outdated --format=freeze				\
	| grep -v '^\-e'						\
	| cut -d = -f 1							\
	| xargs -n1 pip install -U
