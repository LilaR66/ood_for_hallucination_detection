.PHONY: create_env activate_env clean lint make_public help

#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROFILE = default
PROJECT_NAME = ood_for_hallucination_detection
CONDA_ENV_NAME = oodhallu_env

# detect conda base path dynamically
CONDA_BASE := $(shell conda info --base)
CONDA_SH := $(CONDA_BASE)/etc/profile.d/conda.sh

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Delete all compiled Python files
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using flake8
lint:
	flake8 src


## Create conda env and install dependencies
create_env:
	@echo "Creating conda environment $(CONDA_ENV_NAME) with Python 3.11.13..."
	@conda create --name $(CONDA_ENV_NAME) python=3.11.13 -y
	@echo "Activating conda environment and installing dependencies..."
	@bash -c "source $(CONDA_SH) && conda activate $(CONDA_ENV_NAME) && pip install --upgrade pip && pip install -r requirements.txt && pip install -e ."

## Activate conda environment (run in interactive shell)
activate_env:
	@echo "To activate the environment, run:"
	@echo "source $(CONDA_SH) && conda activate $(CONDA_ENV_NAME)"

## Make the project public
make_public:
	@ mkdir ood_for_hallucination_detection_public
	@ mkdir ood_for_hallucination_detection_public/src
	@ cp -rf src/src ood_for_hallucination_detection_public/src/src
	@ touch ood_for_hallucination_detection_public/demo_notebook.ipynb
	@ cp .submission_template/* ood_for_hallucination_detection_public/
	@ cp requirements.txt ood_for_hallucination_detection_public/
	@ cp .gitignore ood_for_hallucination_detection_public/
	@ . .submission_template/tuto.sh
	@ rm -rf ood_for_hallucination_detection_public/tuto.sh


#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

# Inspired by <http://marmelab.com/blog/2016/02/29/auto-documented-makefile.html>
# sed script explained:
# /^##/:
# 	* save line in hold space
# 	* purge line
# 	* Loop:
# 		* append newline + line to hold space
# 		* go to next line
# 		* if line starts with doc comment, strip comment character off and loop
# 	* remove target prerequisites
# 	* append hold space (+ newline) to line
# 	* replace newline plus comments by `---`
# 	* print line
# Separate expressions are necessary because labels cannot be delimited by
# semicolon; see <http://stackoverflow.com/a/11799865/1968>
.PHONY: help
help:
	@echo "$$(tput bold)Available rules:$$(tput sgr0)"
	@echo
	@sed -n -e "/^## / { \
		h; \
		s/.*//; \
		:doc" \
		-e "H; \
		n; \
		s/^## //; \
		t doc" \
		-e "s/:.*//; \
		G; \
		s/\\n## /---/; \
		s/\\n/ /g; \
		p; \
	}" ${MAKEFILE_LIST} \
	| LC_ALL='C' sort --ignore-case \
	| awk -F '---' \
		-v ncol=$$(tput cols) \
		-v indent=19 \
		-v col_on="$$(tput setaf 6)" \
		-v col_off="$$(tput sgr0)" \
	'{ \
		printf "%s%*s%s ", col_on, -indent, $$1, col_off; \
		n = split($$2, words, " "); \
		line_length = ncol - indent; \
		for (i = 1; i <= n; i++) { \
			line_length -= length(words[i]) + 1; \
			if (line_length <= 0) { \
				line_length = ncol - indent - length(words[i]) - 1; \
				printf "\n%*s ", -indent, " "; \
			} \
			printf "%s ", words[i]; \
		} \
		printf "\n"; \
	}' \
	| more $(shell test $(shell uname) = Darwin && echo '--no-init --raw-control-chars')
