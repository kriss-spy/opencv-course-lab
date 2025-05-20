MD_FILE = report.md
TEMPLATE_FILE = template.tex
OUTPUT_PDF = report.pdf
OUTPUT_DOCX = report.docx
BIB_FILE = references.bib
BUILD_DIR = build

# Default target: build the PDF
all: pdf

# Conditional target to also build DOCX
ifeq ($(TO_DOCX),true)
all: docx
endif

# Rule to create the PDF
pdf: $(BUILD_DIR)/$(OUTPUT_PDF)

$(BUILD_DIR)/$(OUTPUT_PDF): $(MD_FILE) $(TEMPLATE_FILE)
	@mkdir -p $(BUILD_DIR)
	pandoc "$(MD_FILE)" \
		--lua-filter=obsidian-link.lua \
		--highlight-style=tango \
		--template=$(TEMPLATE_FILE) \
		--bibliography="$(CURDIR)/$(BIB_FILE)" \
		--biblatex --pdf-engine=latexmk \
		--pdf-engine-opt=-xelatex -o \
		$(BUILD_DIR)/$(OUTPUT_PDF)

# Rule to create the DOCX
# docx: $(BUILD_DIR)/$(OUTPUT_DOCX)

# $(BUILD_DIR)/$(OUTPUT_PDF): $(MD_FILE) $(TEMPLATE_FILE)
# 	@mkdir -p $(BUILD_DIR)
# 	pandoc "$(MD_FILE)" \
# 		--lua-filter=obsidian-link.lua \
# 		--highlight-style=tango \
# 		--template=$(TEMPLATE_FILE) \
# 		--pdf-engine=latexmk --pdf-engine-opt=-xelatex \
# 		-o $(BUILD_DIR)/$(OUTPUT_PDF)

# Target to clean up build files
clean:
	rm -rf $(BUILD_DIR)

# Phony targets are not files
.PHONY: all clean pdf docx
