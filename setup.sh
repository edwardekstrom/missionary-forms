#!/usr/bin/env bash
# ============================================================
# setup.sh — One-time setup for Missionary Forms Processor
# ============================================================
set -e

CYAN='\033[0;36m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${CYAN}=== Missionary Forms Processor Setup ===${NC}\n"

# ----------------------------------------------------------
# 1. Check Python 3.11+
# ----------------------------------------------------------
PYTHON=$(command -v python3.11 || command -v python3 || echo "")
if [ -z "$PYTHON" ]; then
    echo -e "${RED}Python 3.11+ is required. Install from https://www.python.org${NC}"
    exit 1
fi
PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
echo -e "${GREEN}✓ Python: $PY_VERSION${NC}"

# ----------------------------------------------------------
# 2. Create virtual environment
# ----------------------------------------------------------
if [ ! -d "venv" ]; then
    echo -e "\n${CYAN}Creating virtual environment…${NC}"
    $PYTHON -m venv venv
fi
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment active${NC}"

# ----------------------------------------------------------
# 3. Install Python dependencies
# ----------------------------------------------------------
echo -e "\n${CYAN}Installing Python dependencies…${NC}"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
echo -e "${GREEN}✓ Python packages installed${NC}"

# ----------------------------------------------------------
# 4. Install Playwright browser
# ----------------------------------------------------------
echo -e "\n${CYAN}Installing Playwright Chromium browser…${NC}"
playwright install chromium
echo -e "${GREEN}✓ Chromium installed${NC}"

# ----------------------------------------------------------
# 5. Check Tesseract OCR
# ----------------------------------------------------------
echo -e "\n${CYAN}Checking Tesseract OCR…${NC}"
if command -v tesseract &>/dev/null; then
    TESS_VERSION=$(tesseract --version 2>&1 | head -1)
    echo -e "${GREEN}✓ Tesseract found: $TESS_VERSION${NC}"
else
    echo -e "${YELLOW}⚠ Tesseract not found. Installing via Homebrew…${NC}"
    if command -v brew &>/dev/null; then
        brew install tesseract
        echo -e "${GREEN}✓ Tesseract installed${NC}"
    else
        echo -e "${RED}Homebrew not found. Install Tesseract manually:${NC}"
        echo "  brew install tesseract"
        echo "  (or download from https://github.com/tesseract-ocr/tesseract)"
    fi
fi

# ----------------------------------------------------------
# 6. Check poppler (required by pdf2image)
# ----------------------------------------------------------
echo -e "\n${CYAN}Checking poppler (for PDF→image conversion)…${NC}"
if command -v pdftoppm &>/dev/null; then
    echo -e "${GREEN}✓ poppler found${NC}"
else
    echo -e "${YELLOW}⚠ poppler not found. Installing via Homebrew…${NC}"
    if command -v brew &>/dev/null; then
        brew install poppler
        echo -e "${GREEN}✓ poppler installed${NC}"
    else
        echo -e "${RED}Install poppler manually: brew install poppler${NC}"
    fi
fi

# ----------------------------------------------------------
# 7. Check Ollama (optional)
# ----------------------------------------------------------
echo -e "\n${CYAN}Checking Ollama (optional, for better OCR accuracy)…${NC}"
if command -v ollama &>/dev/null; then
    echo -e "${GREEN}✓ Ollama found${NC}"
    # Check if llava model is available
    if ollama list 2>/dev/null | grep -q "llava"; then
        echo -e "${GREEN}✓ llava model available${NC}"
    else
        echo -e "${YELLOW}⚠ llava model not found.${NC}"
        read -p "  Download llava model now? (~4 GB) [y/N] " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            ollama pull llava
            echo -e "${GREEN}✓ llava model downloaded${NC}"
        else
            echo -e "${YELLOW}  Skipping llava. Tesseract OCR will be used instead.${NC}"
        fi
    fi
else
    echo -e "${YELLOW}⚠ Ollama not installed. To enable vision-based OCR:${NC}"
    echo "  1. Install Ollama: https://ollama.com"
    echo "  2. Run: ollama pull llava"
    echo "  (Tesseract OCR will be used in the meantime)"
fi

# ----------------------------------------------------------
# 8. Create required directories
# ----------------------------------------------------------
echo -e "\n${CYAN}Creating working directories…${NC}"
mkdir -p input processed logs/screenshots
echo -e "${GREEN}✓ Directories: input/  processed/  logs/screenshots/${NC}"

# ----------------------------------------------------------
# 9. Run setup wizard (credentials)
# ----------------------------------------------------------
echo -e "\n${CYAN}=== One more step: save your Church account credentials ===${NC}"
python main.py setup

# ----------------------------------------------------------
# Done
# ----------------------------------------------------------
echo -e "\n${GREEN}=== Setup complete! ===${NC}"
echo ""
echo "  Drop missionary PDF files into:  $(pwd)/input/"
echo ""
echo "  Then start the watcher:"
echo -e "  ${CYAN}source venv/bin/activate && python main.py watch${NC}"
echo ""
echo "  Or process a single PDF:"
echo -e "  ${CYAN}python main.py process path/to/file.pdf${NC}"
echo ""
echo "  Preview extracted data without opening the browser:"
echo -e "  ${CYAN}python main.py preview path/to/file.pdf${NC}"
echo ""
