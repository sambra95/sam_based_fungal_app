pyinstaller gui.py \
    -y \
    -n mycol_gui \
    --onedir \
    --collect-all streamlit \
    --collect-all cellpose \
    --add-data *.py:. \
    --add-data help_texts:help_texts \
    --add-data helpers:helpers \
    --add-data intro_images:intro_images \
    --add-data views:views \
    --add-data panels:panels

    # --windowed \
    # --icon gui_icon.ico
