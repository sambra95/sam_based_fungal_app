# Standalone GUI for mycol app

- can be downloaded as an executable
- executable is standalone, i.e. no further setup is required

## Pyinstaller

- streamlit needs to be manually added to the pyinstaller spec file

For testing:

```bash
# windowd version
bash gui.sh && open ./dist/mycol_gui.app
# cmd version
bash gui.sh && ./dist/mycol_gui/mycol_gui
```
