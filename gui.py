"""
Run the app using an entrypoint function.

Adapted from alphastats/gui/gui.py

https://github.com/MannLabs/alphapeptstats/blob/HEAD/alphastats/gui/gui.py
"""

import sys
from pathlib import Path
from streamlit.web import cli as stcli

file_location = Path(__file__).parent


def run():
    file_path = str(file_location / "app.py")
    args = [
        "streamlit",
        "run",
        file_path,
        # https://discuss.streamlit.io/t/using-pyinstaller-or-similar-to-create-an-executable/902/4
        "--global.developmentMode=false",
    ]

    # # this is to avoid 'AxiosError: Request failed with status code 403' locally, cf. 
    # https://github.com/streamlit/streamlit/issues/8983
    # # Do not use this in production!
    # if os.environ.get("DISABLE_XSRF", 0):
    #     args.extend(["--server.enableXsrfProtection", "false"])

    sys.argv = args

    sys.exit(stcli.main())

if __name__ == "__main__":
    run()
