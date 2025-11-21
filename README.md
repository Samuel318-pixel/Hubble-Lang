We have support for Windows, but if you use Linux, install Python and PyInstaller with pip3 install pyinstaller, then use pyinstaller --onefile hubble.py. After that, put it in a fixed folder, for example: /home/YOUR_USERNAME/hubble/ or, if you want to make it global, /usr/local/bin/. Create a file called:

hbl #!/bin/bash
"/home/YOUR_USERNAME/hubble/hubble" "$@"
⚠️ IMPORTANT:
Replace YOUR_USERNAME with your Linux username.

✅ STEP 3 — Make it executable

In the terminal:

chmod +x hbl

✅ STEP 4 — Install the command in the system

To add the command to the Linux PATH:

sudo mv hbl /usr/local/bin/

The bash will now recognize:

hbl

from any folder.

🎉 NOW YOU CAN RUN:
hbl my_file.hb

or:
hbl "file with spaces.hb"

and the same goes for macOS
