# Python 3.12.0 — This repo only

The repo requires **Python 3.12.0** exactly. The main python.org download page gives **3.12.3**, which is wrong for this project. Use the steps below so you end up with 3.12.0.

---

## Direct download (3.12.0 only)

| Platform | File | Direct link |
|----------|------|-------------|
| **Windows 64-bit** | `python-3.12.0-amd64.exe` | https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe |
| **Windows 32-bit** | `python-3.12.0.exe` | https://www.python.org/ftp/python/3.12.0/python-3.12.0.exe |
| **Release page** | All 3.12.0 files | https://www.python.org/downloads/release/python-3120/ |

Do **not** use:
- https://www.python.org/downloads/ (that offers 3.12.3)
- Microsoft Store, `winget install Python.Python.3.12`, or generic “Python 3.12” — they install the latest 3.12.x (3.12.3).

---

## Windows: install 3.12.0 and create the venv

1. **Download the 3.12.0 installer** (64-bit):
   - https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe

2. **Run the installer.**
   - Check **“Add python.exe to PATH”** if you want `python` in the terminal to be 3.12.0 (optional).
   - Click **“Customize installation”**.
   - **Optional but recommended:** set **“Customize install location”** to:
     - `C:\Python3120`
     - So it does not overwrite an existing `C:\Python312\` (which might be 3.12.3).
   - Complete the install.

3. **Note the path** where 3.12.0 was installed, for example:
   - `C:\Python3120\` (if you customized), or
   - `C:\Users\<You>\AppData\Local\Programs\Python\Python312\` (default).

4. **Create the venv using that 3.12.0 executable** (from repo root in PowerShell):

   ```powershell
   # Replace with YOUR 3.12.0 path from step 3 (must contain python.exe)
   $Py3120 = "C:\Python3120"

   # Remove old venv (it was built with 3.12.3)
   Remove-Item -Recurse -Force .venv -ErrorAction SilentlyContinue

   # Create venv with 3.12.0 only
   & "$Py3120\python.exe" -m venv .venv

   # Activate and verify
   .\.venv\Scripts\activate
   python --version
   ```
   You must see: **Python 3.12.0**. If you see 3.12.3, the path in `$Py3120` is wrong — fix it and repeat from “Remove old venv”.

5. **If you have multiple Python versions:**  
   To avoid using the wrong one, always create the venv by **full path** as above. After activation, `python` in this repo will be 3.12.0 because it comes from `.venv`.

---

## WSL / Linux: use pyenv so you get 3.12.0

The system `python3` in WSL is often 3.12.3. Use **pyenv** to install and select 3.12.0, then create the venv.

1. **Install pyenv** (if needed):
   ```bash
   curl https://pyenv.run | bash
   ```
   Add to `~/.bashrc` (or `~/.zshrc`):
   ```bash
   export PATH="$HOME/.pyenv/bin:$PATH"
   eval "$(pyenv init -)"
   ```
   Then run `exec $SHELL`.

2. **Install and select Python 3.12.0:**
   ```bash
   pyenv install 3.12.0
   cd /mnt/c/GitHub/CIPHER
   pyenv local 3.12.0
   ```

3. **Confirm the shell uses 3.12.0 before creating the venv:**
   ```bash
   which python
   python --version
   ```
   You must see **Python 3.12.0**. If you see 3.12.3, `pyenv local 3.12.0` did not apply (wrong directory or shell not restarted).

4. **Remove old venv and create a new one with 3.12.0:**
   ```bash
   rm -rf .venv
   python -m venv .venv
   source .venv/bin/activate
   python --version
   ```
   Again, you must see **Python 3.12.0**.

---

## Summary

- **Link for Windows 64-bit:** https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe  
- **Never use** the main “Download Python” page or generic 3.12 installers — they give 3.12.3.  
- **Always** create `.venv` with the 3.12.0 executable (by path on Windows, or via `pyenv local 3.12.0` on WSL).  
- After `python --version` shows 3.12.0 in the activated venv, you’re set.
