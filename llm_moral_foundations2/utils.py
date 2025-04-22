import re

def sanitize_filename(s, whitelist=r"a-zA-Z0-9_"):
    # keep space and hyphen in the form of _
    s = s.strip().replace(" ", "_").replace("'", "").replace("-", "_").replace("/", "_")
    # Keep only characters in the whitelist
    return re.sub(f"[^{whitelist}]", "", s)
