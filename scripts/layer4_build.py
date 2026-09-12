#!/usr/bin/env python3
"""Inline the data bundle into the template -> one self-contained HTML file.
Artifacts must be a single file, and the CSP blocks fetching a sibling .js anyway."""
import os
tpl = open("dashboard/template.html").read()
data = open("dashboard/data.js").read()
out = tpl.replace("/*__DATA__*/", data)
open("dashboard/index.html", "w").write(out)
print(f"dashboard/index.html: {os.path.getsize('dashboard/index.html')/1e6:.2f} MB")
