"""Genera las guías de la interfaz a partir de las de docs/, para que no se desincronicen:
ui/guia.html desde docs/GUIA_DE_USO.md y ui/guide.html desde docs/USER_GUIDE.md. Ejecutar
tras cambiar una guía:

    python scripts/guia_html.py

Convierte el subconjunto de Markdown que usa la guía: títulos, párrafos, listas (con
sublistas), tablas, bloques de código, citas, negrita, cursiva, código y enlaces.
"""
import html
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# (fuente, destino, lang, título, enlace de vuelta)
GUIDES = [
    ("GUIA_DE_USO.md", "guia.html", "es", "Guía de uso", "← Volver al solver"),
    ("USER_GUIDE.md", "guide.html", "en", "User guide", "← Back to the solver"),
]

HEAD = """<!doctype html>
<html lang="{lang}">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title>
<style>
:root{--bg:#f3f3ef;--panel:#fff;--line:#d9d8d0;--fg:#1e1e1c;--muted:#6f6e68;--accent:#2f6fdb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#161615;--panel:#1f1f1d;--line:#34332f;--fg:#ecebe6;--muted:#9a9993}}
:root[data-theme="dark"]{--bg:#161615;--panel:#1f1f1d;--line:#34332f;--fg:#ecebe6;--muted:#9a9993}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.6 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:860px;margin:0 auto;padding:24px 16px 60px}
h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:28px 0 8px;border-bottom:1px solid var(--line);padding-bottom:4px}
h3{font-size:15px;margin:20px 0 6px}
code{font-family:ui-monospace,Menlo,monospace;font-size:13px;background:var(--panel);border:1px solid var(--line);border-radius:4px;padding:0 4px}
pre{background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:10px 12px;overflow-x:auto}
pre code{border:0;padding:0;background:none}
table{border-collapse:collapse;width:100%;display:block;overflow-x:auto}th,td{text-align:left;padding:6px 8px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:12px;color:var(--muted);text-transform:uppercase}
blockquote{margin:10px 0;padding:8px 14px;border-left:3px solid var(--accent);background:var(--panel)}
hr{border:0;border-top:1px solid var(--line);margin:24px 0}
a{color:var(--accent)}.muted{color:var(--muted)}
ol li,ul li{margin:4px 0}
</style>
<script>try{const t=JSON.parse(localStorage.getItem("theme"));if(t)document.documentElement.dataset.theme=t}catch(e){}</script>
</head>
<body>
<main>
<p class="muted"><a href="/">{back}</a></p>
"""
TAIL = "</main>\n</body>\n</html>\n"


def slug(text):
    """Ancla como la de GitHub: minúsculas, sin puntuación, espacios → guiones."""
    t = re.sub(r"[^\w\- ]", "", text.lower(), flags=re.UNICODE)
    return t.strip().replace(" ", "-")


def inline(t):
    codes = []

    def keep(m):
        codes.append("<code>" + html.escape(m.group(1)) + "</code>")
        return f"\x00{len(codes) - 1}\x00"

    t = re.sub(r"`([^`]+)`", keep, t)
    t = html.escape(t, quote=False)
    t = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", t)
    t = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", t)
    t = re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", lambda m: f'<a href="{m.group(2)}">{m.group(1)}</a>', t)
    return re.sub(r"\x00(\d+)\x00", lambda m: codes[int(m.group(1))], t)


def convert(md):
    lines = md.split("\n")
    out = []
    i = 0
    para = []

    def flush():
        if para:
            out.append("<p>" + inline(" ".join(para)) + "</p>")
            para.clear()

    while i < len(lines):
        line = lines[i]
        s = line.strip()
        if s.startswith("```"):
            flush()
            j = i + 1
            while not lines[j].strip().startswith("```"):
                j += 1
            out.append("<pre><code>" + html.escape("\n".join(lines[i + 1:j])) + "</code></pre>")
            i = j + 1
            continue
        m = re.match(r"(#{1,3}) (.*)", s)
        if m:
            flush()
            n, text = len(m.group(1)), m.group(2)
            out.append(f'<h{n} id="{slug(text)}">{inline(text)}</h{n}>')
            i += 1
            continue
        if s == "---":
            flush()
            out.append("<hr>")
            i += 1
            continue
        if s.startswith("|"):
            flush()
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            body = [r for r in rows[1:] if not all(re.fullmatch(r":?-+:?", c) for c in r)]
            out.append("<table><tr>" + "".join(f"<th>{inline(c)}</th>" for c in rows[0]) + "</tr>")
            out.extend("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>" for r in body)
            out.append("</table>")
            continue
        if s.startswith(">"):
            flush()
            quote = []
            while i < len(lines) and lines[i].strip().startswith(">"):
                quote.append(lines[i].strip().lstrip(">").strip())
                i += 1
            out.append("<blockquote>" + inline(" ".join(quote)) + "</blockquote>")
            continue
        if re.match(r"(\s*)([-*]|\d+\.) ", line):
            flush()
            i = list_block(lines, i, out)
            continue
        if not s:
            flush()
        else:
            para.append(s)
        i += 1
    flush()
    return "\n".join(out)


def list_block(lines, i, out):
    """Lista (y sublistas por sangría) desde la línea i; devuelve la siguiente línea."""
    base = len(lines[i]) - len(lines[i].lstrip())
    ordered = bool(re.match(r"\s*\d+\.", lines[i]))
    out.append("<ol>" if ordered else "<ul>")
    item = None
    while i < len(lines):
        line = lines[i]
        if not line.strip():
            nxt = next((l for l in lines[i + 1:] if l.strip()), "")
            if not re.match(r"(\s*)([-*]|\d+\.) ", nxt) or len(nxt) - len(nxt.lstrip()) < base:
                break
            i += 1
            continue
        ind = len(line) - len(line.lstrip())
        m = re.match(r"\s*(?:[-*]|\d+\.) (.*)", line)
        if m and ind == base:
            if item is not None:
                out.append("<li>" + inline(" ".join(item)) + "</li>")
            item = [m.group(1)]
            i += 1
        elif m and ind > base:
            out.append("<li>" + inline(" ".join(item)))
            item = None
            i = list_block(lines, i, out)
            out.append("</li>")
        elif ind > base and item is not None:
            item.append(line.strip())
            i += 1
        else:
            break
    if item is not None:
        out.append("<li>" + inline(" ".join(item)) + "</li>")
    out.append("</ol>" if ordered else "</ul>")
    return i


def main():
    for src, dst, lang, title, back in GUIDES:
        md = open(os.path.join(ROOT, "docs", src), encoding="utf-8").read()
        # Los otros ficheros del repositorio no existen en la interfaz: fuera el enlace a la
        # otra versión y los enlaces a .md quedan como texto.
        md = re.sub(r"^\*\[(?:English version|Versión en español)\]\([^)]*\)\*\n+", "", md, flags=re.M)
        md = re.sub(r"\[([^\]]+)\]\([^)\s]*\.md\)", r"\1", md)
        head = HEAD.replace("{lang}", lang).replace("{title}", title).replace("{back}", back)
        page = head + convert(md) + "\n" + TAIL
        out = os.path.join(ROOT, "ui", dst)
        open(out, "w", encoding="utf-8", newline="\n").write(page)
        print(f"{out}: {len(page)} bytes")


if __name__ == "__main__":
    main()
