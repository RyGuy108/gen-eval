"""Contact sheet of every output for some items: python scripts/contact_sheet.py RUN_ID ITEM [ITEM...] out.jpg"""
import sys

from PIL import Image, ImageDraw

from geneval import db, media

run, items, out = int(sys.argv[1]), sys.argv[2:-1], sys.argv[-1]
T = 256
rows = []
with db.connect() as c:
    for it in items:
        gs = c.execute("""SELECT g.model, g.paths, s.metrics FROM generations g LEFT JOIN scores s
                          ON s.generation_id=g.id AND s.scorer='quality' WHERE run_id=%s AND item_id=%s
                          AND status='complete' ORDER BY g.id""", (run, it)).fetchall()
        rows.append((it, gs))
W = max(len(gs) for _, gs in rows)
sheet = Image.new("RGB", (W * T, len(rows) * (T + 34)), "white")
d = ImageDraw.Draw(sheet)
for r, (it, gs) in enumerate(rows):
    for col, g in enumerate(gs):
        fr = media.load(g["paths"][0], 3)
        im = fr[-1] if len(fr) > 1 else fr[0]
        im.thumbnail((T, T))
        x, y = col * T, r * (T + 34)
        sheet.paste(im, (x, y))
        m = g["metrics"] or {}
        d.text((x + 4, y + T + 2), f"{it} {g['model']}", fill="black")
        d.text((x + 4, y + T + 16), f"q={m.get('q', 0):.2f} pass={m.get('pass')}", fill="black")
sheet.save(out, quality=85)
