"""
Generates 8 sample artisan product images for benchmarking Smart Catalog AI.
"""

from pathlib import Path
from PIL import Image, ImageDraw

SAMPLE_DIR = Path(__file__).parent / "sample_images"
SAMPLE_DIR.mkdir(exist_ok=True)


def make_bamboo_basket():
    img = Image.new("RGB", (400, 400), color="#F4E8D1")
    draw = ImageDraw.Draw(img)
    draw.ellipse([80, 100, 320, 340], fill="#D2B48C", outline="#8B4513", width=6)
    for i in range(120, 330, 20):
        draw.line([80, i, 320, i], fill="#8B4513", width=2)
    for j in range(100, 330, 20):
        draw.line([j, 100, j, 340], fill="#8B4513", width=2)
    draw.text((100, 40), "Handmade Bamboo Basket", fill="#333333")
    path = SAMPLE_DIR / "bamboo_basket.jpg"
    img.save(path, format="JPEG")
    return path


def make_terracotta_vase():
    img = Image.new("RGB", (400, 400), color="#EAEAEA")
    draw = ImageDraw.Draw(img)
    draw.polygon([(150, 80), (250, 80), (280, 200), (230, 340), (170, 340), (120, 200)], fill="#C86D51", outline="#8B3A2B", width=4)
    draw.ellipse([180, 180, 220, 220], fill="#FFFFFF", outline="#000000")
    draw.text((110, 40), "Terracotta Clay Vase", fill="#333333")
    path = SAMPLE_DIR / "terracotta_vase.jpg"
    img.save(path, format="JPEG")
    return path


def make_brass_diya():
    img = Image.new("RGB", (400, 400), color="#2C2C2C")
    draw = ImageDraw.Draw(img)
    draw.ellipse([100, 180, 300, 320], fill="#DAA520", outline="#B8860B", width=5)
    draw.polygon([(190, 180), (210, 180), (200, 120)], fill="#FF4500", outline="#FFD700", width=2)
    draw.text((120, 40), "Traditional Brass Diya", fill="#FFFFFF")
    path = SAMPLE_DIR / "brass_diya.jpg"
    img.save(path, format="JPEG")
    return path


def make_madhubani_painting():
    img = Image.new("RGB", (400, 400), color="#FFF8DC")
    draw = ImageDraw.Draw(img)
    # Double line border characteristic of Madhubani
    draw.rectangle([20, 20, 380, 380], outline="#8B0000", width=5)
    draw.rectangle([30, 30, 370, 370], outline="#000000", width=2)
    # Fish / Peacock folk art motif
    draw.polygon([(100, 200), (300, 150), (300, 250)], fill="#2E8B57", outline="#000000", width=3)
    draw.ellipse([260, 180, 290, 210], fill="#FFD700", outline="#000000")
    draw.text((110, 40), "Madhubani Folk Painting", fill="#333333")
    path = SAMPLE_DIR / "madhubani_painting.jpg"
    img.save(path, format="JPEG")
    return path


def make_bandhani_saree():
    img = Image.new("RGB", (400, 400), color="#DC143C")
    draw = ImageDraw.Draw(img)
    # Bandhani tie-dye dots
    for x in range(50, 360, 40):
        for y in range(50, 360, 40):
            draw.ellipse([x, y, x+12, y+12], fill="#FFFFFF", outline="#FFD700", width=2)
    draw.text((110, 40), "Bandhani Tie-Dye Textile", fill="#FFFFFF")
    path = SAMPLE_DIR / "bandhani_saree.jpg"
    img.save(path, format="JPEG")
    return path


def make_channapatna_toy():
    img = Image.new("RGB", (400, 400), color="#F0F8FF")
    draw = ImageDraw.Draw(img)
    # Lacquer coated wooden stacking toy
    draw.rectangle([160, 260, 240, 320], fill="#0000FF", outline="#000000", width=3)
    draw.rectangle([170, 200, 230, 260], fill="#FF0000", outline="#000000", width=3)
    draw.ellipse([180, 140, 220, 200], fill="#FFD700", outline="#000000", width=3)
    draw.text((100, 40), "Channapatna Wooden Toy", fill="#333333")
    path = SAMPLE_DIR / "channapatna_toy.jpg"
    img.save(path, format="JPEG")
    return path


def make_kantha_stole():
    img = Image.new("RGB", (400, 400), color="#F5F5DC")
    draw = ImageDraw.Draw(img)
    # Kantha running stitch lines
    for y in range(80, 340, 30):
        for x in range(40, 360, 20):
            draw.line([x, y, x+10, y], fill="#4B0082", width=3)
    draw.text((110, 40), "Kantha Stitch Embroidered Stole", fill="#333333")
    path = SAMPLE_DIR / "kantha_stole.jpg"
    img.save(path, format="JPEG")
    return path


def make_blue_pottery():
    img = Image.new("RGB", (400, 400), color="#FFFFFF")
    draw = ImageDraw.Draw(img)
    # Jaipur Blue Pottery tile
    draw.rectangle([60, 60, 340, 340], fill="#00008B", outline="#008080", width=8)
    draw.ellipse([120, 120, 280, 280], fill="#4169E1", outline="#FFFFFF", width=4)
    draw.text((110, 30), "Jaipur Blue Pottery Tile", fill="#333333")
    path = SAMPLE_DIR / "blue_pottery.jpg"
    img.save(path, format="JPEG")
    return path


def generate_all_samples():
    paths = [
        make_bamboo_basket(),
        make_terracotta_vase(),
        make_brass_diya(),
        make_madhubani_painting(),
        make_bandhani_saree(),
        make_channapatna_toy(),
        make_kantha_stole(),
        make_blue_pottery(),
    ]
    print(f"Generated {len(paths)} sample artisan product images in {SAMPLE_DIR}")
    return paths


if __name__ == "__main__":
    generate_all_samples()
