from PIL import Image
import webcolors

# Optional: remap uncommon names to simpler ones
COLOR_REMAP = {
    'springgreen': 'green',
    'lime': 'green',
    'aqua': 'blue',
    'cyan': 'blue',
    'fuchsia': 'pink',
    'magenta': 'pink',
}

def simplify_color_name(name):
    return COLOR_REMAP.get(name, name)

def closest_color(requested_color):
    min_colors = {}
    for hex_val, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(hex_val)
        diff = sum((c1 - c2) ** 2 for c1, c2 in zip((r_c, g_c, b_c), requested_color))
        min_colors[diff] = name
    return min_colors[min(min_colors)]

def get_color_name(rgb):
    try:
        return webcolors.rgb_to_name(rgb)
    except ValueError:
        return closest_color(rgb)

def extract_unique_clean_rows(image_path):
    img = Image.open(image_path).convert('RGBA')
    width, height = img.size
    pixels = img.load()

    clean_rows = []
    prev_row = None

    for y in range(height):
        row = []
        for x in range(width):
            r, g, b, a = pixels[x, y]
            if a == 0:
                continue
            raw_color = get_color_name((r, g, b))
            simple_color = simplify_color_name(raw_color)
            row.append(simple_color)

        if not row:
            continue

        # Remove consecutive duplicates within the row
        compressed_row = [row[0]]
        for color in row[1:]:
            if color != compressed_row[-1]:
                compressed_row.append(color)

        if prev_row is None or compressed_row != prev_row:
            clean_rows.append(compressed_row)
            prev_row = compressed_row.copy()

    return clean_rows

# Example usage
color_rows = extract_unique_clean_rows("New Piskel.png")
for row in color_rows:
    print(row)
