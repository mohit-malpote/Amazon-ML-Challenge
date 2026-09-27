import sys, unicodedata
sys.stdout.reconfigure(encoding='utf-8')

# Build Brahmic Romanizer mapping from Unicode character names
# Standard Indic consonants & vowels map to English Latin letters
INDIC_MAP = {
    'A': 'a', 'AA': 'aa', 'I': 'i', 'II': 'ee', 'U': 'u', 'UU': 'oo',
    'R': 'r', 'RR': 'ri', 'E': 'e', 'AI': 'ai', 'O': 'o', 'AU': 'au',
    'KA': 'k', 'KHA': 'kh', 'GA': 'g', 'GHA': 'gh', 'NGA': 'ng',
    'CA': 'ch', 'CHA': 'chh', 'JA': 'j', 'JHA': 'jh', 'NYA': 'ny',
    'TTA': 't', 'TTHA': 'th', 'DDA': 'd', 'DDHA': 'dh', 'NNA': 'n',
    'TA': 't', 'THA': 'th', 'DA': 'd', 'DHA': 'dh', 'NA': 'n',
    'PA': 'p', 'PHA': 'ph', 'BA': 'b', 'BHA': 'bh', 'MA': 'm',
    'YA': 'y', 'RA': 'r', 'LA': 'l', 'LLA': 'l', 'VA': 'v', 'WA': 'w',
    'SHA': 'sh', 'SSA': 'sh', 'SA': 's', 'HA': 'h',
    'SIGN VIRAMA': '', 'VIRAMA': '', 'SIGN ANUSVARA': 'n', 'SIGN CANDRABINDU': 'n'
}

def romanize_indic(text):
    if not text:
        return ""
    out = []
    for ch in text:
        if ord(ch) < 128:
            out.append(ch)
            continue
        try:
            uname = unicodedata.name(ch)
        except ValueError:
            continue
        parts = uname.split()
        if len(parts) >= 3 and parts[0] in ['DEVANAGARI', 'BENGALI', 'GURMUKHI', 'GUJARATI', 'ORIYA', 'TAMIL', 'TELUGU', 'KANNADA', 'MALAYALAM']:
            key = ' '.join(parts[2:])
            if key in INDIC_MAP:
                out.append(INDIC_MAP[key])
            else:
                out.append(' ')
        else:
            out.append(ch)
    return ''.join(out)

test_cases = [
    'ब्लू सर्विसेज',
    'ग्रीन कंसल्टिंग प्राइवेट लिमिटेड',
    'तिरुपति आईटी प्राइवेट लिमिटेड',
    'हरि हॉस्पिटैलिटी प्राइवेट लिमिटेड',
    'ग्रीन सिस्टम्स प्राइवेट लिमिटेड',
    'সানরাইজ সফটওয়্যার লিমিটেড'
]

for t in test_cases:
    rom = romanize_indic(t)
    print(f"Original: {t} -> Romanized: {rom}")
