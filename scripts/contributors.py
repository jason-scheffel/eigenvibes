# SPDX-FileCopyrightText: 2026 Jason Scheffel <contact@jasonscheffel.com>
# SPDX-License-Identifier: BSD-3-Clause

"""Draw the README's contributors image: one crewmate per contributor.

Each crewmate has the contributor's GitHub avatar in its visor and their
username below. The avatars are stored inside the SVG, because GitHub shows
README images in a way that blocks images loaded from other sites.

Run it with ``uv run python scripts/contributors.py`` after someone new
contributes. It writes ``docs/contributors.svg``.
"""

import base64
import http.client
import json
import urllib.request
from pathlib import Path

REPOSITORY: str = "jason-scheffel/eigenvibes"
COLORS: tuple[str, ...] = (
    "#e03131",
    "#1971c2",
    "#2f9e44",
    "#f08c00",
    "#9c36b5",
    "#e64980",
    "#0c8599",
    "#fab005",
)
WIDTH: int = 150
HEIGHT: int = 200
PER_ROW: int = 6
CREWMATE: str = """\
<g transform="translate({x} {y})">
  <clipPath id="avatar{index}"><circle cx="85" cy="62" r="26"/></clipPath>
  <rect x="12" y="62" width="30" height="66" rx="12" fill="{color}"/>
  <path d="M38 64 Q38 14 85 14 Q132 14 132 64 L132 152 Q132 164 120 164 L104 164 \
Q94 164 94 154 L94 140 L76 140 L76 154 Q76 164 66 164 L50 164 Q38 164 38 152 Z" \
fill="{color}"/>
  <ellipse cx="85" cy="62" rx="38" ry="31" fill="#4dabf7"/>
  <image href="data:{avatar_type};base64,{avatar}" x="59" y="36" width="52" height="52" \
clip-path="url(#avatar{index})"/>
  <text x="{text_x}" y="188" stroke="none" fill="#8b949e">{login}</text>
</g>
"""


def download(url: str) -> tuple[bytes, str]:
    """Download a URL.

    Parameters
    ----------
    url : str
        Address to download.

    Returns
    -------
    content : bytes
        The response body.
    content_type : str
        The response's media type, such as ``image/png``.
    """
    response: http.client.HTTPResponse
    with urllib.request.urlopen(url, timeout=30) as response:
        content: bytes = response.read()
        content_type: str = response.headers.get_content_type()
    return content, content_type


if __name__ == "__main__":
    listing: bytes = download(
        f"https://api.github.com/repos/{REPOSITORY}/contributors?per_page=100"
    )[0]
    people: list[dict[str, object]] = [
        person for person in json.loads(listing) if person["type"] == "User"
    ]
    columns: int = min(len(people), PER_ROW)
    rows: int = -(-len(people) // PER_ROW)
    crewmates: list[str] = []
    index: int
    person: dict[str, object]
    for index, person in enumerate(people):
        avatar: bytes
        avatar_type: str
        avatar, avatar_type = download(f"{person['avatar_url']}&s=160")
        row: int = index // PER_ROW
        in_row: int = min(len(people) - row * PER_ROW, PER_ROW)
        crewmates.append(
            CREWMATE.format(
                x=(index % PER_ROW) * WIDTH + (columns - in_row) * WIDTH // 2,
                y=row * HEIGHT,
                index=index,
                color=COLORS[index % len(COLORS)],
                avatar_type=avatar_type,
                avatar=base64.b64encode(avatar).decode("ascii"),
                text_x=WIDTH // 2 + 10,
                login=person["login"],
            )
        )
    svg: str = (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH * columns}'
        f' {HEIGHT * rows}" width="{WIDTH * columns}" height="{HEIGHT * rows}">\n'
        '<g stroke="#1e1e1e" stroke-width="4" stroke-linejoin="round" font-size="15"'
        ' font-family="-apple-system, Segoe UI, Helvetica, Arial, sans-serif"'
        ' text-anchor="middle">\n' + "".join(crewmates) + "</g>\n</svg>\n"
    )
    (Path(__file__).resolve().parent.parent / "docs" / "contributors.svg").write_text(
        svg, encoding="utf-8"
    )
