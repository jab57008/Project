from pathlib import Path


class ReportBuilder:

    def __init__(self, title: str = "Report"):
        self.title = title
        self._sections = []

    def add_table(self, title: str, headers: list[str], rows: list[list[str]]):
        self._sections.append({"title": title, "headers": headers, "rows": rows})
        return self

    def _fmt_number(self, x) -> str:
        if isinstance(x, (int, float)):
            if isinstance(x, float) and abs(x) < 1e-4 or abs(x) > 1e4:
                return f"{x:.4e}"
            return f"{x:.4f}"
        return str(x)

    def _render_table(self, headers: list, rows: list) -> str:
        lines = ["| " + " | ".join(headers) + " |"]
        lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in rows:
            formatted = [self._fmt_number(cell) for cell in row]
            lines.append("| " + " | ".join(formatted) + " |")
        return "\n".join(lines)

    def render(self) -> str:
        lines = [f"# {self.title}", ""]
        for section in self._sections:
            lines.append(f"## {section['title']}")
            lines.append("")
            lines.append(self._render_table(section["headers"], section["rows"]))
            lines.append("")
        return "\n".join(lines)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(), encoding="utf-8")
