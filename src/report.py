from pathlib import Path


class ReportBuilder:
    """
    简单的 Markdown 报告生成器。
    """

    def __init__(self, title: str = "Report"):
        self.title = title
        self._sections = []

    def add_section(self, title: str, content: str):
        self._sections.append({"type": "text", "title": title, "content": content})
        return self

    def add_table(self, title: str, headers: list[str], rows: list[list[str]]):
        self._sections.append(
            {"type": "table", "title": title, "headers": headers, "rows": rows}
        )
        return self

    def add_image(self, title: str, image_path: Path):
        self._sections.append(
            {"type": "image", "title": title, "path": Path(image_path)}
        )
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

    def render(self, report_path: Path | None = None) -> str:
        lines = [f"# {self.title}", ""]
        for section in self._sections:
            lines.append(f"## {section['title']}")
            lines.append("")
            if section["type"] == "text":
                lines.append(section["content"])
            elif section["type"] == "table":
                lines.append(self._render_table(section["headers"], section["rows"]))
            elif section["type"] == "image":
                img_path = section["path"]
                if report_path is not None:
                    try:
                        img_path = Path(img_path).relative_to(Path(report_path).parent)
                    except ValueError:
                        img_path = Path(img_path)
                rel_path = img_path.as_posix()
                lines.append(f"![{section['title']}]({rel_path})")
            lines.append("")
        return "\n".join(lines)

    def save(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.render(report_path=path), encoding="utf-8")


def format_number(x, decimals: int = 4) -> str:
    """公共格式化函数。"""
    if isinstance(x, (int, float)):
        return f"{x:.{decimals}f}"
    return str(x)
