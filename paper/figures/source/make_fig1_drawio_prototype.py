#!/usr/bin/env python3
"""Generate a draw.io prototype for Figure 1.

This keeps the Figure 1 claim fixed and changes only the diagram grammar:
behavior regimes -> search-centered row -> controlled views -> gate tests ->
separated evidence readout.
"""

from __future__ import annotations

import pathlib
import xml.etree.ElementTree as ET


ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "source" / "fig1_focalswap_architecture_drawio.drawio"


class Diagram:
    def __init__(self) -> None:
        self.next_id = 2
        self.root = ET.Element("root")
        ET.SubElement(self.root, "mxCell", {"id": "0"})
        ET.SubElement(self.root, "mxCell", {"id": "1", "parent": "0"})

    def _id(self) -> str:
        cell_id = str(self.next_id)
        self.next_id += 1
        return cell_id

    def vertex(
        self,
        value: str,
        x: float,
        y: float,
        w: float,
        h: float,
        style: str,
        parent: str = "1",
    ) -> str:
        cell_id = self._id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": value,
                "style": style,
                "vertex": "1",
                "parent": parent,
            },
        )
        ET.SubElement(
            cell,
            "mxGeometry",
            {
                "x": str(x),
                "y": str(y),
                "width": str(w),
                "height": str(h),
                "as": "geometry",
            },
        )
        return cell_id

    def edge(
        self,
        value: str,
        source: str,
        target: str,
        style: str,
        points: list[tuple[float, float]] | None = None,
    ) -> str:
        cell_id = self._id()
        cell = ET.SubElement(
            self.root,
            "mxCell",
            {
                "id": cell_id,
                "value": value,
                "style": style,
                "edge": "1",
                "parent": "1",
                "source": source,
                "target": target,
            },
        )
        geom = ET.SubElement(cell, "mxGeometry", {"relative": "1", "as": "geometry"})
        if points:
            arr = ET.SubElement(geom, "Array", {"as": "points"})
            for x, y in points:
                ET.SubElement(arr, "mxPoint", {"x": str(x), "y": str(y)})
        return cell_id


BASE_FONT = "fontFamily=Helvetica;fontSize=14;fontColor=#111827;"
SMALL_FONT = "fontFamily=Helvetica;fontSize=12;fontColor=#374151;"
TITLE = (
    "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;"
    "fontFamily=Helvetica;fontSize=22;fontStyle=1;fontColor=#111827;"
)
CONTAINER = (
    "rounded=1;whiteSpace=wrap;html=1;arcSize=6;fillColor=#F8FAFC;"
    "strokeColor=#CBD5E1;strokeWidth=1.5;"
)
BOX = (
    "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;"
    "strokeWidth=1.5;" + BASE_FONT
)
GREEN_BOX = (
    "rounded=0;whiteSpace=wrap;html=1;fillColor=#ECFDF5;strokeColor=#10B981;"
    "strokeWidth=2;" + BASE_FONT + "fontStyle=1;"
)
BLUE_BOX = (
    "rounded=0;whiteSpace=wrap;html=1;fillColor=#EFF6FF;strokeColor=#2563EB;"
    "strokeWidth=2;" + BASE_FONT + "fontStyle=1;"
)
PURPLE_BOX = (
    "rounded=0;whiteSpace=wrap;html=1;fillColor=#F5F3FF;strokeColor=#7C3AED;"
    "strokeWidth=2;" + BASE_FONT
)
RED_BOX = (
    "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFF7ED;strokeColor=#EF4444;"
    "strokeWidth=2;" + BASE_FONT
)
ORANGE_BOX = (
    "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFF7ED;strokeColor=#F97316;"
    "strokeWidth=2;" + BASE_FONT
)
GATE_ID = (
    "rounded=0;whiteSpace=wrap;html=1;fillColor=#EEF2FF;strokeColor=#2563EB;"
    "strokeWidth=2;fontFamily=Helvetica;fontSize=16;fontStyle=1;fontColor=#111827;"
)
GATE_ID_ORANGE = GATE_ID.replace("#2563EB", "#F97316")
GATE_DESC = (
    "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#2563EB;"
    "strokeWidth=2;fontFamily=Helvetica;fontSize=12;fontColor=#111827;"
)
GATE_DESC_ORANGE = GATE_DESC.replace("#2563EB", "#F97316")
READOUT_BOX = (
    "rounded=0;whiteSpace=wrap;html=1;fillColor=#FFFFFF;strokeColor=#CBD5E1;"
    "strokeWidth=1.5;fontFamily=Helvetica;fontSize=13;fontColor=#111827;"
)
ANCHOR = "ellipse;html=1;strokeColor=none;fillColor=none;opacity=0;"
ARROW = (
    "endArrow=block;html=1;rounded=0;strokeWidth=2;strokeColor=#64748B;"
    "fontFamily=Helvetica;fontSize=12;fontColor=#475569;labelBackgroundColor=#FFFFFF;"
)
DASHED = ARROW + "dashed=1;dashPattern=6 6;"


def build() -> ET.ElementTree:
    d = Diagram()

    # Containers.
    d.vertex("", 24, 30, 230, 330, CONTAINER)
    d.vertex("Regimes", 44, 50, 190, 32, TITLE)
    d.vertex("", 300, 30, 355, 330, CONTAINER)
    d.vertex("Search-centered row", 320, 50, 310, 32, TITLE)
    d.vertex("", 700, 30, 240, 330, CONTAINER)
    d.vertex("Controlled views", 720, 50, 200, 32, TITLE)
    d.vertex("", 990, 30, 230, 330, CONTAINER)
    d.vertex("Evidence tests", 1010, 50, 190, 32, TITLE)

    # Regimes.
    regime_k = d.vertex("KuaiSAR\nanonymous behavior\nS->R logs", 54, 110, 170, 88, GREEN_BOX)
    regime_q = d.vertex("Qilin\nreal-query\nlexical stress", 54, 244, 170, 88, BLUE_BOX)
    d.vertex("behavior-level settings", 49, 334, 180, 18, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;" + SMALL_FONT)

    # Row.
    pre = d.vertex("pre-search\nhistory", 318, 142, 92, 76, BOX)
    focal = d.vertex("focal\nsearch", 431, 142, 92, 76, GREEN_BOX)
    post = d.vertex("post-search\ntarget", 544, 142, 92, 76, BOX)
    cand = d.vertex("fixed target + candidate set", 318, 260, 318, 68, BOX)
    d.vertex("same user, ordered time window", 341, 333, 274, 20, "text;html=1;strokeColor=none;fillColor=none;align=center;verticalAlign=middle;" + SMALL_FONT)

    d.edge("", pre, focal, ARROW)
    d.edge("", focal, post, ARROW)
    d.edge("", focal, cand, DASHED + "exitX=0.25;exitY=1;entryX=0.25;entryY=0;")
    d.edge("", focal, cand, DASHED + "exitX=0.75;exitY=1;entryX=0.75;entryY=0;")

    # Controlled views.
    views = [
        ("Full-SAR", PURPLE_BOX),
        ("History-only", BOX),
        ("Search-only", BOX),
        ("Search-masked", BOX),
        ("Counterfactual-search", RED_BOX),
    ]
    view_ids = []
    for i, (label, style) in enumerate(views):
        view_ids.append(d.vertex(label, 735, 94 + 52 * i, 170, 42, style))

    # Gates.
    gate_rows = [
        ("G1", "outcome\navailability", GATE_ID, GATE_DESC),
        ("G2", "calibrated\nopportunity", GATE_ID, GATE_DESC),
        ("G3", "focal-search\nidentifiability", GATE_ID_ORANGE, GATE_DESC_ORANGE),
        ("G4", "recovery\nprobe", GATE_ID_ORANGE, GATE_DESC_ORANGE),
    ]
    gate_ids = []
    for i, (gate, desc, gate_style, desc_style) in enumerate(gate_rows):
        y = 94 + 70 * i
        gate_ids.append(d.vertex(gate, 1018, y, 56, 42, gate_style))
        d.vertex(desc, 1082, y, 118, 42, desc_style)

    # Cross-container arrows.
    label_style = "text;html=1;strokeColor=none;fillColor=#FFFFFF;align=center;verticalAlign=middle;" + SMALL_FONT
    d.vertex("select\nrows", 252, 164, 48, 34, label_style)
    d.vertex("instantiate\nviews", 654, 164, 52, 34, label_style)
    d.vertex("score\ncontrasts", 940, 164, 48, 34, label_style)
    reg_out = d.vertex("", 254, 180, 1, 1, ANCHOR)
    row_in = d.vertex("", 300, 180, 1, 1, ANCHOR)
    row_out = d.vertex("", 655, 180, 1, 1, ANCHOR)
    views_in = d.vertex("", 700, 180, 1, 1, ANCHOR)
    views_out = d.vertex("", 940, 180, 1, 1, ANCHOR)
    gates_in = d.vertex("", 990, 180, 1, 1, ANCHOR)
    d.edge("", reg_out, row_in, ARROW)
    d.edge("", row_out, views_in, ARROW)
    d.edge("", views_out, gates_in, ARROW)

    # Readout.
    readout = d.vertex("", 24, 420, 1196, 98, CONTAINER)
    d.vertex("Separated evidence readout", 160, 438, 300, 30, "text;html=1;strokeColor=none;fillColor=none;align=left;verticalAlign=middle;fontFamily=Helvetica;fontSize=20;fontStyle=1;fontColor=#111827;")
    readouts = [
        "ranking gain",
        "calibrated opportunity",
        "named-control response",
        "focal specificity",
        "input-side recovery",
    ]
    for i, label in enumerate(readouts):
        d.vertex(label, 44 + 230 * i, 476, 210, 34, READOUT_BOX)

    d.edge("", gate_ids[-1], readout, ARROW + "curved=1;", [(1115, 395), (640, 395)])

    model = ET.Element(
        "mxGraphModel",
        {
            "dx": "1422",
            "dy": "794",
            "grid": "1",
            "gridSize": "10",
            "guides": "1",
            "tooltips": "1",
            "connect": "1",
            "arrows": "1",
            "fold": "1",
            "page": "1",
            "pageScale": "1",
            "pageWidth": "1260",
            "pageHeight": "550",
            "math": "0",
            "shadow": "0",
        },
    )
    model.append(d.root)
    diagram = ET.Element("diagram", {"id": "fig1", "name": "Figure 1"})
    diagram.append(model)
    mxfile = ET.Element(
        "mxfile",
        {
            "host": "drawio",
            "version": "30.0.4",
            "type": "device",
        },
    )
    mxfile.append(diagram)
    return ET.ElementTree(mxfile)


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    tree = build()
    ET.indent(tree, space="  ")
    tree.write(OUT, encoding="UTF-8", xml_declaration=True)
    print(OUT)


if __name__ == "__main__":
    main()
