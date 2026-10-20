#!/usr/bin/env python3
"""Generates the low-poly Roblox deer.

The deer is built from Blocks and WedgeParts in a neutral standing pose
(ground at Y = 0, facing -Z like every Roblox model). Each animatable body
part is a "bone" part joined to its parent with a Motor6D; the smaller pieces
that give it its shape are welded to that bone.

Outputs (next to this file):
  Deer.rbxmx      drag into Roblox Studio
  BuildDeer.lua   paste into the Studio command bar to build the same model
  deer.json       raw part data for previewing outside Studio (only with --json)
"""

import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# Colors sampled from the reference image.
TAN = (186, 134, 88)
TAN_DARK = (158, 108, 68)
CREAM = (234, 214, 180)
WHITE = (244, 238, 226)
INNER_EAR = (226, 186, 156)
HOOF = (46, 37, 32)
NOSE = (34, 29, 27)
EYE = (22, 18, 16)
ANTLER = (226, 221, 210)

I3 = np.eye(3)


def v(*a):
    return np.array(a, dtype=float)


def unit(a):
    a = np.asarray(a, dtype=float)
    return a / np.linalg.norm(a)


def frame_y(y_axis, x_hint=(1, 0, 0)):
    """Rotation whose Y axis points along y_axis, X as close to x_hint as possible."""
    y = unit(y_axis)
    x = np.asarray(x_hint, dtype=float)
    x = unit(x - y * np.dot(x, y))
    z = np.cross(x, y)
    return np.column_stack([x, y, z])


def frame_z(z_axis, y_hint=(0, 1, 0)):
    """Rotation whose Z axis points along z_axis, Y as close to y_hint as possible."""
    z = unit(z_axis)
    y = np.asarray(y_hint, dtype=float)
    y = unit(y - z * np.dot(y, z))
    x = np.cross(y, z)
    return np.column_stack([x, y, z])


def rot_x(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(deg):
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Piece:
    def __init__(self, name, cls, size, pos, rot, color, bone, cast_shadow=True):
        self.name = name
        self.cls = cls
        self.size = np.asarray(size, dtype=float)
        self.pos = np.asarray(pos, dtype=float)
        self.rot = np.asarray(rot, dtype=float)
        self.color = color
        self.bone = bone
        self.cast_shadow = cast_shadow
        self.transparency = 0.0
        self.children = []


class Rig:
    def __init__(self):
        self.bones = {}
        self.order = []
        self.joints = []  # (motor name, part0, part1, pivot)
        self.counts = {}

    def bone(self, name, parent, pivot, size, pos, rot, color, joint=None, cls="Part"):
        p = Piece(name, cls, size, pos, rot, color, None)
        self.bones[name] = p
        self.order.append(p)
        if parent:
            self.joints.append((joint or name, parent, name, np.asarray(pivot, dtype=float)))
        return p

    def _name(self, bone, tag):
        key = (bone, tag)
        self.counts[key] = self.counts.get(key, 0) + 1
        n = self.counts[key]
        return tag if n == 1 else f"{tag}{n}"

    def box(self, bone, tag, size, pos, rot=I3, color=TAN, shadow=True):
        p = Piece(self._name(bone, tag), "Part", size, pos, rot, color, bone, shadow)
        self.bones[bone].children.append(p)
        return p

    def wedge(self, bone, tag, b, a, c, width, color=TAN, shadow=True):
        """Right-angled wedge: right angle at b, a toward the low tip, c up the tall face."""
        b, a, c = (np.asarray(t, dtype=float) for t in (b, a, c))
        z = unit(b - a)
        cy = c - b
        cy = cy - z * np.dot(cy, z)
        y = unit(cy)
        x = np.cross(y, z)
        size = (width, np.linalg.norm(cy), np.linalg.norm(b - a))
        pos = b + (a - b) / 2 + cy / 2
        p = Piece(self._name(bone, tag), "WedgePart", size, pos, np.column_stack([x, y, z]), color, bone, shadow)
        self.bones[bone].children.append(p)
        return p

    def seg(self, bone, tag, top, bottom, w, d, color=TAN, x_hint=(1, 0, 0), extend=0.0, shadow=True):
        """Block running from top to bottom (its +Y end at top)."""
        top, bottom = np.asarray(top, float), np.asarray(bottom, float)
        r = frame_y(top - bottom, x_hint)
        length = np.linalg.norm(top - bottom) + 2 * extend
        return self.box(bone, tag, (w, length, d), (top + bottom) / 2, r, color, shadow)

    def octo(self, bone, tag, center, rot, w, h, length, bevel, color=TAN, edges=(1, 1, 1, 1)):
        """Prism with chamfered long edges: a low-poly 'rounded' block.

        Built in the rot frame: X = width, Y = height, Z = length.
        edges = which chamfers to fill (top-right, top-left, bottom-right, bottom-left).
        """
        center = np.asarray(center, float)
        rot = np.asarray(rot, float)
        self.box(bone, tag, (w, h - 2 * bevel, length), center, rot, color)
        self.box(bone, tag, (w - 2 * bevel, h, length), center, rot, color)
        hw, hh = w / 2, h / 2
        corners = [(1, 1), (-1, 1), (1, -1), (-1, -1)]
        for (sx, sy), on in zip(corners, edges):
            if not on:
                continue
            bl = v(sx * (hw - bevel), sy * (hh - bevel), 0)
            al = v(sx * hw, sy * (hh - bevel), 0)
            cl = v(sx * (hw - bevel), sy * hh, 0)
            self.wedge(bone, tag, center + rot @ bl, center + rot @ al, center + rot @ cl, length, color)


def build():
    r = Rig()
    pitch = lambda deg: rot_x(deg)

    # ---- Root ---------------------------------------------------------------
    hrp = r.bone("HumanoidRootPart", None, None, (1.4, 1.4, 3.8), (0, 3.25, 0), I3, (163, 162, 165))
    hrp.transparency = 1.0

    # ---- Body ---------------------------------------------------------------
    # deep chest, belly tucking up toward the hips
    r.bone("Hips", "HumanoidRootPart", (0, 3.25, 0), (1.3, 1.0, 1.9), (0, 3.38, 0.95), I3, TAN, joint="Root")
    r.octo("Hips", "Barrel", (0, 3.38, 0.95), I3, 1.44, 1.36, 1.9, 0.36)
    r.octo("Hips", "Rump", (0, 3.42, 1.92), pitch(-30), 1.24, 1.06, 0.62, 0.3)
    r.box("Hips", "Belly", (0.9, 0.14, 1.6), (0, 2.72, 0.75), pitch(-4), CREAM)
    r.box("Hips", "Saddle", (0.68, 0.08, 1.8), (0, 4.06, 0.9), I3, TAN_DARK)

    r.bone("Chest", "Hips", (0, 3.25, 0), (1.3, 1.0, 1.9), (0, 3.25, -0.95), I3, TAN, joint="Waist")
    r.octo("Chest", "Barrel", (0, 3.25, -0.92), I3, 1.46, 1.6, 1.9, 0.4)
    r.octo("Chest", "Brisket", (0, 3.05, -1.78), pitch(25), 1.12, 1.25, 0.6, 0.32)
    r.box("Chest", "Bib", (0.62, 0.8, 0.1), (0, 2.9, -2.08), pitch(25), WHITE)
    r.box("Chest", "Belly", (0.92, 0.14, 1.6), (0, 2.47, -0.75), I3, CREAM)
    r.box("Chest", "Saddle", (0.72, 0.08, 1.5), (0, 4.06, -0.85), I3, TAN_DARK)

    # ---- Neck and head --------------------------------------------------------
    neck_base = v(0, 3.75, -1.55)
    neck_top = v(0, 5.0, -2.25)
    neck_rot = frame_y(neck_top - neck_base)  # -Z of this frame is the throat side
    neck_len = np.linalg.norm(neck_top - neck_base)
    neck_mid = (neck_base + neck_top) / 2
    r.bone("Neck", "Chest", neck_base, (0.7, neck_len, 0.8), neck_mid, neck_rot, TAN, joint="NeckBase")
    neck_frame = neck_rot @ rot_x(90)  # prism length along the neck
    r.octo("Neck", "Neck", neck_mid, neck_frame, 0.82, 0.98, neck_len + 0.3, 0.26)
    r.octo("Neck", "Base", neck_base + v(0, -0.12, 0.08), neck_frame, 1.1, 1.24, 0.9, 0.32)
    r.box("Neck", "Throat", (0.4, neck_len + 0.1, 0.14), neck_mid + neck_rot @ v(0, -0.05, -0.45), neck_rot, WHITE)

    head_rot = pitch(-25)  # head points forward-down
    skull = v(0, 5.12, -2.55)
    hf = lambda p: skull + head_rot @ v(*p)  # head-local point
    r.bone("Head", "Neck", neck_top, (0.7, 0.66, 0.8), skull, head_rot, TAN, joint="Neck")
    r.octo("Head", "Skull", skull, head_rot, 0.82, 0.78, 0.95, 0.22)
    r.octo("Head", "Muzzle", hf((0, -0.1, -0.75)), head_rot, 0.56, 0.5, 0.7, 0.15)
    r.wedge("Head", "Bridge", hf((0, 0.15, -0.47)), hf((0, 0.15, -1.04)), hf((0, 0.39, -0.47)), 0.52)
    r.box("Head", "Nose", (0.42, 0.3, 0.14), hf((0, -0.05, -1.1)), head_rot, NOSE)
    r.box("Head", "Chin", (0.44, 0.12, 0.62), hf((0, -0.36, -0.7)), head_rot, CREAM)
    r.box("Head", "Jaw", (0.66, 0.24, 0.6), hf((0, -0.32, 0.12)), head_rot, CREAM)
    for sx in (1, -1):
        r.box("Head", "Eye", (0.06, 0.15, 0.17), hf((sx * 0.42, 0.1, -0.22)), head_rot, EYE, shadow=False)

    # big leaf-shaped ears sticking out sideways
    for side, sx in (("Right", 1), ("Left", -1)):
        base = hf((sx * 0.34, 0.3, 0.26))
        tip = base + v(sx * 0.82, 0.32, 0.15)
        ear_len = np.linalg.norm(tip - base)
        ear_rot = frame_y(tip - base, (0, 1, 0))  # broad face looking forward
        el = lambda x, y, z: base + ear_rot @ v(x, y, z)
        name = f"{side}Ear"
        r.bone(name, "Head", base, (0.42, ear_len, 0.08), (base + tip) / 2, ear_rot, TAN)
        r.box(name, "Inner", (0.28, ear_len * 0.75, 0.04), base + (tip - base) * 0.5 + v(0, 0, -0.045), ear_rot, INNER_EAR, shadow=False)
        for wx in (0.21, -0.21):  # pointed tip
            r.wedge(name, "Tip", el(0, ear_len / 2, 0), el(wx, ear_len / 2, 0), el(0, ear_len / 2 + 0.3, 0), 0.08)

    # antlers: main beam sweeps up and out, curling forward at the top, five points each
    for sx in (1, -1):
        root = hf((sx * 0.22, 0.36, 0.12))
        p = lambda x, y, z: root + v(sx * x, y, z)
        beam = [p(0, 0, 0), p(0.3, 0.45, 0.2), p(0.62, 0.95, 0.25), p(0.85, 1.5, 0.1), p(0.9, 1.95, -0.2), p(0.8, 2.3, -0.45)]
        r.box("Head", "Burr", (0.26, 0.1, 0.26), beam[0], I3, ANTLER)
        widths = [0.18, 0.16, 0.14, 0.12, 0.1]
        for i in range(5):
            r.seg("Head", "Beam", beam[i + 1], beam[i], widths[i], widths[i], ANTLER, extend=0.05)
        tines = [
            (p(0.12, 0.2, 0.1), p(0.2, 0.55, -0.42), 0.12),  # brow tine, points forward
            (p(0.3, 0.45, 0.2), p(0.32, 1.2, -0.22), 0.12),
            (p(0.62, 0.95, 0.25), p(0.68, 1.72, 0.0), 0.11),
            (p(0.85, 1.5, 0.1), p(1.25, 2.02, 0.22), 0.1),
            (p(0.9, 1.95, -0.2), p(1.08, 2.4, -0.12), 0.08),
        ]
        for start, end, w in tines:
            r.seg("Head", "Tine", end, start, w, w, ANTLER, extend=0.04)

    # ---- Tail ---------------------------------------------------------------
    tail_base = v(0, 3.72, 2.08)
    tail_tip = v(0, 3.25, 2.42)
    tail_rot = frame_y(tail_base - tail_tip)
    r.bone("Tail", "Hips", tail_base, (0.34, np.linalg.norm(tail_tip - tail_base) + 0.1, 0.16), (tail_base + tail_tip) / 2, tail_rot, TAN)
    r.box("Tail", "Underside", (0.3, 0.5, 0.08), (tail_base + tail_tip) / 2 + tail_rot @ v(0, -0.02, 0.1), tail_rot, WHITE)

    # ---- Front legs -----------------------------------------------------------
    for side, sx in (("Right", 1), ("Left", -1)):
        x = sx * 0.44
        shoulder = v(x, 3.15, -1.35)
        elbow = v(x, 2.3, -1.28)
        knee = v(x, 1.28, -1.36)
        fetlock = v(x, 0.3, -1.4)
        up, lo, hoof = f"{side}FrontUpperLeg", f"{side}FrontLowerLeg", f"{side}FrontHoof"
        r.bone(up, "Chest", shoulder, (0.28, np.linalg.norm(elbow - knee), 0.32), (elbow + knee) / 2, frame_y(elbow - knee), TAN, joint=f"{side}Shoulder")
        r.seg(up, "Shoulder", shoulder + v(0, 0.25, 0), elbow, 0.44, 0.66)
        r.seg(up, "Forearm", elbow + v(0, 0.2, 0), knee, 0.26, 0.36, extend=0.02)
        r.box(up, "Back", (0.2, 0.7, 0.1), elbow + v(0, -0.35, 0.16), I3, CREAM)
        r.bone(lo, up, knee, (0.18, np.linalg.norm(knee - fetlock), 0.2), (knee + fetlock) / 2, frame_y(knee - fetlock), TAN, joint=f"{side}Knee")
        r.box(lo, "Knee", (0.25, 0.2, 0.27), knee, I3, TAN)
        r.box(lo, "Fetlock", (0.21, 0.16, 0.24), fetlock + v(0, 0.02, 0.02), I3, TAN)
        r.bone(hoof, lo, fetlock, (0.22, 0.3, 0.2), fetlock + v(0, -0.15, 0.03), I3, HOOF, joint=f"{side}FrontFetlock")
        r.wedge(hoof, "Toe", fetlock + v(0, -0.3, -0.07), fetlock + v(0, -0.3, -0.22), fetlock + v(0, -0.02, -0.07), 0.22, HOOF)

    # ---- Hind legs ------------------------------------------------------------
    for side, sx in (("Right", 1), ("Left", -1)):
        x = sx * 0.46
        hip = v(x, 3.35, 1.4)
        stifle = v(x, 2.35, 1.12)
        hock = v(x, 1.38, 1.72)
        fetlock = v(x, 0.3, 1.6)
        th, sh, ca, hoof = f"{side}HindThigh", f"{side}HindShin", f"{side}HindCannon", f"{side}HindHoof"
        r.bone(th, "Hips", hip, (0.5, np.linalg.norm(hip - stifle), 0.8), (hip + stifle) / 2, frame_y(hip - stifle), TAN, joint=f"{side}Hip")
        r.octo(th, "Ham", (hip + stifle) / 2 + v(0, 0.1, 0.12), frame_y(hip - stifle) @ rot_x(90), 0.56, 1.0, 1.35, 0.15)
        r.seg(th, "Rear", hip + v(sx * -0.02, 0.0, 0.5), stifle + v(sx * -0.02, 0.2, 0.42), 0.36, 0.12, CREAM)
        r.bone(sh, th, stifle, (0.3, np.linalg.norm(stifle - hock), 0.42), (stifle + hock) / 2, frame_y(stifle - hock), TAN, joint=f"{side}Stifle")
        r.seg(sh, "Gaskin", stifle + v(0, 0.15, -0.05), hock + v(0, 0.2, -0.05), 0.36, 0.5, extend=0.0)
        r.bone(ca, sh, hock, (0.19, np.linalg.norm(hock - fetlock), 0.22), (hock + fetlock) / 2, frame_y(hock - fetlock), TAN, joint=f"{side}Hock")
        r.box(ca, "Hock", (0.23, 0.32, 0.32), hock + v(0, 0.02, 0.02), rot_x(-15), TAN)
        r.box(ca, "Fetlock", (0.21, 0.16, 0.24), fetlock + v(0, 0.02, 0.02), I3, TAN)
        r.bone(hoof, ca, fetlock, (0.22, 0.3, 0.2), fetlock + v(0, -0.15, 0.03), I3, HOOF, joint=f"{side}HindFetlock")
        r.wedge(hoof, "Toe", fetlock + v(0, -0.3, -0.07), fetlock + v(0, -0.3, -0.22), fetlock + v(0, -0.02, -0.07), 0.22, HOOF)

    # bones that sit fully inside their own shape pieces only exist for the rig:
    # hide them so they cost no draw calls (they still take raycasts)
    for name, b in r.bones.items():
        if b.children and (name in ("Hips", "Chest", "Neck", "Head") or name.endswith(("UpperLeg", "Thigh", "Shin"))):
            b.transparency = 1.0

    return r


# ---------------------------------------------------------------------------
# Output helpers


def cf_world(piece):
    return piece.pos, piece.rot


def rel(p0_pos, p0_rot, p1_pos, p1_rot):
    """p0:Inverse() * p1"""
    rot = p0_rot.T @ p1_rot
    pos = p0_rot.T @ (p1_pos - p0_pos)
    return pos, rot


def fmt(x):
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return "0" if s in ("-0", "") else s


def cf_numbers(pos, rot):
    return [pos[0], pos[1], pos[2]] + [rot[i][j] for i in range(3) for j in range(3)]


def all_pieces(r):
    for b in r.order:
        yield b
        yield from b.children


def joint_frames(r, j):
    name, p0, p1, pivot = j
    a, b = r.bones[p0], r.bones[p1]
    c0 = rel(a.pos, a.rot, pivot, I3)
    c1 = rel(b.pos, b.rot, pivot, I3)
    return c0, c1


def write_json(r, path):
    out = []
    for p in all_pieces(r):
        out.append({
            "name": p.name, "cls": p.cls, "size": p.size.tolist(), "pos": p.pos.tolist(),
            "rot": p.rot.tolist(), "color": p.color, "transparency": p.transparency,
            "bone": p.bone,
        })
    joints = [{"name": j[0], "part0": j[1], "part1": j[2], "pivot": j[3].tolist()} for j in r.joints]
    with open(path, "w") as f:
        json.dump({"parts": out, "joints": joints}, f)


class Xml:
    def __init__(self):
        self.lines = []
        self.ref_n = 0

    def ref(self):
        self.ref_n += 1
        return f"RBX{self.ref_n:04d}"


def xml_cf(tag, name, pos, rot):
    n = cf_numbers(pos, rot)
    keys = ["X", "Y", "Z", "R00", "R01", "R02", "R10", "R11", "R12", "R20", "R21", "R22"]
    inner = "".join(f"<{k}>{fmt(val)}</{k}>" for k, val in zip(keys, n))
    return f'<{tag} name="{name}">{inner}</{tag}>'


def write_rbxmx(r, path):
    refs = {}
    for p in all_pieces(r):
        refs[id(p)] = None
    counter = [0]

    def nref():
        counter[0] += 1
        return f"RBX{counter[0]:04d}"

    for p in all_pieces(r):
        refs[id(p)] = nref()
    bone_ref = {name: refs[id(b)] for name, b in r.bones.items()}

    out = []
    w = out.append

    def part_xml(p, indent, is_bone):
        pad = "  " * indent
        w(f'{pad}<Item class="{p.cls}" referent="{refs[id(p)]}">')
        w(f"{pad}  <Properties>")
        props = [
            f'<string name="Name">{p.name}</string>',
            xml_cf("CoordinateFrame", "CFrame", p.pos, p.rot),
            f'<Vector3 name="size"><X>{fmt(p.size[0])}</X><Y>{fmt(p.size[1])}</Y><Z>{fmt(p.size[2])}</Z></Vector3>',
            f'<Color3uint8 name="Color3uint8">{0xFF000000 | (p.color[0] << 16) | (p.color[1] << 8) | p.color[2]}</Color3uint8>',
            '<token name="Material">272</token>',
            f'<float name="Transparency">{fmt(p.transparency)}</float>',
            '<bool name="Anchored">false</bool>',
            f'<bool name="CanCollide">{"true" if p.name == "HumanoidRootPart" else "false"}</bool>',
            f'<bool name="CanTouch">{"true" if is_bone else "false"}</bool>',
            '<bool name="CanQuery">true</bool>',
            f'<bool name="CastShadow">{"true" if p.cast_shadow and p.transparency < 1 else "false"}</bool>',
            f'<bool name="Massless">{"false" if p.name == "HumanoidRootPart" else "true"}</bool>',
            '<token name="TopSurface">0</token>',
            '<token name="BottomSurface">0</token>',
        ]
        if p.cls == "Part":
            props.append('<token name="shape">1</token>')
        for line in props:
            w(f"{pad}    {line}")
        w(f"{pad}  </Properties>")

    w('<roblox xmlns:xmime="http://www.w3.org/2005/05/xmlmime" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
      'xsi:noNamespaceSchemaLocation="http://www.roblox.com/roblox.xsd" version="4">')
    w(f'  <Item class="Model" referent="{nref()}">')
    w("    <Properties>")
    w('      <string name="Name">Deer</string>')
    w(f'      <Ref name="PrimaryPart">{bone_ref["HumanoidRootPart"]}</Ref>')
    pivot = xml_cf("CFrame", "", v(0, 0, 0), I3).replace(' name=""', "")
    w(f'      <OptionalCoordinateFrame name="WorldPivotData">{pivot}</OptionalCoordinateFrame>')
    w("    </Properties>")

    # Humanoid + Animator make it movable (Humanoid:MoveTo) and animatable
    hip_height = r.bones["HumanoidRootPart"].pos[1] - r.bones["HumanoidRootPart"].size[1] / 2
    w(f'    <Item class="Humanoid" referent="{nref()}">')
    w("      <Properties>")
    for line in [
        '<string name="Name">Humanoid</string>',
        '<token name="RigType">1</token>',
        f'<float name="HipHeight">{fmt(hip_height)}</float>',
        '<bool name="AutomaticScalingEnabled">false</bool>',
        '<bool name="RequiresNeck">false</bool>',
        '<bool name="BreakJointsOnDeath">false</bool>',
        '<token name="DisplayDistanceType">2</token>',
        '<token name="HealthDisplayType">2</token>',
        '<float name="WalkSpeed">14</float>',
    ]:
        w(f"        {line}")
    w("      </Properties>")
    w(f'      <Item class="Animator" referent="{nref()}"><Properties><string name="Name">Animator</string></Properties></Item>')
    w("    </Item>")

    joint_by_child = {j[2]: j for j in r.joints}
    for b in r.order:
        part_xml(b, 2, True)
        if b.name in joint_by_child:
            j = joint_by_child[b.name]
            (c0p, c0r), (c1p, c1r) = joint_frames(r, j)
            w(f'      <Item class="Motor6D" referent="{nref()}">')
            w("        <Properties>")
            w(f'          <string name="Name">{j[0]}</string>')
            w(f'          <Ref name="Part0">{bone_ref[j[1]]}</Ref>')
            w(f'          <Ref name="Part1">{bone_ref[j[2]]}</Ref>')
            w(f"          {xml_cf('CoordinateFrame', 'C0', c0p, c0r)}")
            w(f"          {xml_cf('CoordinateFrame', 'C1', c1p, c1r)}")
            w("        </Properties>")
            w("      </Item>")
        for c in b.children:
            part_xml(c, 3, False)
            c0p, c0r = rel(b.pos, b.rot, c.pos, c.rot)
            w(f'        <Item class="Weld" referent="{nref()}">')
            w("          <Properties>")
            w('            <string name="Name">Weld</string>')
            w(f'            <Ref name="Part0">{refs[id(b)]}</Ref>')
            w(f'            <Ref name="Part1">{refs[id(c)]}</Ref>')
            w(f"            {xml_cf('CoordinateFrame', 'C0', c0p, c0r)}")
            w(f"            {xml_cf('CoordinateFrame', 'C1', v(0, 0, 0), I3)}")
            w("          </Properties>")
            w("        </Item>")
            w("      </Item>")
        w("    </Item>")
    w("  </Item>")
    w("</roblox>")
    with open(path, "w") as f:
        f.write("\n".join(out) + "\n")


def lua_cf(pos, rot):
    return "CFrame.new(" + ", ".join(fmt(x) for x in cf_numbers(pos, rot)) + ")"


def write_lua(r, path):
    L = []
    w = L.append
    hrp = r.bones["HumanoidRootPart"]
    w("--[[")
    w("  Low-poly deer, generated by build_deer.py. Do not edit by hand.")
    w("")
    w("  Paste into the Roblox Studio command bar (or run as a plugin/Script) to")
    w("  build a rigged \"Deer\" model in Workspace, standing at the camera focus.")
    w("  Every body part is joined with a Motor6D, so the model opens in the")
    w("  Animation Editor and walks with Humanoid:MoveTo().")
    w("]]")
    w("")
    w("-- { name, class, parent bone (nil = bone itself), size, CFrame, color, transparency, castShadow }")
    w("local PARTS = {")
    for b in r.order:
        for p in [b] + b.children:
            parent = "nil" if p is b else f'"{b.name}"'
            w(f'\t{{"{p.name}", "{p.cls}", {parent}, Vector3.new({fmt(p.size[0])}, {fmt(p.size[1])}, {fmt(p.size[2])}), '
              f"{lua_cf(p.pos, p.rot)}, Color3.fromRGB({p.color[0]}, {p.color[1]}, {p.color[2]}), "
              f"{fmt(p.transparency)}, {'true' if p.cast_shadow and p.transparency < 1 else 'false'}}},")
    w("}")
    w("")
    w("-- { motor name, part0, part1, pivot (world position in the rest pose) }")
    w("local JOINTS = {")
    for name, p0, p1, pivot in r.joints:
        w(f'\t{{"{name}", "{p0}", "{p1}", Vector3.new({fmt(pivot[0])}, {fmt(pivot[1])}, {fmt(pivot[2])})}},')
    w("}")
    w("")
    w(f"local HIP_HEIGHT = {fmt(hrp.pos[1] - hrp.size[1] / 2)}")
    w("""
local model = Instance.new("Model")
model.Name = "Deer"

local bones = {}
for _, info in ipairs(PARTS) do
	local name, class, boneName, size, cframe, color, transparency, castShadow = table.unpack(info)
	local part = Instance.new(class)
	part.Name = name
	if class == "Part" then
		part.Shape = Enum.PartType.Block
	end
	part.Size = size
	part.CFrame = cframe
	part.Color = color
	part.Material = Enum.Material.SmoothPlastic
	part.TopSurface = Enum.SurfaceType.Smooth
	part.BottomSurface = Enum.SurfaceType.Smooth
	part.Transparency = transparency
	part.CastShadow = castShadow
	part.Anchored = false
	part.CanCollide = name == "HumanoidRootPart"
	part.CanTouch = boneName == nil
	part.CanQuery = true
	part.Massless = name ~= "HumanoidRootPart"
	if boneName then
		local bone = bones[boneName]
		local weld = Instance.new("Weld")
		weld.Part0 = bone
		weld.Part1 = part
		weld.C0 = bone.CFrame:ToObjectSpace(cframe)
		weld.Parent = part
		part.Parent = bone
	else
		bones[name] = part
		part.Parent = model
	end
end

for _, info in ipairs(JOINTS) do
	local name, part0Name, part1Name, pivot = table.unpack(info)
	local part0, part1 = bones[part0Name], bones[part1Name]
	local motor = Instance.new("Motor6D")
	motor.Name = name
	motor.Part0 = part0
	motor.Part1 = part1
	motor.C0 = part0.CFrame:Inverse() * CFrame.new(pivot)
	motor.C1 = part1.CFrame:Inverse() * CFrame.new(pivot)
	motor.Parent = part1
end

model.PrimaryPart = bones.HumanoidRootPart
model.WorldPivot = CFrame.new() -- pivot on the ground between the feet

local humanoid = Instance.new("Humanoid")
humanoid.RigType = Enum.HumanoidRigType.R15
humanoid.AutomaticScalingEnabled = false
humanoid.HipHeight = HIP_HEIGHT
humanoid.RequiresNeck = false
humanoid.BreakJointsOnDeath = false
humanoid.DisplayDistanceType = Enum.HumanoidDisplayDistanceType.None
humanoid.HealthDisplayType = Enum.HumanoidHealthDisplayType.AlwaysOff
humanoid.WalkSpeed = 14
Instance.new("Animator").Parent = humanoid
humanoid.Parent = model

local camera = workspace.CurrentCamera
local spot = camera and camera.Focus.Position or Vector3.zero
model:PivotTo(CFrame.new(spot.X, spot.Y, spot.Z))
model.Parent = workspace

local selection = game:GetService("Selection")
pcall(function()
	selection:Set({ model })
end)
print("Deer built:", #model:GetDescendants(), "instances")
""".rstrip("\n"))
    with open(path, "w") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    rig = build()
    if "--json" in sys.argv:
        write_json(rig, os.path.join(HERE, "deer.json"))
    write_rbxmx(rig, os.path.join(HERE, "Deer.rbxmx"))
    write_lua(rig, os.path.join(HERE, "BuildDeer.lua"))
    n = sum(1 for _ in all_pieces(rig))
    print(f"{n} parts, {len(rig.joints)} Motor6Ds")
