# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Read xclbin PDI metadata/hashes in bounded chunks, without native tools."""

import hashlib
import struct
import uuid
import xml.etree.ElementTree as ET
from pathlib import Path


def inspect_xclbin(path):
    """Inventory each PDI and map kernel names to its UUID and byte count."""
    path = Path(path)
    total = path.stat().st_size
    with path.open("rb") as stream:

        def read(offset, size):
            if offset < 0 or size < 0 or offset + size > total:
                raise ValueError("xclbin reference is outside the file")
            stream.seek(offset)
            data = stream.read(size)
            if len(data) != size:
                raise ValueError("truncated xclbin")
            return data

        head = read(0, 456)
        if head[:8] != b"xclbin2\0":
            raise ValueError("expected xclbin2")
        count = struct.unpack_from("<I", head, 448)[0]
        if count > 65536:
            raise ValueError("invalid section count")
        sections = {}
        for i in range(count):
            kind, _, offset, size = struct.unpack("<I16s4xQQ", read(456 + 40 * i, 40))
            if offset + size > total:
                raise ValueError("section is outside xclbin")
            sections[kind] = offset, size
        offset, size = sections[32]

        def part(at, length):
            if at < 0 or at + length > size:
                raise ValueError("PDI metadata reference is outside AIE_PARTITION")
            return read(offset + at, length)

        n, ptr = struct.unpack("<II", part(120, 8))
        width = struct.unpack("<H", part(32, 2))[0]
        nc, cp = struct.unpack("<II", part(40, 8))
        columns = [struct.unpack("<H", part(cp + 2 * i, 2))[0] for i in range(nc)]
        pdis, by_id = [], {}
        for i in range(n):
            desc = part(ptr + 96 * i, 96)
            length, start, groups, gptr = struct.unpack_from("<IIII", desc, 16)
            digest = hashlib.sha256()
            for at in range(0, length, 1024 * 1024):
                digest.update(part(start + at, min(1024 * 1024, length - at)))
            row = {
                "uuid": str(uuid.UUID(bytes=desc[:16])),
                "bytes": length,
                "sha256": digest.hexdigest(),
                "groups": [],
                "kernel_ids": [],
            }
            for j in range(groups):
                group = part(gptr + 96 * j, 96)
                pid, nk, kp = struct.unpack_from("<QII", group, 8)
                ids = [struct.unpack("<Q", part(kp + 8 * k, 8))[0] for k in range(nk)]
                row["groups"].append({"pdi_id": pid, "kernel_ids": ids})
                row["kernel_ids"].extend(ids)
                for kid in ids:
                    if kid in by_id:
                        raise ValueError("kernel id maps to multiple PDIs")
                    by_id[kid] = row
            pdis.append(row)
        xml_at, xml_size = sections[2]
        if xml_size > 16 * 1024 * 1024:
            raise ValueError("unexpectedly large kernel XML")
        root = ET.fromstring(read(xml_at, xml_size).rstrip(b"\0"))
        kernels = {}
        for kernel in root.findall("./platform/device/core/kernel"):
            ext = kernel.find("extended-data")
            if ext is None or "dpu_kernel_id" not in ext.attrib:
                continue
            kid = int(ext.attrib["dpu_kernel_id"], 0)
            pdi = by_id[kid]
            kernels[kernel.attrib["name"]] = {
                "num_args": len(kernel.findall("arg")),
                "kernel_id": kid,
                "pdi_uuid": pdi["uuid"],
                "pdi_bytes": pdi["bytes"],
                "pdi_sha256": pdi["sha256"],
            }
    return {
        "path": str(path),
        "file_bytes": total,
        "xclbin_uuid": str(uuid.UUID(bytes=head[416:432])),
        "pdis": pdis,
        "kernels": kernels,
        "partition": {"column_width": width, "start_columns": columns},
        "note": "PDI size/hash is measured; physical reloads require the switch probe, not metadata alone",
    }


def step_inventory(steps):
    """Attach PDI identities to saved descriptions or live graph steps."""
    images, rows = {}, []
    for step in steps:
        path = str(step["xclbin"])
        if path not in images:
            images[path] = inspect_xclbin(path)
        kernel = step["kernel"].split(":")[0]
        rows.append({**step, **images[path]["kernels"][kernel]})
    transitions = {}
    for name in ("kernel", "pdi_uuid", "pdi_sha256"):
        transitions[name] = sum(
            (a["xclbin"], a[name]) != (b["xclbin"], b[name])
            for a, b in zip(rows, rows[1:])
        )
    return {
        "images": list(images.values()),
        "steps": rows,
        "adjacent_transitions": transitions,
    }


def identical_pair(inventory):
    """Find a real adjacent same-image pair with equal bytes, unequal UUIDs."""
    for a, b in zip(inventory["steps"], inventory["steps"][1:]):
        if (
            a["xclbin"] == b["xclbin"]
            and a.get("pdi_sha256")
            and a["pdi_sha256"] == b.get("pdi_sha256")
            and a["pdi_uuid"] != b["pdi_uuid"]
        ):
            return [a["index"], b["index"]]
    return None
