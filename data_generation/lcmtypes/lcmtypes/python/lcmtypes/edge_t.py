"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

import lcmtypes.trajectory_t

import lcmtypes.vertex_t

class edge_t(object):
    __slots__ = ["vertex_src", "vertex_dst", "trajectory"]

    def __init__(self):
        self.vertex_src = lcmtypes.vertex_t()
        self.vertex_dst = lcmtypes.vertex_t()
        self.trajectory = lcmtypes.trajectory_t()

    def encode(self):
        buf = BytesIO()
        buf.write(edge_t._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        assert self.vertex_src._get_packed_fingerprint() == lcmtypes.vertex_t._get_packed_fingerprint()
        self.vertex_src._encode_one(buf)
        assert self.vertex_dst._get_packed_fingerprint() == lcmtypes.vertex_t._get_packed_fingerprint()
        self.vertex_dst._encode_one(buf)
        assert self.trajectory._get_packed_fingerprint() == lcmtypes.trajectory_t._get_packed_fingerprint()
        self.trajectory._encode_one(buf)

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != edge_t._get_packed_fingerprint():
            raise ValueError("Decode error")
        return edge_t._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = edge_t()
        self.vertex_src = lcmtypes.vertex_t._decode_one(buf)
        self.vertex_dst = lcmtypes.vertex_t._decode_one(buf)
        self.trajectory = lcmtypes.trajectory_t._decode_one(buf)
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if edge_t in parents: return 0
        newparents = parents + [edge_t]
        tmphash = (0x1fae492d71eedf94+ lcmtypes.vertex_t._get_hash_recursive(newparents)+ lcmtypes.vertex_t._get_hash_recursive(newparents)+ lcmtypes.trajectory_t._get_hash_recursive(newparents)) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff)  + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if edge_t._packed_fingerprint is None:
            edge_t._packed_fingerprint = struct.pack(">Q", edge_t._get_hash_recursive([]))
        return edge_t._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

