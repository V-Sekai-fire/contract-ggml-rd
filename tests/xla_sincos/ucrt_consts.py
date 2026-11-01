import struct, sys
d = open(r"C:/Windows/System32/ucrtbase.dll", "rb").read()
pe = struct.unpack_from("<I", d, 0x3c)[0]
nsec = struct.unpack_from("<H", d, pe + 6)[0]
optsz = struct.unpack_from("<H", d, pe + 20)[0]
secs = []
for i in range(nsec):
    o = pe + 24 + optsz + 40 * i
    name = d[o:o+8].rstrip(b"\0").decode()
    vsz, va, rsz, rptr = struct.unpack_from("<IIII", d, o + 8)
    secs.append((name, va, vsz, rsz, rptr))
def off(rva):
    for n, va, vsz, rsz, rp in secs:
        if va <= rva < va + max(vsz, rsz): return rp + rva - va
    raise KeyError(hex(rva))
def dbl(a, n=1): return [struct.unpack_from("<d", d, off(a - 0x180000000) + 8 * i)[0] for i in range(n)]
def raw(a, n=16): return d[off(a - 0x180000000): off(a - 0x180000000) + n].hex()
if __name__ == "__main__":
    for a in (0x18010bef0, 0x18010bef8):
        print(hex(a), "float", struct.unpack_from("<f", d, off(a - 0x180000000))[0], raw(a, 4))
    for a in (0x18010be40, 0x18010aea0, 0x18010aea8, 0x18010ae98, 0x18010ae90, 0x18010be20, 0x18010be10, 0x18010be30, 0x18010be50, 0x18010bea0,
              0x18010bdf0, 0x18010ae60, 0x18010ae78, 0x18010ae80, 0x18010ae70, 0x18010ae68, 0x18010bdd0, 0x18010be90, 0x18010be00):
        print(hex(a), raw(a), [repr(x) for x in dbl(a, 2)])
