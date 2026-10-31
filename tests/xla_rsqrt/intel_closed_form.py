import re, numpy as np
import urllib.request
src = urllib.request.urlopen("https://raw.githubusercontent.com/bochs-emu/Bochs/master/bochs/cpu/sse_rcp.cc").read().decode()
def tbl(name):
    m = re.search(name + r"\[1024\] = \{(.*?)\};", src, re.S)
    return np.array([int(x, 16) for x in re.findall(r"0x[0-9a-fA-F]+", m.group(1))])
t0, t1 = tbl("rsqrt_table0"), tbl("rsqrt_table1")
print("sizes", len(t0), len(t1), "first", hex(t0[0]), hex(t1[0]))
# Bochs: packFloat32(0, exp, table << 8): mantissa(23 bits) = table << 8 ; table is 15 bits (0x7ff0.. -> 15 bit)
for name, t, scale in (("table1 (x in [1,2))", t1, 1.0), ("table0 (x in [2,4))", t0, 2.0)):
    i = np.arange(1024)
    x = (1.0 + (i + 0.5) / 1024.0) * scale          # 1.yyyyyyyyyy1
    v = 1.0 / np.sqrt(x)                              # in (0.5, 1]
    # value represented by the table entry: mantissa m = table<<8 in 2^-23 units, value = (1 + m/2^23) * 2^(e-127) with e = 126
    ent = (1.0 + (t.astype(np.float64) * 256.0) / 2.0 ** 23) * 0.5
    for q in (11, 12, 13):
        sc = 2.0 ** q
        rr = np.round(v * sc) / sc
        print("%-20s round 1/sqrt(mid) to %d bits: mismatches %d of 1024" % (name, q, int((rr != ent).sum())))
