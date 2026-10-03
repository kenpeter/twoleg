import struct, glob
from collections import defaultdict

base="/home/kenpeter/work/twoleg/training/logs/rsl_rl/velocity/2026-10-01_08-39-53_velocity/"
f=glob.glob(base+"events.out.tfevents.*")[0]
d=open(f,'rb').read()

off=0; n=len(d); recs=[]
while off+12<=n:
    ln=struct.unpack('<Q', d[off:off+8])[0]
    off+=12
    if off+ln>n: break
    recs.append(d[off:off+ln]); off+=ln+4

def varint(buf, i):
    val=0; shift=0
    while True:
        b=buf[i]; i+=1; val|=(b&0x7f)<<shift; shift+=7
        if not (b&0x80): break
    return val, i

def parse_summary(sub):
    vals=[]; i=0; L=len(sub)
    while i<L:
        key=sub[i]; i+=1; field=key>>3; wire=key&7
        if wire==2:
            ln,i=varint(sub,i)
            s=sub[i:i+ln]; i+=ln
            if field==1:
                tag=None; simple=None; j=0; M=len(s)
                while j<M:
                    k=s[j]; j+=1; f2=k>>3; w2=k&7
                    if w2==2:
                        l2,j=varint(s,j)
                        ss=s[j:j+l2]; j+=l2
                        if f2==1: tag=ss.decode('utf-8','replace')
                    elif w2==5:
                        if f2==2: simple=struct.unpack('<f', s[j:j+4])[0]
                        j+=4
                    elif w2==0:
                        l2,j=varint(s,j)
                        if f2==2: simple=l2
                    else: break
                if tag is not None: vals.append((tag, simple))
        else: break
    return vals

# last value per tag, in file order (chronological)
last={}
alltags=set()
for rec in recs:
    # find summary submessage: scan for field with wire 2 tag that contains a Value
    # simpler: locate summary by searching for 0x0a (field1 wire2) string "..." then 0x15/0x09 simple
    # Instead, find tag strings directly
    import re
    for m in re.finditer(rb'[ -~]{3,}', rec):
        s=m.group()
        if b'/' in s or b'_' in s:
            # candidate tag, find its numeric value after
            pass
    # Use targeted parse: event may have summary at field 1,3,5. Try each top-level wire2
    i=0; L=len(rec)
    while i<L:
        key=rec[i]; i+=1; field=key>>3; wire=key&7
        if wire==2:
            ln,i=varint(rec,i)
            sub=rec[i:i+ln]; i+=ln
            # detect if sub looks like a Summary.Value (contains a string tag)
            if b'\x0a' in sub and (b'\x15' in sub or b'\t' in sub):
                for tag,val in parse_summary(sub):
                    last[tag]=val
                    alltags.add(tag)
        elif wire==1:
            i+=8
        elif wire==0:
            _,i=varint(rec,i)
        else:
            break

print("ALL TAGS:", len(alltags))
kept=['reward','episode','track','mean_ep','command','error','lin_vel','ang_vel','torso','height','base','orient','upright','pose','dof','action']
for tag in sorted(alltags):
    tl=tag.lower()
    if any(k in tl for k in kept):
        print(f"{tag} = {last[tag]}")
