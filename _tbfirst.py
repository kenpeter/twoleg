import struct, glob
from collections import defaultdict

base="/home/kenpeter/work/twoleg/training/logs/rsl_rl/velocity/2026-10-01_08-39-53_velocity/"
f=glob.glob(base+"events.out.tfevents.*")[0]
d=open(f,'rb').read()
off=0; n=len(d); recs=[]
while off+12<=n:
    ln=struct.unpack('<Q', d[off:off+8])[0]; off+=12
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

first={}; last={}
for rec in recs:
    i=0; L=len(rec)
    while i<L:
        key=rec[i]; i+=1; field=key>>3; wire=key&7
        if wire==2:
            ln,i=varint(rec,i); sub=rec[i:i+ln]; i+=ln
            if b'\x0a' in sub and (b'\x15' in sub or b'\t' in sub):
                for tag,val in parse_summary(sub):
                    if tag not in first: first[tag]=val
                    last[tag]=val
        elif wire==1: i+=8
        elif wire==0:
            _,i=varint(rec,i)
        else: break

for t in ['Train/mean_reward','Train/mean_episode_length','Episode_Reward/track_linear_velocity','Episode_Reward/upright','Episode_Termination/fell_over','Episode_Reward/pose']:
    print(t,'FIRST=',first.get(t),'LAST=',last.get(t))
