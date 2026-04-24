import serial, time, struct
PORT='COM5'; BAUD=2000000
with serial.Serial(PORT,BAUD,timeout=0.1) as s:
    time.sleep(0.3); s.reset_input_buffer()
    s.write(b'O\r'); time.sleep(0.15); s.read(10)
    print(">>> SPIN THE WHEEL NOW — watching 15 seconds <<<")
    buf=b''; deadline=time.time()+15; max_pos=0.0; n=0
    while time.time()<deadline:
        chunk=s.read(64)
        if chunk:
            buf+=chunk
            while b'\r' in buf:
                lb,buf=buf.split(b'\r',1)
                ln=lb.decode('ascii',errors='ignore').strip()
                if not ln or ln[0]!='t' or len(ln)<5: continue
                try:
                    fid=int(ln[1:4],16); dlc=int(ln[4],16)
                    if (fid>>5)==0 and (fid&0x1F)==0x09 and dlc>=8:
                        pos,vel=struct.unpack_from('<ff',bytes.fromhex(ln[5:5+16]))
                        max_pos=max(max_pos,abs(pos)); n+=1
                        if abs(pos)>0.001 or n%50==0:
                            print(f"  pos={pos:+.4f} turns  vel={vel:+.4f} t/s")
                except Exception:
                    pass
    s.write(b'C\r')
    print(f"\n{n} frames. Max pos = {max_pos:.4f} turns  ({'NON-ZERO OK' if max_pos>0.001 else 'still zero'})")
