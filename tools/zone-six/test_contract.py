#!/usr/bin/env python3
"""Synthetic HTTP contract checks: no cloud calls, private models or user audio."""
import argparse, asyncio, importlib, io, json, math, struct, sys, tempfile, wave, zipfile
from pathlib import Path

def test(root):
    sys.path[:0]=[str(root),str(root/'web/backend')]
    import service
    settings=service.Settings();settings.turnstile_secret='';settings.proxy_secret='';settings.team_code='';settings.per_ip_per_day=100
    submitted=[];store=service.MemoryStore()
    with tempfile.TemporaryDirectory() as td:
        app=service.create_app(Path(td),store,lambda jid: submitted.append(jid) or 'local',settings=settings,cancel=lambda jid:None)
        class Client:
            def request(self,method,path,data=b'',headers=None):
                messages=[];sent=False
                async def receive():
                    nonlocal sent
                    if not sent:sent=True;return {'type':'http.request','body':data,'more_body':False}
                    await asyncio.sleep(60)
                async def send(m):messages.append(m)
                scope={'type':'http','asgi':{'version':'3.0'},'http_version':'1.1','scheme':'http','method':method,'path':path,'raw_path':path.encode(),'query_string':b'','headers':[(k.encode(),v.encode()) for k,v in (headers or {}).items()],'client':('127.0.0.1',10000),'server':('localhost',80)}
                asyncio.run(app(scope,receive,send))
                status=next(m['status'] for m in messages if m['type']=='http.response.start')
                body=b''.join(m.get('body',b'') for m in messages if m['type']=='http.response.body')
                return status,json.loads(body or b'{}')
        client=Client()
        assert client.request('GET','/api/limits')[0]==200
        audio=io.BytesIO()
        with wave.open(audio,'wb') as w:
            w.setnchannels(1);w.setsampwidth(2);w.setframerate(22050)
            w.writeframes(b''.join(struct.pack('<h',int(4000*math.sin(n*2*math.pi*100/22050))) for n in range(22050)))
        boundary='zone-six-test'
        body=('--'+boundary+'\r\nContent-Disposition: form-data; name="file"; filename="synthetic.wav"\r\nContent-Type: audio/wav\r\n\r\n').encode()+audio.getvalue()+('\r\n--'+boundary+'--\r\n').encode()
        status,result=client.request('POST','/api/jobs',body,{'content-type':'multipart/form-data; boundary='+boundary})
        assert status==200,result
        jid=result['id'];assert submitted==[jid]
        assert client.request('GET','/api/jobs/'+jid)[0]==200
        assert client.request('POST','/api/jobs/'+jid+'/cancel')[0]==200
        assert client.request('GET','/api/jobs/'+jid)[1]['status'] in ['cancelled','error']
    print('Upload, limits, polling and cancellation contract passed')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source-zip',type=Path,required=True);a=p.parse_args()
    with tempfile.TemporaryDirectory() as td:
        with zipfile.ZipFile(a.source_zip) as z:z.extractall(td)
        test(Path(td)/'source')
