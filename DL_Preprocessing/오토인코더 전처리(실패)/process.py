import pandas as pd
from json import dumps,loads
from threading import Thread,Lock
from copy import deepcopy

id_locks={pid: Lock() for pid in range(1, 49689)}
cache_lock=Lock()
count_lock=Lock()
last_json_lock=Lock()


jsons=r"product_jsons/"

# for i in range(1,49689):    
#     with open(jsons+f"{i}.json","w") as f:
#         temp={j:0 for j in range(1,49689) if j!=i}
#         f.write(dumps(temp))

MAX_RAM=49688 # MAX_RAM*572KB
JSONs={i:None for i in range(1,49689)}
last_json_ids=[]
count=0
prior_id=None
def readJson(id):
    global count
    global prior_id
    with cache_lock:
        data=JSONs[id]
        if data:
            return deepcopy(data)
    if count>=MAX_RAM:    
        with open(jsons+f"{id}.json","r") as f:
            DATA= loads(f.read())
        latest_json=None
        with cache_lock:
            with last_json_lock:
                if id not in last_json_ids:
                    last_json_ids.append(id)
                if count>=MAX_RAM:
                    latest=last_json_ids.pop(0)
                if JSONs[latest]:
                    latest_json=deepcopy(JSONs[latest])
                JSONs[id]=DATA
                JSONs[latest]=None
                result=deepcopy(JSONs[id])
        if latest_json!=None:
            with open(jsons+f"{latest}.json","w") as f:
                f.write(dumps(latest_json))
        return result
    else:
        with open(jsons+f"{id}.json","r") as f:
            data=loads(f.read())
        with count_lock:
            count+=1
        with cache_lock:
            JSONs[id]= data
        return data
def saveJson(id,JSON):
    with cache_lock:
        data=JSONs[id]
        if data:
            JSONs[id]=JSON
            return
    with open(jsons+f"{id}.json","w") as f:
        f.write(dumps(JSON))
prior=r"archive\order_products__prior.csv"
train=r"archive\order_products__train.csv"

df_prior=pd.read_csv(prior)
df=df_prior[["order_id","product_id"]]
d=df.groupby('order_id')['product_id'].apply(list).to_dict()

ThreadN=32

class PThread(Thread):
    def __init__(self,t_id,keys):
        super().__init__()
        self.keys=keys
        self.t_id=t_id
    def run(self):
        for idx,key in enumerate(self.keys):
            arr=d[key]
            if self.t_id==0:
                print(f"0 thread: {idx/len(self.keys)*100:.2f}%")
            for id in arr:
                with id_locks[id]:
                    id=int(id)
                    x=readJson(id)
                    for val in [k for k in arr if k!=id]:
                        x[str(val)]+=1
                    saveJson(id,x)
if __name__=="__main__":
    Size=len(d)
    datas_per_thread=int(Size/ThreadN)
    threads:list[Thread]=[None]*(ThreadN)
    extra_thread=None
    KEYS=list(d.keys())
    offset=0
    for i in range(ThreadN):
        thread=PThread(i,KEYS[offset:datas_per_thread+offset])
        threads[i]=thread
        offset+=datas_per_thread
    for thread in threads:
        thread.start()
    print("start")
    if offset!=Size:
        extra_thread=PThread(ThreadN,KEYS[offset:Size])
        extra_thread.start()
        extra_thread.join()
    for thread in threads:
        thread.join()