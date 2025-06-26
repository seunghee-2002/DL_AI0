from multiprocessing import Process, Manager
import pandas as pd
import os

WORK_SIZE=16
SP=1748
EP=49688
MAIN_DIR=r"C:\Users\qw803\DL"
prior=r"archive\order_products__prior.csv"
train=r"archive\order_products__train.csv"
ADDR=prior

def worker(p_id,DIR,DICT):
    from json import dumps
    location=DIR+f"/{p_id}.json"
    JSON={i:0 for i in range(1,49689) if i!=p_id}
    for key in DICT:
        arr=DICT[key]
        if p_id not in arr:
            continue
        for other in arr:
            if other!=p_id:JSON[other]+=1
    with open(location,"w") as f:
        f.write(dumps(JSON))
if __name__=="__main__":
    DIR=MAIN_DIR+"/"+ADDR.split("__")[1].split(".")[0]
    if not os.path.isdir(DIR):
        os.mkdir(DIR)
    df=pd.read_csv(ADDR)[["order_id","product_id"]]
    data=df.groupby('order_id')['product_id'].apply(list).to_dict()
    offset=0
    START=SP-WORK_SIZE
    END=SP
    print("start!!")
    with Manager() as manager:
        while END!=EP+1:
            processes:list[Process]=[]
            START=(START+WORK_SIZE)
            END=min(END+WORK_SIZE,EP+1)
            for p_id in range(START,END):
                p=Process(target=worker,args=(p_id,DIR,data))
                processes.append(p)
            for p in processes:
                p.start()
            for p in processes:
                p.join()
            print(f"{END-1}번까지 완료!!")